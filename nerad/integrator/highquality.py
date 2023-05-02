import gc
from typing import Tuple, Union

import drjit as dr
import mitsuba as mi
import torch
from tqdm import tqdm

from nerad.integrator import register_integrator


@register_integrator("highquality")
class HighQuality(mi.SamplingIntegrator):
    """
    This class takes care of rendering an image, but instead of rendering the whle pixels all at once, just rendering small blocks at a time and iteratively repeate the same for all blocks until the whole image is rendererd.
    Integrator: mi.SamplingIntegrator
        This is the integrator that takes care of rendering the blocks.
    """

    def __init__(self, block_size: int, integrator: mi.SamplingIntegrator):
        super().__init__(mi.Properties())
        self.block_size = block_size
        self.integrator = integrator

    def prepare(self,
                sensor: mi.Sensor,
                block: mi.ImageBlock,
                seed: int = 0,
                spp: int = 0,
                ):
        """
        This method is another implementation of method pepare in mitsuba/src/python/python/ad/common.py
        The difference is that it prepares the sampler to sample for a wavefront size that matches the block size not the film size.
        """

        film_on_sensor = sensor.film()
        sampler = sensor.sampler()

        if spp != 0:
            sampler.set_sample_count(spp)

        spp = sampler.sample_count()
        sampler.set_samples_per_wavefront(spp)

        block_size = block.size()

        if film_on_sensor.sample_border():
            raise NotImplementedError()

        wavefront_size = dr.prod(block_size) * spp

        if wavefront_size > 2**32:
            raise Exception(
                "The total number of Monte Carlo samples required by this "
                "rendering task (%i) exceeds 2^32 = 4294967296. Please use "
                "fewer samples per pixel or render using multiple passes."
                % wavefront_size)

        sampler.seed(seed, wavefront_size)

        return sampler, spp

    def render(self,
               scene: mi.Scene,
               sensor: Union[int, mi.Sensor] = 0,
               seed: int = 0,
               spp: int = 0,
               develop: bool = True,
               evaluate: bool = True) -> mi.TensorXf:
        """
        This method is another implementation of method render() in mitsuba/src/python/python/ad/common.py
        The difference is that it breaks down the rendering task into smaller comutations of blocks in the image instead of rendering it all at one pass.
        It uses the underlying integrator to render each block.
        """

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        # Prepare the spiral
        spiral = mi.Spiral(sensor.film().crop_size(), mi.ScalarVector2i(0, 0), self.block_size)
        sensor.film().prepare(self.integrator.aov_names())
        has_aov = len(self.integrator.aov_names()) > 0

        for i in tqdm(range(spiral.block_count())):
            block_offset, block_size, block_id = spiral.next_block()
            # Prepare an ImageBlock as specified by the film and block size
            block = sensor.film().create_block(block_size)
            block.set_offset(block_offset)

            # Disable derivatives in all of the following
            with dr.suspend_grad():
                with torch.no_grad():
                    # Prepare the film and sample generator for rendering
                    sampler, spp = self.prepare(
                        sensor=sensor,
                        block=block,
                        seed=(seed+1)*block_id,
                        spp=spp)
                    # Generate a set of rays starting at the sensor
                    ray, weight, pos = self.sample_rays(scene, sensor, sampler, block)
                    # Launch the Monte Carlo sampling process in primal mode
                    if issubclass(type(self.integrator), mi.CppADIntegrator):
                        L, valid, _ = self.integrator.sample(
                            mode=dr.ADMode.Primal,
                            scene=scene,
                            sampler=sampler,
                            ray=ray,
                            depth=mi.UInt32(0),
                            δL=None,
                            state_in=None,
                            reparam=None,
                            active=mi.Bool(True)
                        )
                    else:
                        L, valid, aov = self.integrator.sample(
                            scene,
                            sampler,
                            ray,
                            None,
                            active=mi.Bool(True))

                    # Only use the coalescing feature when rendering enough samples
                    # block.set_coalesce(block.coalesce() and spp >= 4)

                    # Accumulate into the image block
                    alpha = dr.select(valid, mi.Float(1), mi.Float(0))
                    if has_aov:
                        # Assumption: weight is always [1.0, 1.0, 1.0]
                        floatLs = [L[0], L[1], L[2], alpha, weight[0]]
                        all_channels = floatLs + aov
                        block.put(pos, all_channels)
                        del aov
                    else:
                        block.put(pos, ray.wavelengths, L * weight, alpha)

                    sampler.schedule_state()
                    dr.eval(block.tensor())

                    # Explicitly delete any remaining unused variables
                    del ray, weight, pos, L, valid, alpha
                    gc.collect()

                    # Perform the weight division and return an image tensor
                    sensor.film().put_block(block)
                    torch.cuda.empty_cache()
                    dr.flush_malloc_cache()

        primal_image = sensor.film().develop()
        dr.schedule(primal_image)
        if evaluate:
            dr.eval()
            dr.sync_thread()

        return primal_image

    def render_forward(self: mi.Integrator,
                       scene: mi.Scene,
                       params,
                       sensor: Union[int, mi.Sensor] = 0,
                       seed: int = 0,
                       spp: int = 0) -> mi.TensorXf:
        """
        This method is another implementation of method render() in mitsuba/src/python/python/ad/common.py
        The difference is that it breaks down the rendering task into smaller comutations of blocks in the image instead of rendering it all at one pass.
        It uses the underlying integrator to render each block.
        """

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        # Prepare the spiral
        spiral = mi.Spiral(sensor.film().crop_size(), mi.ScalarVector2i(0, 0), self.block_size)
        sensor.film().prepare(self.integrator.aov_names())
        has_aov = len(self.integrator.aov_names()) > 0

        for i in tqdm(range(spiral.block_count())):
            block_offset, block_size, block_id = spiral.next_block()
            # Prepare an ImageBlock as specified by the film and block size
            block = sensor.film().create_block(block_size)
            block.set_offset(block_offset)

            # Disable derivatives in all of the following
            with dr.suspend_grad():

                # Prepare the film and sample generator for rendering
                sampler, spp = self.prepare(
                    sensor=sensor,
                    block=block,
                    seed=(seed+1)*block_id,
                    spp=spp)
                # Generate a set of rays starting at the sensor
                ray, weight, pos = self.sample_rays(scene, sensor, sampler, block)
                # Launch the Monte Carlo sampling process in primal mode

                with dr.resume_grad():
                    dr.enable_grad(params)
                    dr.set_grad(params, 1)

                    if issubclass(type(self.integrator), mi.ad.common.ADIntegrator):
                        L, valid, _ = self.integrator.sample(
                            mode=dr.ADMode.Primal,
                            scene=scene,
                            sampler=sampler,
                            ray=ray,
                            depth=mi.UInt32(0),
                            δL=None,
                            state_in=None,
                            reparam=None,
                            active=mi.Bool(True)
                        )
                    else:
                        L, valid, aov = self.integrator.sample(
                            scene,
                            sampler,
                            ray,
                            None,
                            active=mi.Bool(True))

                    dr.forward_to(L)
                    L = dr.grad(L)
                    dr.disable_grad(params)

                # Only use the coalescing feature when rendering enough samples
                # block.set_coalesce(block.coalesce() and spp >= 4)

                # Accumulate into the image block
                alpha = dr.select(valid, mi.Float(1), mi.Float(0))
                if has_aov:
                    # Assumption: weight is always [1.0, 1.0, 1.0]
                    floatLs = [L[0], L[1], L[2], alpha, weight[0]]
                    all_channels = floatLs + aov
                    block.put(pos, all_channels)
                    del aov
                else:
                    block.put(pos, ray.wavelengths, L * weight, alpha)

                sampler.schedule_state()
                dr.eval(block.tensor())

                # Explicitly delete any remaining unused variables
                del ray, weight, pos, L, valid, alpha
                gc.collect()

                # Perform the weight division and return an image tensor
                sensor.film().put_block(block)
                torch.cuda.empty_cache()
                dr.flush_malloc_cache()

        grad_image = sensor.film().develop()
        dr.schedule(grad_image)

        return grad_image

    def sample_rays(
        self,
        scene: mi.Scene,
        sensor: mi.Sensor,
        sampler: mi.Sampler,
        block: mi.ImageBlock
    ) -> Tuple[mi.RayDifferential3f, mi.Spectrum, mi.Vector2f, mi.Float]:
        """
        This method is another implementation of method sample_rays() in mitsuba/src/python/python/ad/common.py
        The difference is that it prepares the samples for a block of rays instead of the whole image plane pixels.
        """

        block_size = block.size()
        rfilter = sensor.film().rfilter()
        border_size = rfilter.border_size()

        if sensor.film().sample_border():
            block_size += 2 * border_size
            raise NotImplementedError()

        spp = sampler.sample_count()

        # Compute discrete sample position
        idx = dr.arange(mi.UInt32, dr.prod(block_size) * spp)

        # Try to avoid a division by an unknown constant if we can help it
        log_spp = dr.log2i(spp)
        if 1 << log_spp == spp:
            idx >>= dr.opaque(mi.UInt32, log_spp)
        else:
            idx //= dr.opaque(mi.UInt32, spp)

        # Compute the position on the image plane
        pos = mi.Vector2i()
        pos.y = idx // block_size[0]
        pos.x = dr.fma(-block_size[0], pos.y, idx)

        if sensor.film().sample_border():
            pos -= border_size
            raise NotImplementedError()

        pos += mi.Vector2i(block.offset())

        # Cast to floating point and add random offset
        pos_f = mi.Vector2f(pos) + sampler.next_2d()

        # Re-scale the position to [0, 1]^2
        scale = dr.rcp(mi.ScalarVector2f(sensor.film().crop_size()))
        # offset = -mi.ScalarVector2f(block.offset()) * scale #TODO: check why this is wrong in the orignial mitsuba.ad.common.py
        pos_adjusted = pos_f * scale

        aperture_sample = mi.Vector2f(0.0)
        if sensor.needs_aperture_sample():
            aperture_sample = sampler.next_2d()

        time = sensor.shutter_open()
        if sensor.shutter_open_time() > 0:
            time += sampler.next_1d() * sensor.shutter_open_time()

        wavelength_sample = 0
        if mi.is_spectral:
            wavelength_sample = sampler.next_1d()

        with dr.resume_grad():
            ray, weight = sensor.sample_ray_differential(
                time=time,
                sample1=wavelength_sample,
                sample2=pos_adjusted,
                sample3=aperture_sample
            )

        # With box filter, ignore random offset to prevent numerical instabilities
        splatting_pos = mi.Vector2f(pos) if rfilter.is_box_filter() else pos_f
        return ray, weight, splatting_pos

    def to_string(self):
        return (
            "Highquality[\n"

            "]"
        )
