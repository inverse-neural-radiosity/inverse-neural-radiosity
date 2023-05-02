import drjit as dr
import mitsuba as mi


class ShapeSampler():
    def __init__(self, scene) -> None:
        self.valid_inds = self.compute_valid_sahpes(scene)
        self.sampler = mi.load_dict({'type': 'independent'})
        self.PCG = None

    def sample_on_shape(self, active, shape):
        sample_state = self.sampler.wavefront_size()

        if sample_state != len(active):
            print('error in sampler')

        pos_rnd = self.sampler.next_2d(active)
        dir_rnd = self.sampler.next_2d(active)

        pos_samples = shape.sample_position(time=0, sample=pos_rnd)

        si = mi.SurfaceInteraction3f(ps=pos_samples, wavelengths=[])
        si.initialize_sh_frame()
        si.shape = shape
        si.t = si.p[0]*0
        si.time = si.p[0]*0

        is_twosided = mi.has_flag(si.shape.bsdf().flags(), mi.BSDFFlags.BackSide)

        dir_samples = mi.warp.square_to_uniform_hemisphere(dir_rnd)

        samplingProb = mi.warp.square_to_uniform_hemisphere_pdf(dir_samples)*pos_samples.pdf
        dir_samples[is_twosided] = mi.warp.square_to_uniform_sphere(dir_rnd)
        samplingProb[is_twosided] = mi.warp.square_to_uniform_sphere_pdf(dir_samples)*pos_samples.pdf

        si.wi = dir_samples

        return si, samplingProb

    def rng(self, n):
        if (self.PCG is None):
            self.PCG = mi.PCG32(size=n)
        return self.PCG

    def sample_input(self, scene, n, seed):
        sample_on_shape = False

        self.sampler.set_sample_count(n)
        self.sampler.seed(seed, n)

        self.sampler.schedule_state()
        dr.eval()

        if (sample_on_shape):
            shapes = scene.shapes_dr()
            indices = self.sample_valid_shape_indices()

            shape = dr.gather(mi.ShapePtr, shapes, indices)
            active = ~dr.isnan(indices)
            si, prob = self.sample_on_shape(active, shape)
            to_ret = si
        else:
            pos = mi.Vector3f(self.sampler.next_1d(), self.sampler.next_1d(), self.sampler.next_1d())
            dir = mi.warp.square_to_uniform_sphere(self.sampler.next_2d())
            pos = pos*(scene.bbox().max - scene.bbox().min) + scene.bbox().min

            rays = mi.Ray3f(o=pos, d=dir)
            prob = mi.warp.square_to_uniform_sphere_pdf(dir)
            to_ret = rays

        return to_ret, prob

    def compute_valid_sahpes(self, scene):
        i = 0
        valid_inds, invalid_inds = [], []
        for sh in scene.shapes():
            try:
                sh.surface_area()
                valid_inds.append(i)
            except:
                invalid_inds.append(i)
            i += 1

        valid_inds = mi.UInt32(valid_inds)
        return valid_inds

    def sample_valid_shape_indices(self):
        max_len = len(self.valid_inds)
        random_indices_in_the_valid_array = dr.minimum(mi.UInt32(self.sampler.next_1d() * max_len), max_len-1)
        indices = dr.gather(mi.UInt32, self.valid_inds, random_indices_in_the_valid_array)
        return indices
