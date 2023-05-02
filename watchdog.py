import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

import pexpect
import torch
from omegaconf import OmegaConf

from train import find_latest_ckpt


def main():
    parser = ArgumentParser(
        "Training Watchdog",
        description=(
            "Runs training script and automatically restarts if it crashes.\n"
            "Specify options train.py after typing --\n"
            "Example: to monitor the training 'python train.py n_steps=1000 bsdf=principled learning_rate=0.005',\n"
            "run: 'python watchdog.py -- n_steps=1000 bsdf=principled learning_rate=0.005'."
        ),
    )
    parser.add_argument("--max_retries", type=int, default=100, help="No longer restarts after this number of failures.")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout for training")

    argv = sys.argv[1:]

    extra_args = []
    if "--" in argv:
        split_idx = argv.index("--")
        extra_args = argv[split_idx+1:]
        argv = argv[:split_idx]

    args = parser.parse_args(argv)
    max_retries = args.max_retries
    print(max_retries)

    # find output folder
    proc = subprocess.run(
        [
            "python",
            "train.py",
            *extra_args,
            "is_watchdog_init=true",
        ],
        capture_output=True,
        check=True,
    )
    watchdog_info = {
        line.split(":")[1]: line.split(":")[2]
        for line in proc.stdout.decode("utf-8").split("\n") if line.startswith("watchdog:")
    }
    out_folder = watchdog_info.get("out_root")
    if out_folder is None:
        print("Failed to get output folder from train.py output, check the command without watchdog first.")
        return

    print(f"Training output: {out_folder}")
    out_folder = Path(out_folder)
    out_name = out_folder.name

    # get target steps
    train_cfg = OmegaConf.load(out_folder / ".hydra/config.yaml")
    n_steps = train_cfg.n_steps
    print(f"Target training steps: {n_steps}")

    n_retries = 0
    while True:
        n_retries += 1
        if n_retries > max_retries:
            print("Max retires reached, aborting")
            break

        print(f"====== Watchdog Attempt {n_retries} ======")
        child = pexpect.spawn(
            "python",
            ["train.py", *extra_args, f"out_dir={out_name}"],
            timeout=args.timeout,
            logfile=sys.stdout,
            encoding="utf-8",
        )

        # wait until child is done
        child.expect(["drjit-autodiff: variable leak detected", pexpect.EOF, 'Dr.Jit exhausted the available memory'])

        # check if child exits
        idx = child.expect([pexpect.EOF, pexpect.TIMEOUT], timeout=2)
        if idx == 1:
            # child timed out, kill it
            print("Child seems stuck, terminating")
            child.terminate()

        # check if training has completed
        ckpt_file = find_latest_ckpt(out_folder / "checkpoints")
        if ckpt_file is None:
            print("No checkpoint has been saved, retrying")
            continue
        ckpt = torch.load(ckpt_file, map_location="cpu")
        latest_step = ckpt["step"]
        del ckpt

        if latest_step >= n_steps:
            print(f"Latest step is {latest_step}, ending")
            break
        print(f"Latest step is {latest_step}, retrying")


if __name__ == "__main__":
    main()
