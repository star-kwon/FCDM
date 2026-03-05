
"""
Code adapted from https://github.com/chuanyangjin/fast-DiT
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../train_gen'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))


import torch
import torch.distributed as dist
from models.fcdm_models import FCDM_models
from models.diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse

import yaml
from omegaconf import OmegaConf
from pathlib import Path

def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config

def create_npz_from_sample_folder(sample_dir, max_samples=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    sample_dir = Path(sample_dir)
    files = sorted(sample_dir.glob("*.png"))
    if max_samples is not None:
        files = files[:max_samples]

    samples = [
        np.asarray(Image.open(fp), dtype=np.uint8)
        for fp in tqdm(files, desc="Loading samples")
    ]
    arr = np.stack(samples, axis=0)

    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr=arr)
    del samples, arr
    return npz_path

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.)/2.
    x = x.permute(1,2,0).numpy()
    x = (255*x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x

def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Load model:
    latent_size = args.image_size // 8
    model = FCDM_models[args.model](
        num_classes=args.num_classes,
        in_channels=args.in_channels
    ).to(device)
    # Load a pre-trained checkpoint:
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["ema"])
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))

    if args.hf_model_dir is not None:
        vae = AutoencoderKL.from_pretrained(
            args.hf_model_dir,
            local_files_only=True
        ).to(device)
    else:
        vae = AutoencoderKL.from_pretrained(args.hf_model_name).to(device)

    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    # Create folder to save samples:
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.image_size}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}"

    if args.ddpm:
        folder_name += f"-ddpm"

    sample_folder_dir = f"{args.sample_dir}/{folder_name}"

    exp_dir_name = args.sample_dir
    fid_dir      = os.path.join(exp_dir_name, "fids")

    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        os.makedirs(fid_dir, exist_ok=True)

        print(f"Saving .png samples at {sample_folder_dir}")
        print(f"Saving metrics at {fid_dir}")

    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    for _ in pbar:
        # Sample inputs:
        z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
        y = torch.randint(0, args.num_classes, (n,), device=device)

        # Setup classifier-free guidance:
        if using_cfg:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * n, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
            sample_fn = model.forward_with_cfg
        else:
            model_kwargs = dict(y=y)
            sample_fn = model.forward

        # Sample images:
        if args.ddpm:
            samples = diffusion.p_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
            )
        else:
            samples = diffusion.ddim_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
            )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

        if args.vae_ckpt is not None:

            samples = vae.decode(samples / args.vae_scaling_factor)

        else:

            samples = vae.decode(samples / args.vae_scaling_factor).sample
            samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        # Save samples to disk as individual .png files
        for i, sample in enumerate(samples):
            index = i * dist.get_world_size() + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        sample_batch = create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(FCDM_models.keys()), default="FCDM-XL")
    parser.add_argument("--vae-ckpt",  type=str,  default=None)
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--ddpm", type=bool, default=False)
    parser.add_argument("--vae-scaling-factor", type=float, default=0.18215)
    parser.add_argument("--hf-model-name", type=str, default="stabilityai/sd-vae-ft-ema")
    parser.add_argument("--hf-model-dir", type=str, default=None)
    parser.add_argument("--in-channels", type=int, default=4)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--num-fid-samples", type=int, default=50000)
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a FCDM checkpoint.")
    parser.add_argument("--per_proc_batch_size", "--per-proc-batch-size", type=int, default=64,
                        help="Number of samples to generate *per* GPU/process in each forward pass.")
    args = parser.parse_args()
    main(args)