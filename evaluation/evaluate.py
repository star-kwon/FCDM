"""
Code adapted from https://github.com/chuanyangjin/fast-DiT
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../train_gen'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from tqdm import tqdm
import os
import argparse
from evaluator import Evaluator
import tensorflow.compat.v1 as tf

def calculate_metrics(ref_batch, sample_batch, fid_path):
    config = tf.ConfigProto(
        allow_soft_placement=True  
    )
    config.gpu_options.allow_growth = True
    evaluator = Evaluator(tf.Session(config=config))

    evaluator.warmup()

    ref_acts = evaluator.read_activations(ref_batch)
    ref_stats, ref_stats_spatial = evaluator.read_statistics(ref_batch, ref_acts)

    sample_acts = evaluator.read_activations_arr(sample_batch)
    sample_stats, sample_stats_spatial = evaluator.read_statistics(sample_batch, sample_acts)

    with open(fid_path, 'w') as fd:

        fd.write("Computing evaluations...\n")
        fd.write(f"Inception Score:{evaluator.compute_inception_score(sample_acts[0])}\n" )
        fd.write(f"FID:{sample_stats.frechet_distance(ref_stats)}\n")
        fd.write(f"sFID:{sample_stats_spatial.frechet_distance(ref_stats_spatial)}\n")
        prec, recall = evaluator.compute_prec_recall(ref_acts[0], sample_acts[0])
        fd.write(f"Precision:{prec}\n")
        fd.write(f"Recall:{recall}\n")

def main(args):
    """
    Run sampling.
    """
    print(tf.config.list_physical_devices('GPU'))

    fid_path = args.output_fid

    sample_batch = args.sample_batch
    calculate_metrics(args.ref_batch, sample_batch, fid_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-ckpt",  type=str,  default=None)
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--vae-scaling-factor", type=float, default=0.18215)
    parser.add_argument("--ref-batch", type=str, default="VIRTUAL_imagenet256_labeled.npz")
    parser.add_argument("--sample-batch", type=str, default="stabilityai/sd-vae-ft-ema")
    parser.add_argument("--hf-model-name", type=str, default="stabilityai/sd-vae-ft-ema")
    parser.add_argument("--wavelets", type=str, default=False)
    parser.add_argument("--diff-proj", type=str, default=False)
    parser.add_argument("--gaussian-registers", type=str, default=False)
    parser.add_argument("--in-channels", type=int, default=4)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--num-fid-samples", type=int, default=1000)
    parser.add_argument("--cfg-scale",  type=float, default=1.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--output_fid", type=str, default=None,
                        help="Optional path to a FCDM checkpoint.")
    args = parser.parse_args()
    main(args)