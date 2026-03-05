# Diffusion Models with ConvNeXt (FCDM)
<p align="left">
  <b>Official PyTorch Implementation</b>
</p>

![FCDM samples](visuals/visual.jpg)

This repository contains the PyTorch implementation (training, sampling, and model definitions) for **FCDM**, a paper exploring diffusion models with the ConvNeXt architecture. It supports both local and distributed (cluster) environments.

---

## 🛠 1. Installation & Setup

### 1.1. Environment Setup
First, create and activate the Conda environment:
```bash
conda env create -f environment.yml
conda activate fcdm
```

### 1.2. (Optional) Manual VAE Download for Offline Clusters

> 💡 **Note:** Some cluster environments restrict outgoing internet connections due to security policies, preventing automatic model downloads from Hugging Face. If you are in such an environment, follow these steps to manually download and transfer the SD-VAE (or EQVAE) snapshot.

**1. Download the snapshot locally:** Run the script below on your local machine. This creates a folder (e.g., `sd-vae-ft-ema/`) containing the necessary model files (`config.json`, `pytorch_model.bin`, etc.).
  ```bash
  python preparation/download_snapshot.py
  ```

**2. Transfer the files to your cluster:** Upload the downloaded folder to your remote cluster.
```bash
scp -r sd-vae-ft-ema/ user@your_cluster_address:/path/to/your_cluster_dir/
```

**3. Specify the snapshot path:** When launching any training or sampling scripts, append the `--hf-model-dir` argument to explicitly point to your transferred snapshot directory. (default is `None`).
```bash
python your_script.py --hf-model-dir /path/to/your_cluster_dir/sd-vae-ft-ema
```

---

## 🗂 2. Data Preparation

### 2.1. Extract Latent Representations
Extract the encoded features and corresponding labels from your dataset. These paths will be provided as `--feature-path` and `--label-path` during training.
  ```bash
  torchrun --nnodes=1 --nproc_per_node=4  preparation/extract_features.py \
      --data-path /path/to/imagenet \
      --features-path /path/to/latents
  ```

### 2.2. (Optional) Cache Latents to Zarr Format
If your cluster imposes a file number limit, we recommend storing the latents in Zarr format.
First, manually set the feature, label, and output paths in the script, then run:
  ```bash
  python preparation/save_zarr.py
  ```
Once cached, you can pass the Zarr path to any script using: `--zarr-path /path/to/cache.zarr`

---

## 🚀 3. Training
Run the following command to train FCDM on your precomputed latents:
  ```bash
  accelerate launch --mixed_precision bf16 --num_processes 4 train_gen/train.py \
    --model ACD-XL \
    --label-path /path/to/labels \
    --feature-path /path/to/features \
    --results-dir results
  ```
* **Optional Arguments:** `--hf-model-dir`, `--zarr-path`
> 💡 **Note:** We found that `--mixed_precision bf16` is more stable than `fp16`, especially when resuming training.

---

## 🎨 4. Sampling & Evaluation

### 4.1. Sample Images for Evaluation
Generate 50,000 samples for evaluation. This script will automatically save the generated images and a corresponding `.npz` file.
  ```bash
torchrun --nnodes=1 --nproc_per_node=4 train_gen/sample_ddp.py \
    --model ADC-XL \
    --num-fid-samples 50000 \
    --ckpt /path/to/checkpoint \
    --sample-dir samples \
    --ddpm True \
    --cfg-scale 1.0
  ```
* **Optional Arguments:** `--hf-model-dir`

### 4.2. Evaluation:
Before running evaluation, download the reference `.npz` file from [ADM's TensorFlow evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations) and place it in the appropriate directory. 

> 💡 **Note:** Evaluation requires a TensorFlow environment. We strongly recommend setting up a separate Conda environment for this to avoid potential dependency conflicts with PyTorch.

Run the evaluation:
  ```bash
  python train_gen/evaluate.py \
    --num-fid-samples 50000 \
    --sample-dir samples \
    --sample-batch /path/to/sample/sample.npz \
    --ref-batch /path/to/ref.npz \
    --output_fid /path/to/fid_output.txt
  ```

## Acknowledgments
We thank Farnood Salehi, Jingwei Tang, and Jakob Buhmann for helpful discussions. 
Taesung Kwon is supported by the NRF Sejong Science Fellowship.

This codebase borrows from existing diffusion repositories, most notably Meta's [DiT](https://github.com/facebookresearch/DiT) and OpenAI's [ADM](https://github.com/openai/guided-diffusion).

## License
The codes are licensed under CC-BY-NC. See [`LICENSE.txt`](LICENSE.txt) for details.