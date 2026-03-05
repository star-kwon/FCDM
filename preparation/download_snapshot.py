from huggingface_hub import snapshot_download

# this will create exactly this folder, with config.json inside
local_dir = snapshot_download(
    repo_id="stabilityai/sd-vae-ft-ema",
    local_dir="./sd-vae-ft-ema",
    resume_download=True
)