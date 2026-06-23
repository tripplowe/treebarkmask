# treebarkmask

Automated foreground masking of individual tree boles in field photographs, using
[Segment Anything 3 (SAM 3)](https://huggingface.co/facebook/sam3) with a
geometry-aware quality-control gate, packaged to run in Docker on an NVIDIA GPU.

The photos follow a standardized protocol — one tree bole filling most of the
frame, with variable background depending on forest and lighting. The goal is a
clean binary mask of the bole for every image, generated without per-image manual
prompting, with automatic triage of the few that don't come out clean.

## How it works

Each image flows through this pipeline:

```
image ──> [rembg pre-pass (optional, OFF by default)] ──> SAM 3 (text prompt) ──> geometric QC ──> masks/  or  review/
```

- **SAM 3** segments the bole from a fixed text prompt (`"tree trunk"`), so no
  clicking or per-image boxes. It returns one or more candidate instance masks
  with confidence scores.
- **Geometric QC** is the automation glue. Rather than trust the model blindly, it
  scores each mask against priors the acquisition protocol guarantees — a single
  connected blob, spanning the frame vertically, roughly centered, filling a
  sensible fraction of the frame, reasonably solid (not scattered). Masks that pass
  go to `masks/`; the rest go to `review/` for a human to check. The full QC metrics
  for every image are logged to a CSV so you can see *why* anything failed and tune
  the thresholds to your imagery.
- **rembg** is an optional cheap CPU pre-pass (a saliency-based background remover).
  The idea is to clear easy frames on the CPU and only escalate hard ones to the
  GPU. In practice, on bark imagery where the trunk fills the frame, saliency has
  nothing distinct to latch onto and rembg clears almost nothing, so it's **disabled
  by default**. It remains in the code; enable it with `--use-rembg` if your imagery
  has clean, separable foregrounds (see "When to enable rembg").

If you've used SAM before, the mental model is: SAM 3 does concept-prompted instance
segmentation, and everything around it here is batch orchestration plus an automatic
accept/reject rule grounded in the dataset's geometry.

## Repository contents

- `bole_mask_pipeline.py` — the pipeline (orchestration, QC, routing)
- `review_masks.py` — builds an HTML overlay gallery for visually checking masks
- `debug_sam3.py` — single-image SAM 3 harness for diagnosing setup issues
- `Dockerfile` — CUDA 13 base, PyTorch (cu130), SAM 3, rembg with weights pre-baked
- `requirements.txt` — Python dependencies
- `docker-compose.yml` — multi-GPU full-corpus run (one shard per GPU)
- `.env.example` — template for your local paths (copy to `.env`)
- `.dockerignore`

## Prerequisites

- An **NVIDIA GPU** with bfloat16 support (Ampere / compute capability 8.0+
  recommended — e.g. RTX 30-series, RTX 40-series, A-series). SAM 3 here runs under
  bf16 autocast; on pre-Ampere cards (Turing, sm75) bf16 is emulated/unsupported and
  you'll need to switch the autocast dtype to `float16` (see Troubleshooting).
- An **NVIDIA driver new enough for CUDA 13.0**. Run `nvidia-smi` and check the
  reported "CUDA Version" is **≥ 13.0**. If it's lower (e.g. 12.x), see "Older CUDA
  drivers" in Troubleshooting — you'll change two version pins, not the whole setup.
- **Docker** (recent) plus the **NVIDIA Container Toolkit** (Setup step 2).
- **~15 GB disk** for the image, plus ~3.2 GB for the SAM 3 checkpoint.
- A **Hugging Face account** with approved access to the gated SAM 3 checkpoint
  (Setup step 3 — request this early, approval can take up to a day).

This was developed on dual RTX A4500s (20 GB each), but a single modern GPU works
fine; the model is ~0.8B parameters and inference fits comfortably on typical
workstation cards. One GPU just means using the single-container run command
instead of the compose file. **No local NVIDIA GPU at all?** See
[Running without a local NVIDIA GPU](#running-without-a-local-nvidia-gpu) for
Colab, cloud-VM, and HPC-cluster options.

## Setup

### 1. Clone

```bash
git clone https://github.com/tripplowe/treebarkmask.git
cd treebarkmask
```

### 2. Install the NVIDIA Container Toolkit

This lets Docker containers see the GPU. (Linux/Debian-Ubuntu shown; see NVIDIA's
docs for other distros.)

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify Docker can reach your GPU(s):

```bash
docker run --rm --gpus all nvidia/cuda:13.0.1-base-ubuntu24.04 nvidia-smi
```

If that tag 404s, pick the newest `13.0.x-base-ubuntu24.04` from
[Docker Hub](https://hub.docker.com/r/nvidia/cuda/tags) (and update the `devel` tag
in the `Dockerfile` to match).

### 3. Get the SAM 3 checkpoint (gated)

The checkpoint can't ship in the image. Request access at
<https://huggingface.co/facebook/sam3> (fill the form accurately; approval can take
up to 24 hours). Then create a read token at
<https://huggingface.co/settings/tokens> and download the whole repo — both
`sam3.pt` and `config.json` are needed for offline loading:

```bash
export HF_TOKEN=hf_your_token_here
python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("facebook/sam3", local_dir="./sam3_ckpt")
PY
ls -lh sam3_ckpt    # expect sam3.pt (~3.2 GB) and config.json
```

### 4. Configure your paths

```bash
cp .env.example .env
```

Edit `.env` to point at your data, output directory, and the checkpoint folder, and
set your user id/group id so output files are owned by you:

```bash
# in .env
INPUT_DIR=/absolute/path/to/your/bark/photos
OUTPUT_DIR=/absolute/path/to/your/workdir
CKPT_DIR=/absolute/path/to/treebarkmask/sam3_ckpt
HOST_UID=1000     # use the output of: id -u
HOST_GID=1000     # use the output of: id -g
```

### 5. Build the image

```bash
docker build -t bole-mask:latest .
```

This installs PyTorch (CUDA 13 wheels), SAM 3 from source, and pre-fetches the
rembg weights so runs need no network.

## Running

### Calibrate first (single GPU, ~100 images)

Always do a small pass before committing to a full corpus — it verifies the whole
toolchain and lets you check the QC thresholds against *your* photos. Replace the
two `-v` host paths with yours:

```bash
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  -v /path/to/your/bark/photos:/data/in:ro \
  -v /path/to/your/workdir:/data/out \
  -v /path/to/treebarkmask/sam3_ckpt:/models:ro \
  bole-mask:latest \
  --input-dir /data/in --output-dir /data/out/out \
  --sam3-checkpoint /models/sam3.pt --prompt "tree trunk" \
  --gpu 0 --max-fill 0.98 --limit 100
```

### Full corpus

**Single GPU:** the same command without `--limit`.

**Multiple GPUs:** use compose, which runs one shard per GPU (each processes a
disjoint slice of the corpus). With `.env` filled in:

```bash
docker compose up --build
docker compose down        # after both shards exit
```

`docker-compose.yml` is set up for two GPUs; add or remove `serviceN` blocks to
match your hardware, incrementing `device_ids`. Note the deliberate quirk: each
container is given exactly one GPU, which appears as index 0 *inside* the container,
so **every shard passes `--gpu 0`** — the physical card is selected by `device_ids`,
not by `--gpu`.

## Running without a local NVIDIA GPU

This pipeline requires CUDA, so it can't use an Apple Silicon GPU, an AMD GPU, or
integrated laptop graphics. (It will fall back to CPU, but SAM 3 on CPU is minutes
per image — fine for trying one or two, not a corpus.) If you don't have a local
NVIDIA GPU, here are three alternatives, from closest-to-this-README to most
accessible.

### Option 1 — Rent a cloud GPU VM (Docker workflow unchanged)

*Best for: running the full corpus; keeping this README exactly as written.*

Providers such as Lambda, RunPod, Vast.ai, Paperspace, or the major clouds
(AWS/GCP/Azure) rent hourly Linux VMs with an NVIDIA GPU, and many ship a "deep
learning" image with Docker and the NVIDIA Container Toolkit already installed.
Once you have a shell on the VM, follow Setup steps 2–5 as written (skip step 2 if
the toolkit is preinstalled — `docker run --rm --gpus all
nvidia/cuda:13.0.1-base-ubuntu24.04 nvidia-smi` confirms it), copy your photos up
(`rsync -av ./photos/ user@vm:/data/in/`), download the checkpoint on the VM with
your HF token, and run the single-GPU command from "Full corpus." A modest GPU
(L4, A10, or any RTX-class card) is plenty; if you get a T4, see "Older GPUs" below.

### Option 2 — Google Colab (free / low-cost; no Docker)

*Best for: testing and small batches without spending money.*

Colab can't run Docker, so you run the pipeline natively in the Colab Python
runtime instead. Open a notebook, set **Runtime → Change runtime type → GPU**, then
run these cells in order:

```python
# 1. Check the GPU and whether it supports bfloat16
!nvidia-smi -L
import torch; print("bf16 supported:", torch.cuda.is_bf16_supported())
```

```bash
# 2. Get the code
!git clone https://github.com/tripplowe/treebarkmask.git
%cd treebarkmask
```

```bash
# 3. Install dependencies (uses Colab's preinstalled torch; mirrors the Dockerfile)
!pip install -q -r requirements.txt "git+https://github.com/facebookresearch/sam3.git"
```

> This pins NumPy to 1.26. If Colab asks to restart the runtime, do it
> (**Runtime → Restart session**), then re-run the `%cd treebarkmask` cell before
> continuing.

```python
# 4. Free Colab GPUs are T4s (Turing, no bf16). If cell 1 printed False, switch to fp16:
import torch
if not torch.cuda.is_bf16_supported():
    !sed -i 's/torch.bfloat16/torch.float16/g' bole_mask_pipeline.py
    print("patched bole_mask_pipeline.py to fp16")
```

```python
# 5. Mount Google Drive for the checkpoint, your photos, and outputs
from google.colab import drive; drive.mount('/content/drive')
```

```python
# 6. Download the gated SAM 3 checkpoint once, cached on Drive (see Setup step 3 for access)
import os
os.environ["HF_TOKEN"] = "hf_your_token_here"   # from huggingface.co/settings/tokens
from huggingface_hub import snapshot_download
ckpt = "/content/drive/MyDrive/sam3_ckpt"
if not os.path.exists(ckpt + "/sam3.pt"):
    snapshot_download("facebook/sam3", local_dir=ckpt)
```

```bash
# 7. Run (put your photos in a Drive folder; outputs go to Drive so they survive disconnects)
!python bole_mask_pipeline.py \
  --input-dir  /content/drive/MyDrive/bark_photos \
  --output-dir /content/drive/MyDrive/treebarkmask_out \
  --sam3-checkpoint /content/drive/MyDrive/sam3_ckpt/sam3.pt \
  --prompt "tree trunk" --gpu 0 --max-fill 0.98 --limit 50
```

Colab caveats: sessions disconnect when idle and cap at a few hours, and the free
T4 has 16 GB — fine for inference but not for the full ~5,000-image corpus in one
sitting. Use `--limit` or process subsets, and keep outputs on Drive. The review
gallery works the same way; after generating it, download the `review_gallery/`
folder from Drive and open `index.html` locally — no SSH tunnel needed.

### Option 3 — University HPC cluster (Apptainer / Singularity)

*Best for: students with cluster access and large runs.*

Shared clusters usually forbid Docker but provide Apptainer (formerly Singularity),
which can run Docker images. Build the image somewhere you have Docker and push it
to a registry (or save a tarball with `docker save`), then on the cluster:

```bash
apptainer build bole-mask.sif docker://YOUR_REGISTRY/bole-mask:latest
# or from a tarball:  apptainer build bole-mask.sif docker-archive://bole-mask.tar

apptainer exec --nv \
  --bind /path/to/photos:/data/in:ro \
  --bind /path/to/workdir:/data/out \
  --bind /path/to/sam3_ckpt:/models:ro \
  bole-mask.sif \
  python /app/bole_mask_pipeline.py \
    --input-dir /data/in --output-dir /data/out/out \
    --sam3-checkpoint /models/sam3.pt --prompt "tree trunk" \
    --gpu 0 --max-fill 0.98
```

`--nv` exposes the GPU and `--bind` replaces Docker's `-v`. Wrap the `apptainer
exec` in a SLURM script (`#SBATCH --gres=gpu:1`) for queued runs. If the cluster's
GPUs are Turing-era, apply the fp16 edit (below) before building.

### Older GPUs (Turing / pre-Ampere, including Colab's free T4)

These lack bfloat16. Switch the autocast dtype in `bole_mask_pipeline.py` from
`torch.bfloat16` to `torch.float16` (the `sed` line in the Colab cells above does
this automatically; for Docker/cluster, edit before building). `debug_sam3.py`
prints whether bf16 is supported. Everything else is unchanged.

## Outputs

Everything lands under `OUTPUT_DIR/out/`:

- `masks/<stem>.png` — accepted binary masks (255 = bole, 0 = background)
- `review/<stem>.png` — best-effort masks that failed QC, for manual inspection
- `cutouts/<stem>.png` — RGBA background-removed images (only with `--save-cutouts`)
- `manifest_shard*.csv` — per-image log: engine, pass/fail, failure reason, and the
  QC metrics (`fill_fraction`, `vspan`, `solidity`, `centroid_x`, `fragmentation`)

Quick post-run summary and manifest merge:

```bash
cd OUTPUT_DIR/out
ls masks/ | wc -l
cut -d, -f2,3,4 manifest_shard*.csv | grep -v '^engine' | sort | uniq -c
{ head -1 manifest_shard0.csv; tail -n +2 -q manifest_shard*.csv; } > manifest.csv
```

## Reviewing masks visually

`review_masks.py` overlays each mask on its original photo (background dimmed, mask
outlined — green = accepted, red = review) and builds a single scrollable HTML page,
**review items first, then accepted**, with QC metrics under each image. Run it in
the container:

```bash
docker run --rm --user "$(id -u):$(id -g)" --entrypoint python \
  -v /path/to/your/bark/photos:/data/in:ro \
  -v /path/to/your/workdir:/data/out \
  -v /path/to/treebarkmask/review_masks.py:/app/review_masks.py:ro \
  bole-mask:latest /app/review_masks.py \
  --input-dir /data/in --output-dir /data/out/out --which all --max 0
```

`--which review` renders only the failures (fast); `--which all` renders everything;
`--max 0` means no cap. Then open the gallery in a browser:

```bash
cd OUTPUT_DIR/out/review_gallery
python3 -m http.server 8085
```

If you're working locally, open <http://localhost:8085>. If the workstation is
remote, forward the port over SSH from your laptop and open the same URL locally:

```bash
ssh -N -L 8085:localhost:8085 you@workstation
```

(Pick any free port; both numbers just have to match. The All/Accepted/Review
buttons filter the page without regenerating it.)

## Tuning the QC gate

The QC thresholds are CLI flags; defaults are in `QCConfig` at the top of
`bole_mask_pipeline.py`. The workflow is: run a calibration pass, read the `reason`
column of the manifest to see which check rejects most masks, look at those masks in
the gallery, then adjust. Key knobs:

- `--max-fill` (default 0.92) — upper bound on mask-area / image-area. Raise toward
  0.98 if boles fill most of the frame (this dataset needed 0.98); a value near 1.0
  is usually background leak.
- `--min-fill` (default 0.30) — lower bound; raise if tiny partial masks slip through.
- `--min-vspan` (default 0.85) — required vertical extent; lower if boles don't reach
  the top/bottom edges.
- `--min-solidity` (default 0.80) — area / convex-hull area; lower for irregular boles.
- `--min-fragmentation` (default 0.95) — largest-component / total; lower to tolerate
  masks split by branches, tags, or shadow.
- `--center-tol` (default 0.20) — allowed horizontal offset of the centroid from center.

## When to enable rembg

Leave it off for frame-filling subjects like these boles. Turn it on with
`--use-rembg` only if your imagery has a clearly separable foreground against a
distinct background, where a free CPU saliency pass can clear the easy majority and
reserve the GPU for hard cases. With `--use-rembg`, rembg runs first and only its QC
failures escalate to SAM 3.

## Troubleshooting

**Build fails: `module 'numpy' has no attribute 'long'`** — a NumPy 1.x/2.x clash.
SAM 3 pins NumPy 1.26 while a newer SciPy needs NumPy ≥ 2.0. Fixed already:
`requirements.txt` pins SciPy to the bridge release `1.13.1` (supports NumPy
1.22–2.2), and the Dockerfile installs requirements and SAM 3 in one `pip` command
so the set resolves together.

**SAM 3 stage: `ModuleNotFoundError: No module named 'einops'` (or similar)** —
SAM 3's package under-declares runtime deps. `einops` and `ninja` are in
`requirements.txt`. If another module is reported missing, add its name there and
rebuild. Do **not** install `flash-attn-3` — it targets Hopper (H100); SAM 3 falls
back to standard attention without it.

**SAM 3 fails every image: `RuntimeError: mat1 and mat2 must have the same dtype,
but got BFloat16 and Float`** — SAM 3's backbone runs activations in bf16 but the
checkpoint loads in fp32. Fixed already: the pipeline loads on CUDA and wraps
inference in `torch.autocast("cuda", dtype=torch.bfloat16)`. (Not a device issue —
`device="cuda"` alone doesn't fix it.)

**SAM 3 fails every image: `TypeError ... Got unsupported ScalarType BFloat16`** —
NumPy has no bf16 dtype. Fixed already: the pipeline casts `.float()` before
`.numpy()` on masks and scores.

**Pre-Ampere GPU / `bf16 supported: False`** — run `debug_sam3.py` to check; if bf16
is unsupported, change the two `torch.bfloat16` references in `bole_mask_pipeline.py`
(the autocast call in `run_sam3_stage`) to `torch.float16` and rebuild.

**Older CUDA drivers (`nvidia-smi` shows CUDA < 13.0)** — change the `Dockerfile`
base image to a matching CUDA (e.g. `nvidia/cuda:12.6.x-devel-ubuntu24.04`) and the
torch index in the Dockerfile from `cu130` to your version (e.g. `cu126`), then
rebuild. Everything else is unchanged.

**No images found / permission denied** — on Linux you need the execute (traverse)
bit on every directory in the input path, not just read on the files. `ls
INPUT_DIR | head` confirms you can both list and reach them.

**`bash: UID: readonly variable`** — `UID` is reserved in bash. The compose file
uses `HOST_UID`/`HOST_GID` (set in `.env`); don't try to `export UID`.

**Output files owned by root** — pass `--user "$(id -u):$(id -g)"` on `docker run`,
or set `HOST_UID`/`HOST_GID` in `.env` for compose.

**Port already in use when serving the gallery** — pick another port; the
`http.server` argument and both sides of the `ssh -L` mapping just have to agree.

## AI assistance disclosure

The mask-generation pipeline, containerized environment, and associated code were
developed with the assistance of a large language model (Claude Opus 4.8,
Anthropic; claude.ai) for code drafting, debugging, and documentation. All
generated code was reviewed, tested, and validated by the authors, who take full
responsibility for the methodology and results. The tool was not used for study
design or interpretation of results.

## Acknowledgements

Built on Meta's [Segment Anything 3](https://github.com/facebookresearch/sam3) and
[rembg](https://github.com/danielgatis/rembg).
