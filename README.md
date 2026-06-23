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
instead of the compose file.

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
