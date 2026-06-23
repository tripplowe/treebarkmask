#!/usr/bin/env python3
#
# AI assistance disclosure:
#   The mask-generation pipeline, containerized environment, and associated code
#   were developed with the assistance of a large language model (Claude Opus 4.8,
#   Anthropic; claude.ai) for code drafting, debugging, and documentation. All
#   generated code was reviewed, tested, and validated by the authors, who take
#   full responsibility for the methodology and results. The tool was not used for
#   study design or interpretation of results.
#
"""
Tree-bole mask generation pipeline (cheap-first cascade).

Stage 1  rembg (CPU, parallel)  -> geometric QC
Stage 2  SAM 3 (GPU)            -> geometric QC      [only on Stage-1 failures]
Routing  pass -> outputs/masks/        fail-both -> outputs/review/

The QC gate exploits the acquisition protocol (bole dominant, centred, spanning
the frame top-to-bottom) to decide automatically whether a mask is good. The same
QC is applied to both engines, so the only thing that escalates to the GPU are the
images rembg could not handle cleanly.

--------------------------------------------------------------------------------
INSTALL (Python 3.10+; verified SAM 3 API as of 2026):

  # PyTorch built for your CUDA (A4500 = Ampere; CUDA 12.x driver):
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

  pip install rembg[cpu] onnxruntime opencv-python-headless pillow numpy scipy

  # SAM 3 (gated): request access at https://huggingface.co/facebook/sam3 first,
  # then `hf auth login`, then install the package and place sam3.pt locally:
  pip install git+https://github.com/facebookresearch/sam3.git

--------------------------------------------------------------------------------
RUN (single GPU, full corpus):

  python bole_mask_pipeline.py \
      --input-dir  /data/boles \
      --output-dir /data/boles_out \
      --sam3-checkpoint /models/sam3.pt \
      --prompt "tree trunk" --workers 24 --gpu 0

CALIBRATE FIRST on ~100 images with --limit 100 and inspect outputs/ before the
full run; the QC thresholds below are the knobs to tune.

DUAL-GPU (run two shards at once, one per A4500 — each does rembg+SAM3 on half):

  python bole_mask_pipeline.py ... --gpu 0 --shard-index 0 --shard-count 2 --workers 12 &
  python bole_mask_pipeline.py ... --gpu 1 --shard-index 1 --shard-count 2 --workers 12 &
  wait
"""

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
from PIL import Image, ImageOps
from scipy import ndimage

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


# ----------------------------------------------------------------------------- #
# QC configuration (the tuning knobs)
# ----------------------------------------------------------------------------- #
@dataclass
class QCConfig:
    min_fragmentation: float = 0.95   # largest_component / total_foreground (~1 = single blob)
    min_fill: float = 0.30            # mask area / image area, lower bound
    max_fill: float = 0.92            # ... upper bound (>this usually means it grabbed background)
    center_tol: float = 0.20          # |centroid_x - 0.5| must be <= this
    min_vspan: float = 0.85           # mask bbox height / image height (bole spans the frame)
    min_solidity: float = 0.80        # area / convex-hull area (boles are roughly columnar)
    close_kernel: int = 7             # morphological close to bridge small gaps before measuring
    edge_margin_px: int = 2           # tolerance for "touches top / bottom edge"


@dataclass
class QCResult:
    ok: bool = False
    reason: str = ""
    fragmentation: float = 0.0
    fill_fraction: float = 0.0
    centroid_x: float = 0.0
    vspan: float = 0.0
    solidity: float = 0.0
    touches_top: bool = False
    touches_bottom: bool = False
    final_mask: Optional[np.ndarray] = field(default=None, repr=False)


def evaluate_qc(mask_bool: np.ndarray, cfg: QCConfig) -> QCResult:
    """Clean a candidate mask, measure it against the protocol priors, decide pass/fail.
    Returns a QCResult whose final_mask is the largest hole-filled component."""
    H, W = mask_bool.shape
    m = mask_bool.astype(np.uint8)
    total_fg = int(m.sum())
    if total_fg == 0:
        return QCResult(reason="empty")

    if cfg.close_kernel > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.close_kernel, cfg.close_kernel))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return QCResult(reason="empty_after_clean")

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    largest_area = int(areas.max())
    fg_after = max(int(m.sum()), 1)
    fragmentation = largest_area / fg_after

    comp = (labels == largest).astype(np.uint8)
    comp = ndimage.binary_fill_holes(comp).astype(np.uint8)

    ys, xs = np.where(comp > 0)
    area = int(comp.sum())
    fill_fraction = area / float(H * W)
    cx = float(xs.mean()) / W
    y0, y1 = int(ys.min()), int(ys.max())
    vspan = (y1 - y0 + 1) / float(H)
    touches_top = y0 <= cfg.edge_margin_px
    touches_bottom = y1 >= H - 1 - cfg.edge_margin_px

    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    hull_area = max(cv2.contourArea(hull), 1.0)
    solidity = area / hull_area

    r = QCResult(
        fragmentation=fragmentation, fill_fraction=fill_fraction, centroid_x=cx,
        vspan=vspan, solidity=solidity, touches_top=bool(touches_top),
        touches_bottom=bool(touches_bottom), final_mask=(comp * 255).astype(np.uint8),
    )

    checks = [
        (fragmentation >= cfg.min_fragmentation, "fragmented"),
        (cfg.min_fill <= fill_fraction <= cfg.max_fill, "fill_out_of_range"),
        (abs(cx - 0.5) <= cfg.center_tol, "off_center"),
        (vspan >= cfg.min_vspan, "short_vspan"),
        (solidity >= cfg.min_solidity, "low_solidity"),
    ]
    failed = [name for ok, name in checks if not ok]
    r.ok = len(failed) == 0
    r.reason = "pass" if r.ok else ",".join(failed)
    return r


def save_mask(mask_u8: np.ndarray, dest: Path):
    cv2.imwrite(str(dest), mask_u8)


def save_cutout(image_path: Path, mask_u8: np.ndarray, dest: Path):
    img = ImageOps.exif_transpose(Image.open(image_path)).convert("RGBA")
    arr = np.array(img)
    arr[..., 3] = mask_u8
    Image.fromarray(arr).save(dest)


# ----------------------------------------------------------------------------- #
# Stage 1 — rembg, run in a CPU process pool (one onnxruntime session per worker)
# ----------------------------------------------------------------------------- #
_SESSION = None
_CFG = None
_OPTS = None


def _init_worker(model_name, cfg: QCConfig, opts: dict):
    global _SESSION, _CFG, _OPTS
    from rembg import new_session
    # Force CPU so the GPUs stay free for SAM 3.
    _SESSION = new_session(model_name, providers=["CPUExecutionProvider"])
    _CFG = cfg
    _OPTS = opts


def _rembg_one(path_str: str):
    from rembg import remove
    path = Path(path_str)
    try:
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        matte = remove(img, session=_SESSION, only_mask=True, post_process_mask=True)
        mask_bool = np.array(matte) > _OPTS["rembg_threshold"]
        qc = evaluate_qc(mask_bool, _CFG)
        if qc.ok:
            stem = path.stem
            save_mask(qc.final_mask, Path(_OPTS["masks_dir"]) / f"{stem}.png")
            if _OPTS["save_cutouts"]:
                save_cutout(path, qc.final_mask, Path(_OPTS["cutouts_dir"]) / f"{stem}.png")
        return {"path": path_str, "engine": "rembg", "passed": qc.ok,
                "reason": qc.reason, "fragmentation": qc.fragmentation,
                "fill_fraction": qc.fill_fraction, "centroid_x": qc.centroid_x,
                "vspan": qc.vspan, "solidity": qc.solidity,
                "touches_top": qc.touches_top, "touches_bottom": qc.touches_bottom}
    except Exception as e:  # never let one bad image kill the batch
        return {"path": path_str, "engine": "rembg", "passed": False,
                "reason": f"error:{type(e).__name__}", "fragmentation": 0.0,
                "fill_fraction": 0.0, "centroid_x": 0.0, "vspan": 0.0,
                "solidity": 0.0, "touches_top": False, "touches_bottom": False}


# ----------------------------------------------------------------------------- #
# Stage 2 — SAM 3, single process pinned to one GPU
# ----------------------------------------------------------------------------- #
def _coerce_masks(masks, H, W):
    """Normalise SAM 3 mask output to a list of HxW bool arrays.
    NOTE: verify on first run — adjust the threshold branch if your build returns
    logits rather than probabilities."""
    out = []
    for m in masks:
        a = m.detach().float().cpu().numpy() if hasattr(m, "detach") else np.asarray(m)
        a = np.squeeze(a)
        if a.dtype == bool:
            b = a
        elif a.dtype.kind == "f":
            b = (a > 0.5) if a.max() <= 1.0 else (a > 0.0)  # prob vs logit
        else:
            b = a > 0
        if b.shape != (H, W):
            b = cv2.resize(b.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        out.append(b)
    return out


def run_sam3_stage(failures, cfg: QCConfig, opts: dict, manifest_rows):
    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SAM3] loading checkpoint {opts['sam3_checkpoint']} on {device} ...", flush=True)
    processor = Sam3Processor(
        build_sam3_image_model(checkpoint_path=opts["sam3_checkpoint"], device=device),
        device=device,
    )
    # SAM 3's backbone runs activations in bfloat16 but the checkpoint loads in
    # fp32; without autocast the first linear hits a BFloat16-vs-Float dtype
    # mismatch. autocast reconciles inputs and weights per op (enabled only on
    # CUDA, since CPU has no bf16 matmul path).
    use_autocast = device == "cuda"

    for i, path_str in enumerate(failures, 1):
        path = Path(path_str)
        row = {"path": path_str, "engine": "sam3", "passed": False, "reason": "",
               "fragmentation": 0.0, "fill_fraction": 0.0, "centroid_x": 0.0,
               "vspan": 0.0, "solidity": 0.0, "touches_top": False, "touches_bottom": False}
        try:
            img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            W, H = img.size
            with torch.inference_mode(), torch.autocast(
                    device_type=device, dtype=torch.bfloat16, enabled=use_autocast):
                state = processor.set_image(img)
                out = processor.set_text_prompt(state=state, prompt=opts["prompt"])

            masks = out["masks"]
            scores = out["scores"]
            scores = scores.detach().float().cpu().numpy() if hasattr(scores, "detach") else np.asarray(scores)
            scores = np.ravel(scores)

            if len(masks) == 0:
                row["reason"] = "no_detection"
            else:
                bool_masks = _coerce_masks(masks, H, W)
                order = np.argsort(-scores)  # best score first
                best_qc, best_row = None, None
                for idx in order:
                    qc = evaluate_qc(bool_masks[idx], cfg)
                    cand = {**row, "reason": qc.reason, "fragmentation": qc.fragmentation,
                            "fill_fraction": qc.fill_fraction, "centroid_x": qc.centroid_x,
                            "vspan": qc.vspan, "solidity": qc.solidity,
                            "touches_top": qc.touches_top, "touches_bottom": qc.touches_bottom}
                    if best_qc is None:
                        best_qc, best_row = qc, cand  # remember top-scoring candidate
                    if qc.ok:
                        best_qc, best_row = qc, cand   # first QC-passing instance wins
                        break
                row = best_row
                row["passed"] = best_qc.ok
                stem = path.stem
                if best_qc.ok:
                    save_mask(best_qc.final_mask, Path(opts["masks_dir"]) / f"{stem}.png")
                    if opts["save_cutouts"]:
                        save_cutout(path, best_qc.final_mask, Path(opts["cutouts_dir"]) / f"{stem}.png")
                elif best_qc.final_mask is not None:
                    # keep the best-effort mask so a human has something to correct
                    save_mask(best_qc.final_mask, Path(opts["review_dir"]) / f"{stem}.png")
        except Exception as e:
            row["reason"] = f"error:{type(e).__name__}"

        manifest_rows.append(row)
        if i % 50 == 0:
            print(f"[SAM3] {i}/{len(failures)}", flush=True)


# ----------------------------------------------------------------------------- #
# Driver
# ----------------------------------------------------------------------------- #
def gather_images(input_dir: Path):
    return sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)


def main():
    ap = argparse.ArgumentParser(description="Tree-bole mask pipeline (rembg -> SAM 3 cascade).")
    ap.add_argument("--input-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--sam3-checkpoint", required=True, type=str)
    ap.add_argument("--prompt", default="tree trunk")
    ap.add_argument("--rembg-model", default="isnet-general-use",
                    help="isnet-general-use (cleaner on natural objects) or u2net")
    ap.add_argument("--rembg-threshold", type=int, default=127)
    ap.add_argument("--workers", type=int, default=max(os.cpu_count() - 2, 1))
    ap.add_argument("--gpu", type=str, default="0", help="CUDA device index for SAM 3")
    ap.add_argument("--save-cutouts", action="store_true", help="also write RGBA background-removed PNGs")
    ap.add_argument("--limit", type=int, default=0, help="process only first N images (calibration)")
    ap.add_argument("--use-rembg", action="store_true",
                    help="enable the rembg pre-pass (off by default). When set, rembg runs "
                         "first on CPU and only its QC failures escalate to SAM 3. On imagery "
                         "where the bole fills the frame, rembg clears almost nothing, so the "
                         "default is to send all images straight to SAM 3.")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    # QC knobs
    ap.add_argument("--min-fragmentation", type=float, default=0.95)
    ap.add_argument("--min-fill", type=float, default=0.30)
    ap.add_argument("--max-fill", type=float, default=0.92)
    ap.add_argument("--center-tol", type=float, default=0.20)
    ap.add_argument("--min-vspan", type=float, default=0.85)
    ap.add_argument("--min-solidity", type=float, default=0.80)
    args = ap.parse_args()

    # Pin the GPU before torch/sam3 are imported anywhere.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cfg = QCConfig(min_fragmentation=args.min_fragmentation, min_fill=args.min_fill,
                   max_fill=args.max_fill, center_tol=args.center_tol,
                   min_vspan=args.min_vspan, min_solidity=args.min_solidity)

    masks_dir = args.output_dir / "masks"
    review_dir = args.output_dir / "review"
    cutouts_dir = args.output_dir / "cutouts"
    for d in (masks_dir, review_dir, cutouts_dir):
        d.mkdir(parents=True, exist_ok=True)

    opts = {
        "rembg_threshold": args.rembg_threshold, "prompt": args.prompt,
        "sam3_checkpoint": args.sam3_checkpoint, "save_cutouts": args.save_cutouts,
        "masks_dir": str(masks_dir), "review_dir": str(review_dir), "cutouts_dir": str(cutouts_dir),
    }

    images = gather_images(args.input_dir)
    if args.shard_count > 1:
        images = images[args.shard_index::args.shard_count]
    if args.limit > 0:
        images = images[:args.limit]
    if not images:
        print("No images found.", file=sys.stderr)
        sys.exit(1)
    stage1 = f"rembg='{args.rembg_model}' on {args.workers} CPU workers" if args.use_rembg else "disabled (default)"
    print(f"[plan] {len(images)} images | shard {args.shard_index}/{args.shard_count} "
          f"| rembg pre-pass: {stage1} | SAM3 on GPU {args.gpu}")

    # ---- Stage 1: rembg on CPU (off by default; enable with --use-rembg) ------
    manifest_rows = []
    if not args.use_rembg:
        failures = [str(p) for p in images]
        print(f"[rembg] disabled; all {len(failures)} images go straight to SAM 3")
    else:
        failures = []
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_init_worker,
                                 initargs=(args.rembg_model, cfg, opts)) as ex:
            futs = [ex.submit(_rembg_one, str(p)) for p in images]
            for j, fut in enumerate(as_completed(futs), 1):
                row = fut.result()
                manifest_rows.append(row)
                if not row["passed"]:
                    failures.append(row["path"])
                if j % 200 == 0:
                    print(f"[rembg] {j}/{len(images)}  (escalating so far: {len(failures)})", flush=True)
        print(f"[rembg] done: {len(images) - len(failures)} passed, "
              f"{len(failures)} escalating to SAM 3")

    # ---- Stage 2: SAM 3 on GPU, only the failures -----------------------------
    if failures:
        run_sam3_stage(failures, cfg, opts, manifest_rows)

    # ---- Manifest -------------------------------------------------------------
    # One CSV per shard so parallel processes never collide; merge afterwards.
    manifest_path = args.output_dir / f"manifest_shard{args.shard_index}.csv"
    cols = ["path", "engine", "passed", "reason", "fragmentation", "fill_fraction",
            "centroid_x", "vspan", "solidity", "touches_top", "touches_bottom"]
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in manifest_rows:
            w.writerow({k: r.get(k, "") for k in cols})

    accepted = sum(1 for r in manifest_rows if r["passed"])
    review = sum(1 for r in manifest_rows if not r["passed"])
    print(f"\n[summary] accepted={accepted}  review_queue={review}")
    print(f"[summary] masks -> {masks_dir}")
    print(f"[summary] review (best-effort masks) -> {review_dir}")
    print(f"[summary] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
