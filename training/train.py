import os
import math
import argparse
import json
import wandb
import numpy as np
from tqdm.auto import trange
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, ConcatDataset

from dense_unet import DenseUNet
from matrix_generator_chrom import MatrixDataset

# =========================
# Default Config
# =========================
DEFAULT_LR = 5e-5
DEFAULT_BATCH_SIZE = 8
DEFAULT_MATRIX_SIZE = 128
DEFAULT_STEP_SIZE = 64
DEFAULT_TRAIN_ITER = 60000
DEFAULT_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_PATTERN = "/scratch/pioneer/users/kxt439/DeepLoop2.0/target/ld_thres_1.0_npzs/{chrom}.npz"


# =========================
# Utils
# =========================
def normalize_ratio_str(ratio):
    """
    Keep a clean string form for paths and output dirs.
    Examples:
        1.0 -> "1.0"
        0.1 -> "0.1"
        0.05 -> "0.05"
    """
    return str(float(ratio)) if isinstance(ratio, (int, float)) else str(ratio)


def reconstruction_loss(preds, y):
    """
    Standard DeepLoop2 reconstruction loss against the LoopDenoise target.

    preds: [B, R, H, W]
    y    : [B, H, W]

    No target masking or zero-out is applied.
    """
    rep_target = y.unsqueeze(1)
    loss = 0.0
    for rep_i in range(preds.shape[1]):
        out = preds[:, rep_i].unsqueeze(1)
        loss += F.mse_loss(out, rep_target)
    return loss / preds.shape[1]


def parse_top_pcts(s):
    """
    Parse a comma-separated list of target top-percent values.
    Example: "1,5,10" -> [1.0, 5.0, 10.0]
    """
    if s is None or str(s).strip() == "":
        return []
    vals = []
    for item in str(s).split(","):
        item = item.strip()
        if not item:
            continue
        val = float(item)
        if val <= 0 or val >= 100:
            raise ValueError(f"Top-percent values must be between 0 and 100, got {val}")
        vals.append(val)
    return vals


def pct_to_metric_name(pct):
    """
    Convert 1.0 -> "top_target_1pct", 0.5 -> "top_target_0p5pct".
    """
    s = ("%g" % float(pct)).replace(".", "p")
    return f"top_target_{s}pct"


def fixed_pct_to_metric_name(pct):
    """
    Convert 1.0 -> "fixed_top_target_1pct", 0.5 -> "fixed_top_target_0p5pct".
    """
    return "fixed_" + pct_to_metric_name(pct)


def load_fixed_target_cutoffs_from_json(path, requested_top_pcts=None):
    """
    Load exact genome/eval-wide target cutoffs computed by compute_exact_target_cutoffs.py.

    Supported JSON formats:
      1) {"cutoffs": {"top_1pct": 2.3, "top_5pct": 1.4, ...}, ...}
      2) {"top_1pct": 2.3, "top_5pct": 1.4, ...}
      3) {"top_target_1pct": 2.3, ...}
      4) {"fixed_top_target_1pct": 2.3, ...}

    Returns keys in the internal base-name format expected by evaluate():
      {"top_target_1pct": cutoff, "top_target_5pct": cutoff, ...}
    evaluate() will log these as eval_mse/fixed_top_target_1pct, etc.
    """
    if path is None or str(path).strip() == "":
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"fixed cutoff json not found: {path}")

    with open(path, "r") as f:
        obj = json.load(f)

    raw = obj.get("cutoffs", obj) if isinstance(obj, dict) else obj
    if not isinstance(raw, dict):
        raise ValueError(f"Could not parse cutoff json. Expected a dict, got {type(raw)} from {path}")

    requested = None
    if requested_top_pcts:
        requested = set(("%g" % float(p)).replace(".", "p") for p in requested_top_pcts)

    out = {}
    for k, v in raw.items():
        key = str(k)
        if key.startswith("fixed_"):
            key = key[len("fixed_"):]
        if key.startswith("top_target_") and key.endswith("pct"):
            pct_name = key[len("top_target_"):-len("pct")]
            internal_key = key
        elif key.startswith("top_") and key.endswith("pct"):
            pct_name = key[len("top_"):-len("pct")]
            internal_key = f"top_target_{pct_name}pct"
        else:
            continue

        if requested is not None and pct_name not in requested:
            continue
        out[internal_key] = float(v)

    if not out:
        raise ValueError(
            f"No usable cutoffs found in {path}. Expected keys like top_1pct/top_5pct/top_10pct "
            f"or top_target_1pct/top_target_5pct/top_target_10pct."
        )
    return out


def build_input_pattern(depth_ratio):
    ratio_str = normalize_ratio_str(depth_ratio)
    # New 280M dataset
    if ratio_str == "2.0":
        return (
            "/scratch/pioneer/users/kxt439/DeepLoop2.0/"
            f"train_chroms/chroms/depth_{ratio_str}/"
            "{chrom}_sub2_280M_{rep}.npz"
        )
    # Original full-depth dataset
    elif ratio_str == "1.0":
        return (
            f"/scratch/pioneer/users/kxt439/DeepLoop2.0/"
            f"train_chroms/chroms/depth_{ratio_str}/"
            "{chrom}_{rep}.npz"
        )
    # Original downsampled datasets
    else:
        return (
            f"/scratch/pioneer/users/kxt439/DeepLoop2.0/"
            f"train_chroms/chroms/depth_{ratio_str}/"
            "{chrom}_{rep}_" + ratio_str + ".npz"
        )

def build_input_paths(chrom, rep_names, depth_ratio):
    pattern = build_input_pattern(depth_ratio)
    return [pattern.format(chrom=chrom, rep=rep) for rep in rep_names]


def build_target_path(chrom):
    return TARGET_PATTERN.format(chrom=chrom)


def build_viz_dir(depth_ratio, out_root):
    ratio_str = normalize_ratio_str(depth_ratio)
    return os.path.join(out_root, f"ds_{ratio_str}", "viz")


def wsd_lr(step, total_steps, base_lr, warm_frac=0.05, stable_frac=0.7, final_lr_frac=0.25):
    warm_steps = int(total_steps * warm_frac)
    stable_steps = int(total_steps * stable_frac)
    decay_steps = max(1, total_steps - warm_steps - stable_steps)

    if step < warm_steps:
        return base_lr * (step + 1) / max(1, warm_steps)

    if step < warm_steps + stable_steps:
        return base_lr

    t = (step - warm_steps - stable_steps) / decay_steps
    min_lr = base_lr * final_lr_frac
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))


@torch.no_grad()
def evaluate(model, eval_loader, matrix_size, device, max_batches=None, corr_max_values=2_000_000, fixed_target_cutoffs=None):
    """
    Validation protocol:
      1) Keep the original sampled MSE as eval_loss for training stability monitoring.
      2) Add signal-aware metrics computed from mean prediction across reps:
         - Pearson correlation vs validation target
         - Spearman correlation vs validation target
         - stratified MSE by within-window distance bands
         - MSE for target high-signal/background pixels
         - MSE on fixed target cutoffs derived once before training

    Note: eval_loader is built with shuffle=False and eval datasets use jitter=False,
    so the first max_batches batches are fixed/reproducible across eval calls.
    """
    model.eval()

    total_loss = 0.0
    total_items = 0

    # Within-window distance masks. These are cheap proxies for near-/mid-/long-range signal.
    ii = torch.arange(matrix_size, device=device).view(-1, 1)
    jj = torch.arange(matrix_size, device=device).view(1, -1)
    dist = torch.abs(ii - jj)
    distance_masks = {
        "near_diag_0_32": dist < 32,
        "mid_range_32_64": (dist >= 32) & (dist < 64),
        "long_range_64_plus": dist >= 64,
    }

    mse_sum = {k: 0.0 for k in distance_masks}
    mse_count = {k: 0 for k in distance_masks}
    signal_mse_sum = {"target_high_gt1": 0.0, "target_bg_le1": 0.0}
    signal_mse_count = {"target_high_gt1": 0, "target_bg_le1": 0}

    fixed_target_cutoffs = fixed_target_cutoffs or {}
    fixed_mse_sum = {"fixed_" + name: 0.0 for name in fixed_target_cutoffs}
    fixed_mse_count = {"fixed_" + name: 0 for name in fixed_target_cutoffs}

    # Reservoir-like deterministic subsampling for correlation to avoid huge CPU memory use.
    rng = np.random.default_rng(12345)
    pred_corr_chunks = []
    target_corr_chunks = []
    corr_values_seen = 0

    for batch_idx, (x, y) in enumerate(eval_loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        preds, _, _ = model(x.view(-1, 1, matrix_size, matrix_size))
        preds = preds.view(x.shape)
        rep_target = y.unsqueeze(1)

        # Original eval loss: same definition as before.
        rec_loss = 0.0
        for rep_i in range(x.shape[1]):
            out = preds[:, rep_i].unsqueeze(1)
            rec_loss += F.mse_loss(out, rep_target)
        rec_loss /= x.shape[1]

        bs = x.shape[0]
        total_loss += rec_loss.item() * bs
        total_items += bs

        # Signal-aware metrics use the mean prediction across all input reps.
        pred_mean = preds.mean(dim=1)  # [B, H, W]
        sq_err = (pred_mean - y) ** 2

        for name, mask_2d in distance_masks.items():
            vals = sq_err[:, mask_2d]
            mse_sum[name] += vals.sum().item()
            mse_count[name] += vals.numel()

        high_mask = y > 1.0
        bg_mask = ~high_mask
        if high_mask.any():
            vals = sq_err[high_mask]
            signal_mse_sum["target_high_gt1"] += vals.sum().item()
            signal_mse_count["target_high_gt1"] += vals.numel()
        if bg_mask.any():
            vals = sq_err[bg_mask]
            signal_mse_sum["target_bg_le1"] += vals.sum().item()
            signal_mse_count["target_bg_le1"] += vals.numel()

        # Fixed target-cutoff MSE:
        # Cutoffs are estimated once from the original validation target distribution before training,
        # then reused for every run/model. This avoids unfair batch-wise percentile shifts when
        # comparing full / zero_lt / mask_lt target modes.
        y_flat_t = y.detach().reshape(-1)
        sq_flat_t = sq_err.reshape(-1)
        for base_name, cutoff in fixed_target_cutoffs.items():
            metric_name = "fixed_" + base_name
            fixed_mask = y_flat_t >= float(cutoff)
            if fixed_mask.any():
                vals = sq_flat_t[fixed_mask]
                fixed_mse_sum[metric_name] += vals.sum().item()
                fixed_mse_count[metric_name] += vals.numel()

        # Deterministic cap for correlation values.
        pred_flat = pred_mean.detach().float().cpu().numpy().reshape(-1)
        target_flat = y.detach().float().cpu().numpy().reshape(-1)
        n_new = pred_flat.size
        remaining = corr_max_values - corr_values_seen
        if remaining > 0:
            if n_new <= remaining:
                pred_corr_chunks.append(pred_flat)
                target_corr_chunks.append(target_flat)
                corr_values_seen += n_new
            else:
                idx = rng.choice(n_new, size=remaining, replace=False)
                pred_corr_chunks.append(pred_flat[idx])
                target_corr_chunks.append(target_flat[idx])
                corr_values_seen += remaining

        if max_batches is not None and (batch_idx + 1) >= max_batches:
            break

    eval_loss = total_loss / max(1, total_items)
    metrics = {"eval_loss": eval_loss}

    for name in distance_masks:
        metrics[f"eval_mse/{name}"] = mse_sum[name] / max(1, mse_count[name])
    for name in signal_mse_sum:
        metrics[f"eval_mse/{name}"] = signal_mse_sum[name] / max(1, signal_mse_count[name])
    for name in fixed_mse_sum:
        metrics[f"eval_mse/{name}"] = fixed_mse_sum[name] / max(1, fixed_mse_count[name])
        metrics[f"eval_count/{name}"] = int(fixed_mse_count[name])

    if pred_corr_chunks and target_corr_chunks:
        pred_all = np.concatenate(pred_corr_chunks)
        target_all = np.concatenate(target_corr_chunks)
        finite_mask = np.isfinite(pred_all) & np.isfinite(target_all)
        pred_all = pred_all[finite_mask]
        target_all = target_all[finite_mask]

        if pred_all.size > 1 and np.std(pred_all) > 0 and np.std(target_all) > 0:
            metrics["eval_corr/pearson"] = float(np.corrcoef(pred_all, target_all)[0, 1])
            sp = spearmanr(pred_all, target_all).correlation
            metrics["eval_corr/spearman"] = float(sp) if np.isfinite(sp) else float("nan")
        else:
            metrics["eval_corr/pearson"] = float("nan")
            metrics["eval_corr/spearman"] = float("nan")
        metrics["eval_corr/n_values"] = int(pred_all.size)

    model.train()
    return metrics

def save_checkpoint(path, model, opt=None, global_step=None, iter_i=None, eval_loss=None, extra=None):
    """
    Save checkpoint in old prediction-compatible format only.
    The saved .pt file is directly model.state_dict(), without wrapper keys like
    model_state_dict / optimizer_state_dict.
    """
    torch.save(model.state_dict(), path)


def make_viz(model, viz_x, viz_y, train_loader, iter_i, viz_dir, tag="eval"):
    model.eval()
    with torch.no_grad():
        rep = viz_x[:, 0].unsqueeze(1).to(DEVICE)
        out, _, _ = model(rep)

    x_in = rep.detach().cpu().numpy()
    out = out.detach().cpu().numpy()
    y_np = viz_y.detach().cpu().numpy()

    if train_loader.dataset.log_transform:
        x_in = np.expm1(x_in)
        out = np.expm1(out)
        y_np = np.expm1(y_np)

    fig, axs = plt.subplots(8, 5, figsize=(8, 16))
    n_show = min(8, x_in.shape[0])

    for batch_i in range(n_show):
        axs[batch_i][0].imshow(x_in[batch_i, 0], cmap="Reds", vmin=1, vmax=2, interpolation=None)
        axs[batch_i][0].set_xticks([])
        axs[batch_i][0].set_yticks([])

        pred = out[batch_i, 0]
        pred = (pred + pred.T) / 2
        axs[batch_i][1].imshow(pred, cmap="Reds", vmin=1, vmax=2, interpolation=None)
        axs[batch_i][1].set_xticks([])
        axs[batch_i][1].set_yticks([])

        axs[batch_i][2].imshow(pred, cmap="Reds", interpolation=None)
        axs[batch_i][2].set_xticks([])
        axs[batch_i][2].set_yticks([])

        axs[batch_i][3].imshow(y_np[batch_i], cmap="Reds", interpolation=None)
        axs[batch_i][3].set_xticks([])
        axs[batch_i][3].set_yticks([])

        axs[batch_i][4].imshow(y_np[batch_i], cmap="Reds", vmin=1, vmax=2, interpolation=None)
        axs[batch_i][4].set_xticks([])
        axs[batch_i][4].set_yticks([])

    plt.tight_layout()
    heatmap_path = os.path.join(viz_dir, "heatmaps", f"dense_unet_{tag}_{iter_i:07d}.jpg")
    plt.savefig(heatmap_path)
    plt.close()

    filters = next(model.parameters()).detach().cpu().numpy()
    n_filters = filters.shape[0]
    n_rows = int(math.sqrt(n_filters))
    n_cols = math.ceil(n_filters / n_rows)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
    axs = np.array(axs).reshape(n_rows, n_cols)

    for i in range(n_filters):
        axs[i // n_cols, i % n_cols].imshow(filters[i, 0], cmap="RdGy_r")
        axs[i // n_cols, i % n_cols].set_xticks([])
        axs[i // n_cols, i % n_cols].set_yticks([])

    for i in range(n_filters, n_rows * n_cols):
        axs[i // n_cols, i % n_cols].axis("off")

    plt.tight_layout()
    filt_path = os.path.join(viz_dir, "filters", f"dense_unet_{tag}_{iter_i:07d}.jpg")
    plt.savefig(filt_path)
    plt.close()

    model.train()
    return heatmap_path, filt_path


def train(
    model,
    opt,
    train_loader,
    eval_loader,
    train_iterations,
    base_lr,
    total_steps,
    global_step,
    matrix_size,
    viz_dir,
    depth_ratio,
    eval_every=1000,
    eval_batches=100,
    ckpt_every=2000,
    save_step_ckpts=True,
    warm_frac=0.10,
    stable_frac=0.60,
    min_lr_frac=0.30,
    eval_viz_every=1,
    fixed_target_cutoffs=None,
):
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(os.path.join(viz_dir, "heatmaps"), exist_ok=True)
    os.makedirs(os.path.join(viz_dir, "filters"), exist_ok=True)

    last_ckpt_path = os.path.join(viz_dir, "last.pt")
    best_ckpt_path = os.path.join(viz_dir, "best.pt")
    step_ckpt_dir = os.path.join(viz_dir, "checkpoints")
    if save_step_ckpts:
        os.makedirs(step_ckpt_dir, exist_ok=True)

    def iterate_dataset(data_loader):
        data_iter = iter(data_loader)
        while True:
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                x, y = next(data_iter)
            yield x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)

    train_iter_gen = iterate_dataset(train_loader)

    eval_vis_iter = iter(eval_loader)
    viz_x, viz_y = next(eval_vis_iter)

    best_eval_loss = float("inf")

    for iter_i in (pbar := trange(train_iterations)):
        x, y = next(train_iter_gen)

        cur_lr = wsd_lr(
            global_step,
            total_steps,
            base_lr,
            warm_frac=warm_frac,
            stable_frac=stable_frac,
            final_lr_frac=min_lr_frac,
        )
        for pg in opt.param_groups:
            pg["lr"] = cur_lr

        opt.zero_grad(set_to_none=True)

        rep_loss = 0.0
        rec_loss = 0.0

        preds, indices, _ = model(x.view(-1, 1, matrix_size, matrix_size))
        preds = preds.view(x.shape)

        rec_loss = reconstruction_loss(preds=preds, y=y)

        for rep_i in range(x.shape[1]):
            out = preds[:, rep_i].unsqueeze(1)
            for rep_j in range(x.shape[1]):
                if rep_i != rep_j:
                    rep2_target = preds[:, rep_j].unsqueeze(1).detach()
                    rep_loss += F.mse_loss(out, rep2_target)

        rep_loss /= x.shape[1] * (x.shape[1] - 1)

        rec_loss.backward()
        opt.step()

        wandb.log({
            "train/mse": rec_loss.item(),
            "learning_rate": cur_lr,
            "global_step": global_step,
        })

        global_step += 1

        if save_step_ckpts and ckpt_every > 0 and global_step % ckpt_every == 0:
            step_ckpt_path = os.path.join(step_ckpt_dir, f"step_{global_step:08d}.pt")
            save_checkpoint(
                path=step_ckpt_path,
                model=model,
                opt=opt,
                global_step=global_step,
                iter_i=iter_i,
                eval_loss=None,
                extra={
                    "depth_ratio": str(depth_ratio),
                    "type": "step",
                    "global_step": global_step,
                },
            )
            print(f"[STEP CKPT] saved {step_ckpt_path}", flush=True)

        pbar.set_description(
            f"rec loss: {rec_loss.item():.4f} | rep loss: {rep_loss.item():.4f} | lr: {cur_lr:.2e}"
        )

        if iter_i % eval_every == 0:
            eval_metrics = evaluate(
                model=model,
                eval_loader=eval_loader,
                matrix_size=matrix_size,
                device=DEVICE,
                max_batches=eval_batches,
                fixed_target_cutoffs=fixed_target_cutoffs,
            )
            eval_loss = eval_metrics["eval_loss"]
            # Keep the current val/mse name while also logging the richer old-style metrics.
            eval_metrics["val/mse"] = eval_loss

            wandb.log({
                **eval_metrics,
                "best_eval_loss_so_far": min(best_eval_loss, eval_loss),
                "global_step": global_step,
            })

            save_checkpoint(
                path=last_ckpt_path,
                model=model,
                opt=opt,
                global_step=global_step,
                iter_i=iter_i,
                eval_loss=eval_loss,
                extra={"depth_ratio": str(depth_ratio), "type": "last"},
            )

            is_best = eval_loss < best_eval_loss
            if is_best:
                best_eval_loss = eval_loss
                save_checkpoint(
                    path=best_ckpt_path,
                    model=model,
                    opt=opt,
                    global_step=global_step,
                    iter_i=iter_i,
                    eval_loss=eval_loss,
                    extra={"depth_ratio": str(depth_ratio), "type": "best"},
                )
                print(f"[BEST] iter={iter_i} global_step={global_step} eval_loss={eval_loss:.6f}")

            # This is a fixed eval-batch visualization, not a random training batch.
            # It is useful for visual sanity checks, but should not replace quantitative metrics.
            do_eval_viz = (eval_viz_every is not None and eval_viz_every > 0 and
                           ((global_step // max(1, eval_every)) % eval_viz_every == 0))
            if do_eval_viz:
                heatmap_path, filter_path = make_viz(
                    model=model,
                    viz_x=viz_x,
                    viz_y=viz_y,
                    train_loader=train_loader,
                    iter_i=iter_i,
                    viz_dir=viz_dir,
                    tag="best" if is_best else "eval_fixed_batch",
                )

                wandb.log({
                    "eval_fixed_batch/reconstruction": wandb.Image(heatmap_path),
                    "filters": wandb.Image(filter_path),
                    "global_step": global_step,
                })

    final_ckpt_path = os.path.join(step_ckpt_dir, f"final_step_{global_step:08d}.pt") if save_step_ckpts else os.path.join(viz_dir, f"final_step_{global_step:08d}.pt")
    save_checkpoint(
        path=final_ckpt_path,
        model=model,
        opt=opt,
        global_step=global_step,
        iter_i=train_iterations - 1,
        eval_loss=None,
        extra={
            "depth_ratio": str(depth_ratio),
            "type": "final",
            "global_step": global_step,
        },
    )
    print(f"[FINAL CKPT] saved {final_ckpt_path}", flush=True)
    print(f"Finished chromosome block. Best eval loss so far: {best_eval_loss:.6f}")
    return global_step, best_eval_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth_ratio", type=str, required=True,
                        help='e.g. "1.0", "0.5", "0.25", "0.1", "0.05"')
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--matrix_size", type=int, default=DEFAULT_MATRIX_SIZE)
    parser.add_argument("--step_size", type=int, default=DEFAULT_STEP_SIZE)
    parser.add_argument("--train_iter", type=int, default=DEFAULT_TRAIN_ITER)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--eval_max_batches", type=int, default=100)
    parser.add_argument("--eval_top_pcts", type=str, default="1,5,10",
                        help='Fixed target top-percent metrics to track, e.g. "1,5,10".')
    parser.add_argument(
        "--fixed_cutoff_json",
        type=str,
        default="/scratch/pioneer/users/kxt439/DeepLoop2.0/target/ld_thres_1.0_npzs/target_top_cutoffs.json",
        help="JSON produced by compute_exact_target_cutoffs.py for the NEW LoopDenoise target.",
    )
    parser.add_argument("--eval_viz_every", type=int, default=1)
    parser.add_argument("--ckpt_every", type=int, default=2000)
    parser.add_argument("--no_step_ckpts", action="store_true")
    parser.add_argument("--warm_frac", type=float, default=0.10)
    parser.add_argument("--stable_frac", type=float, default=0.60)
    parser.add_argument("--min_lr_frac", type=float, default=0.30)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument(
        "--out_root",
        type=str,
        default="/scratch/pioneer/users/kxt439/DeepLoop2.0/denseunet_train_loopdenoise_target",
    )
    parser.add_argument("--wandb_project", type=str, default="dense_unet_loopdenoise_target")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
    )

    args = parser.parse_args()
    target_top_pcts = parse_top_pcts(args.eval_top_pcts)

    depth_ratio = normalize_ratio_str(args.depth_ratio)
    viz_dir = build_viz_dir(depth_ratio, args.out_root)
    os.makedirs(viz_dir, exist_ok=True)

    if args.warm_frac < 0 or args.stable_frac < 0 or args.warm_frac + args.stable_frac > 1:
        raise ValueError("warm_frac and stable_frac must be >= 0 and sum to <= 1")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_name = args.wandb_name or f"deeploop2_loopdenoise_target_ds_{depth_ratio}"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        group=args.wandb_group,
        mode=args.wandb_mode,
        config={
            "depth_ratio": depth_ratio,
            "target_pattern": TARGET_PATTERN,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "matrix_size": args.matrix_size,
            "step_size": args.step_size,
            "train_iter": args.train_iter,
            "seed": args.seed,
            "eval_every": args.eval_every,
            "eval_max_batches": args.eval_max_batches,
            "eval_top_pcts_for_fixed_cutoffs": target_top_pcts,
            "fixed_cutoff_json": args.fixed_cutoff_json,
            "ckpt_every": args.ckpt_every,
            "save_step_ckpts": not args.no_step_ckpts,
            "warm_frac": args.warm_frac,
            "stable_frac": args.stable_frac,
            "min_lr_frac": args.min_lr_frac,
            "target": "LoopDenoise prediction",
            "target_training": "full ordinary MSE; no zero-out/mask/cutoff",
        },
    )

    rep_names = [
        "H9", "H7", "HUES1", "HUES2", "HUES4", "HUES8", "ESI051", "ESI035", "ESI017",
        "51213", "51339", "ACS_1011", "ACS_1019", "ACS_1026", "AG28869", "CJ-IPS",
        "H1", "51286", "AG28041", "DIS1016", "GM23716", "GM23720",
        "iPSC_075C", "iPSC_088B", "iPSC_1029C", "iPSC_1106"
    ]

    if depth_ratio == "2.0":
        rep_names = [f"rep{i}" for i in range(2, 21)]

    train_chroms = [f"chr{i}" for i in range(1, 23, 2)]
    eval_chroms = [f"chr{i}" for i in range(2, 23, 2)]

    print("Device:", DEVICE)
    print("depth_ratio:", depth_ratio)
    print("target_pattern:", TARGET_PATTERN)
    print("viz_dir:", viz_dir)
    print("eval_viz_every:", args.eval_viz_every)

    # -------------------------
    # Build eval datasets
    # -------------------------
    eval_datasets = []
    for chrom in eval_chroms:
        cool_files = build_input_paths(chrom, rep_names, depth_ratio)
        target_file = build_target_path(chrom)

        if all(os.path.exists(f) for f in cool_files + [target_file]):
            eval_data = MatrixDataset(
                cool_files,
                target_file,
                matrix_size=args.matrix_size,
                step_size=args.step_size,
                log_transform=False,
                jitter=False,
                symmetric=True,
            )
            eval_datasets.append(eval_data)
        else:
            print(f"[Eval skip] {chrom} due to missing files")

    if len(eval_datasets) == 0:
        raise RuntimeError("No eval datasets found. Check eval_chroms paths.")

    eval_loader = DataLoader(
        ConcatDataset(eval_datasets),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # -------------------------
    # Load fixed top-target cutoffs for the NEW LoopDenoise target
    # -------------------------
    fixed_target_cutoffs = load_fixed_target_cutoffs_from_json(
        args.fixed_cutoff_json,
        requested_top_pcts=target_top_pcts,
    )
    print("Fixed target cutoffs used for validation:")
    print("  source:", args.fixed_cutoff_json)
    for name, cutoff in fixed_target_cutoffs.items():
        print(f"  {name}: target >= {cutoff:.6g}")

    cutoff_json_path = os.path.join(viz_dir, "fixed_target_cutoffs_used.json")
    with open(cutoff_json_path, "w") as f:
        json.dump({
            "source": args.fixed_cutoff_json,
            "fixed_target_cutoffs": fixed_target_cutoffs,
            "eval_top_pcts": target_top_pcts,
            "target_pattern": TARGET_PATTERN,
            "note": "Fixed cutoffs from the new LoopDenoise target. Masks are defined by ground-truth target values; training loss is unchanged.",
        }, f, indent=2)
    print(f"Saved cutoffs used by this run to: {cutoff_json_path}")

    # -------------------------
    # Build model
    # -------------------------
    model = DenseUNet(out_channels=1).to(DEVICE)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # -------------------------
    # Available train chroms
    # -------------------------
    available_train_chroms = []
    for chrom in train_chroms:
        cool_files = build_input_paths(chrom, rep_names, depth_ratio)
        target_file = build_target_path(chrom)

        if all(os.path.exists(f) for f in cool_files + [target_file]):
            available_train_chroms.append(chrom)
        else:
            print(f"[Train skip] {chrom} due to missing files")

    if len(available_train_chroms) == 0:
        raise RuntimeError("No train chromosomes found. Check train paths.")

    base_chrom_train_iters = args.train_iter // len(available_train_chroms)
    extra_iters = args.train_iter % len(available_train_chroms)
    chrom_iter_plan = {
        chrom: base_chrom_train_iters + (1 if i < extra_iters else 0)
        for i, chrom in enumerate(available_train_chroms)
    }
    total_steps = sum(chrom_iter_plan.values())

    print(f"Train chroms available: {len(available_train_chroms)}")
    print(f"base_chrom_train_iters = {base_chrom_train_iters}; extra_iters distributed to first {extra_iters} chroms")
    print(f"total_steps (global WSD) = {total_steps}")
    print(f"WSD schedule: warm={args.warm_frac:.3f}, stable={args.stable_frac:.3f}, decay={1 - args.warm_frac - args.stable_frac:.3f}, min_lr_frac={args.min_lr_frac:.3f}")
    print(f"Step checkpoints: save={not args.no_step_ckpts}, ckpt_every={args.ckpt_every}")
    print("Starting chromosome-wise training with GLOBAL WSD...")

    global_step = 0
    overall_best = float("inf")

    for chrom in available_train_chroms:
        print(f"[Loading] {chrom} for training...")

        cool_files = build_input_paths(chrom, rep_names, depth_ratio)
        target_file = build_target_path(chrom)

        dataset = MatrixDataset(
            cool_files,
            target_file,
            matrix_size=args.matrix_size,
            step_size=args.step_size,
            log_transform=False,
        )

        train_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        global_step, chrom_best = train(
            model=model,
            opt=opt,
            train_loader=train_loader,
            eval_loader=eval_loader,
            train_iterations=chrom_iter_plan[chrom],
            base_lr=args.lr,
            total_steps=total_steps,
            global_step=global_step,
            matrix_size=args.matrix_size,
            viz_dir=viz_dir,
            depth_ratio=depth_ratio,
            eval_every=args.eval_every,
            eval_batches=args.eval_max_batches,
            ckpt_every=args.ckpt_every,
            save_step_ckpts=not args.no_step_ckpts,
            warm_frac=args.warm_frac,
            stable_frac=args.stable_frac,
            min_lr_frac=args.min_lr_frac,
            eval_viz_every=args.eval_viz_every,
            fixed_target_cutoffs=fixed_target_cutoffs,
        )

        overall_best = min(overall_best, chrom_best)

    wandb.finish()

    print(f"Finished training on {len(available_train_chroms)} chromosomes.")
    print(f"Overall best eval loss observed: {overall_best:.6f}")


if __name__ == "__main__":
    main()