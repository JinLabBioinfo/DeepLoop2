#!/usr/bin/env python3
# dl_train_finetune.py — fine-tune the ONNX-converted DeepLoop denoiser with W&B logging

import os, argparse, math, time, json, pathlib
import numpy as np
from tqdm.auto import trange, tqdm
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

import onnx
from onnx2torch import convert
from onnx import helper, numpy_helper

# ===== Your local dataset =====
from matrix_generator_chrom import MatrixDataset

# -----------------------------
# Tracer (unchanged)
# -----------------------------
class ShapeTracer:
    def __init__(self, max_print=200):
        self.hooks = []
        self.max_print = max_print
        self.count = 0
    def _hook(self, name):
        def fn(module, inp, out):
            if self.count < self.max_print:
                ish = tuple(inp[0].shape) if isinstance(inp, (list, tuple)) else tuple(inp.shape)
                osh = tuple(out.shape) if hasattr(out, "shape") else "<non-tensor>"
                print(f"[TRACE] {name:40s} in={ish} out={osh}", flush=True)
                self.count += 1
        return fn
    def attach(self, model):
        for name, m in model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d, nn.AvgPool2d, nn.ReLU, nn.Upsample, nn.SiLU, nn.LeakyReLU, nn.BatchNorm2d)):
                self.hooks.append(m.register_forward_hook(self._hook(name)))
    def detach(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

class FusionWrapper(nn.Module):
    """
    Fuse R replicate channels -> 1 via 1x1 conv, then run the ONNX model.
    Init weights as equal-mean so step-0 ≈ average replicate.
    """
    def __init__(self, base: nn.Module, in_reps: int, out_c: int = 1, init: str = "mean"):
        super().__init__()
        self.base = base
        self.fuse = nn.Conv2d(in_reps, out_c, kernel_size=1, bias=False)
        if init == "mean":
            with torch.no_grad():
                self.fuse.weight.zero_()
                self.fuse.weight[:, :, 0, 0] = 1.0 / float(in_reps)

    def forward(self, x):
        # x: [B, R, H, W]
        x = self.fuse(x)  # -> [B, 1, H, W]
        return self.base(x)

# -----------------------------
# Small utils
# -----------------------------
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

def infinite_loader(dl):
    it = iter(dl)
    while True:
        try:
            yield next(it)
        except StopIteration:
            it = iter(dl)

def describe(t: torch.Tensor) -> str:
    try:
        tmin = float(t.min()) if t.numel() else float("nan")
        tmax = float(t.max()) if t.numel() else float("nan")
    except Exception:
        tmin, tmax = float("nan"), float("nan")
    return f"shape={tuple(t.shape)}, dtype={t.dtype}, min={tmin:.4g}, max={tmax:.4g}"

def get_expected_in_channels_from_onnx(onnx_model) -> int:
    init_map = {t.name: t for t in onnx_model.graph.initializer}
    for n in onnx_model.graph.node:
        if n.op_type == "Conv":
            for inp in n.input:
                if inp in init_map:
                    w = init_map[inp]  # [out_c, in_c, kh, kw]
                    return int(w.dims[1])
            break
    return 1

def _tensor_to_np(t: onnx.TensorProto) -> np.ndarray:
    try:
        arr = numpy_helper.to_array(t)
        return np.array(arr)
    except Exception:
        if t.raw_data:
            arr = np.frombuffer(t.raw_data, dtype=np.float32)
            return arr.reshape(t.dims) if t.dims else arr
        if t.float_data:
            return np.array(t.float_data, dtype=np.float32).reshape(t.dims)
        if t.int64_data:
            return np.array(t.int64_data, dtype=np.int64).reshape(t.dims)
        return np.array([], dtype=np.float32)

def find_first_conv_idx(m: onnx.ModelProto) -> int:
    for i, n in enumerate(m.graph.node):
        if n.op_type == "Conv":
            return i
    raise RuntimeError("No Conv node found in ONNX graph.")

# -----------------------------
# NEW: parameter reset
# -----------------------------
def reset_model_params_(m: torch.nn.Module):
    for mod in m.modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(mod.weight, nonlinearity="relu")
            if getattr(mod, "bias", None) is not None:
                nn.init.zeros_(mod.bias)
        elif isinstance(mod, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm2d)):
            if getattr(mod, "weight", None) is not None:
                nn.init.ones_(mod.weight)
            if getattr(mod, "bias", None) is not None:
                nn.init.zeros_(mod.bias)

# -----------------------------
# ONNX surgery (unchanged)
# -----------------------------
def fix_same_upper_padding(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
    for node in onnx_model.graph.node:
        if node.op_type not in ["Conv", "ConvTranspose", "MaxPool"]:
            continue
        idx_to_delete = None
        for i, attr in enumerate(node.attribute):
            if attr.name == "auto_pad" and attr.s == b"SAME_UPPER":
                idx_to_delete = i
                break
        if idx_to_delete is None:
            continue
        del node.attribute[idx_to_delete]
        kshape = None
        for a in node.attribute:
            if a.name == "kernel_shape":
                kshape = list(a.ints)
        pads = [0,0,0,0]
        if kshape:
            pads2 = []
            for k in kshape:
                p = (k - 1) // 2
                pads2.extend([p, p])
            pads = pads2 if len(pads2) == 4 else [pads2[0], pads2[1], pads2[0], pads2[1]]
        node.attribute.append(helper.make_attribute("pads", pads))
    return onnx_model

def patch_input_chain_to_nchw(m: onnx.ModelProto, expected_hw: int, expected_c: int = 1, verbose=True) -> onnx.ModelProto:
    out2node = {}
    for n in m.graph.node:
        for o in n.output:
            out2node[o] = n
    try:
        conv_idx = find_first_conv_idx(m)
    except RuntimeError:
        if verbose: print("[ONNX-PATCH] No Conv found; skip.")
        return m
    conv = m.graph.node[conv_idx]
    if not conv.input:
        return m
    data_in = conv.input[0]
    chain = []
    cur = data_in
    for _ in range(10):
        prod = out2node.get(cur, None)
        if prod is None:
            break
        if prod.op_type in ("Reshape", "Transpose", "Squeeze", "Unsqueeze"):
            chain.append(prod)
            if not prod.input:
                break
            cur = prod.input[0]
        else:
            break
    if verbose:
        print("[ONNX-PATCH] Upstream chain into first Conv (closest first):",
              " -> ".join([n.op_type for n in chain]) or "<none>")
    if not chain:
        if verbose: print("[ONNX-PATCH] No reshape/transpose chain to patch; skipping.")
        return m
    init_map = {t.name: t for t in m.graph.initializer}
    n0 = chain[0]
    if n0.op_type == "Transpose":
        found = False
        for a in n0.attribute:
            if a.name == "perm":
                a.ints[:] = [0, 1, 2, 3]
                found = True
                break
        if not found:
            n0.attribute.append(helper.make_attribute("perm", [0,1,2,3]))
        if verbose:
            print("[ONNX-PATCH] Set nearest Transpose to identity perm=[0,1,2,3].")
    nearest_reshape = None
    for n in chain:
        if n.op_type == "Reshape":
            nearest_reshape = n
            break
    if nearest_reshape is None:
        if verbose:
            print("[ONNX-PATCH] No Reshape in chain; nothing to set to [-1,C,H,W].")
        return m
    if len(nearest_reshape.input) < 2:
        if verbose: print("[ONNX-PATCH] Reshape has no 'shape' input; cannot patch.")
        return m
    shape_name = nearest_reshape.input[1]
    target = np.array([-1, int(expected_c), int(expected_hw), int(expected_hw)], dtype=np.int64)
    if shape_name in init_map:
        new_t = numpy_helper.from_array(target, name=shape_name)
        idx = [i for i,t in enumerate(m.graph.initializer) if t.name == shape_name][0]
        m.graph.initializer[idx].CopyFrom(new_t)
        if verbose:
            print(f"[ONNX-PATCH] Patched nearest Reshape target → {target.tolist()}")
    else:
        new_name = shape_name + "__patched_static_nchw"
        m.graph.initializer.append(numpy_helper.from_array(target, name=new_name))
        nearest_reshape.input[1] = new_name
        if verbose:
            print(f"[ONNX-PATCH] Injected static NCHW shape initializer → {target.tolist()}")
    return m

def verify_some_weights_from_onnx(onnx_model, torch_model, max_checks=5, atol=1e-6):
    init_map = {t.name: t for t in onnx_model.graph.initializer if len(t.dims) > 0}
    tstate = torch_model.state_dict()
    by_shape = {}
    for k, v in tstate.items():
        by_shape.setdefault(tuple(v.shape), []).append((k, v))
    checks = matched = 0
    for name, init in init_map.items():
        if checks >= max_checks:
            break
        shp = tuple(int(x) for x in init.dims)
        if shp not in by_shape:
            continue
        arr = _tensor_to_np(init)
        if arr.size == 0:
            continue
        for tk, tv in by_shape[shp]:
            if np.allclose(arr, tv.detach().cpu().numpy(), atol=atol, rtol=1e-5):
                print(f"[WEIGHTS] OK: ONNX '{name}' ~ Torch '{tk}' {shp}")
                matched += 1
                break
        checks += 1
    print(f"[WEIGHTS] Verified close match for {matched}/{checks} sampled initializers.")


class ExpandReplicates(torch.utils.data.Dataset):
    def __init__(self, base_ds):
        self.base = base_ds
        x0, _ = self.base[0]
        self.R = int(x0.shape[0])  # x0: [R,H,W] or [R,1,H,W]

    def _to_tensor_chw1(self, t):
        if isinstance(t, np.ndarray):
            if t.ndim == 2: t = t[None, ...]
            return torch.from_numpy(t.astype(np.float32, copy=False))
        if isinstance(t, torch.Tensor):
            if t.ndim == 2: t = t.unsqueeze(0)
            return t.contiguous().float()
        arr = np.asarray(t)
        if arr.ndim == 2: arr = arr[None, ...]
        return torch.from_numpy(arr.astype(np.float32, copy=False))

    def __len__(self):
        return len(self.base) * self.R

    def __getitem__(self, k):
        i, r = divmod(k, self.R)
        x, y = self.base[i]
        xr = self._to_tensor_chw1(x[r])
        y  = self._to_tensor_chw1(y)
        return xr, y

# -----------------------------
# Data builders (log_transform=False)
# -----------------------------
def build_loader(rep_names, chroms, matrix_size, step_size, batch_size, target_root, shuffle, expand_reps):
    datasets = []
    for chrom in chroms:
        cool_files = [f"/scratch/pioneer/users/kxt439/DeepLoop2.0/rep_coolers/downsample/{chrom}/0.05/{chrom}_downsample_{rep}_0.05.npz" for rep in rep_names]
        target_file = f"{target_root}/{chrom}_target.npz"
        if not all(os.path.exists(f) for f in cool_files + [target_file]):
            print(f"[Skipping] {chrom} due to missing files")
            continue
        base = MatrixDataset(
            cool_files, target_file,
            matrix_size=matrix_size, step_size=step_size,
            log_transform=False, symmetric=True, jitter=False
        )
        datasets.append(ExpandReplicates(base) if expand_reps else base)

    if not datasets:
        raise RuntimeError("No chromosomes found with all required files.")

    ds = torch.utils.data.ConcatDataset(datasets)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=False)
    return loader, ds

def make_batch_adapter(dataloader, expected_c=1):
    it = iter(dataloader)
    x0, y0 = next(it)
    print("[Adapter] Sample batch:")
    print("  x0", describe(x0))
    print("  y0", describe(y0))
    logged = {"done": False}

    def _to_nchw1(t):
        if isinstance(t, np.ndarray): t = torch.from_numpy(t)
        if t.ndim == 3: t = t.unsqueeze(1)
        if t.ndim != 4: raise ValueError(f"Unexpected ndim={t.ndim}")
        if t.shape[1] != 1:
            t = t[:, :1] if t.shape[1] > 1 else F.pad(t, (0,0,0,0,0,1-t.shape[1]))
        return t.contiguous().float()

    def adapter(x, y):
        x, y = _to_nchw1(x), _to_nchw1(y)
        if not logged["done"]:
            print("[Adapter] After:", f"x->{tuple(x.shape)}", f"y->{tuple(y.shape)}")
            logged["done"] = True
        return x, y

    _ = adapter(x0, y0)
    return adapter



def fix_pred_layout(pred: torch.Tensor, expected_hw: int) -> torch.Tensor:
    if not isinstance(pred, torch.Tensor):
        return pred
    t = pred
    if t.ndim == 4 and t.shape[1] not in (1, 2, 3) and t.shape[-1] in (1, 2, 3):
        t = t.permute(0, 3, 1, 2).contiguous()
    if t.ndim == 4:
        N, C, H, W = t.shape
        if H == 1 and W == expected_hw and (N % expected_hw) == 0:
            B = N // expected_hw
            t = t.reshape(B, C, expected_hw, expected_hw)
    if t.ndim == 4 and t.shape[1] > 1:
        t = t[:, :1]
    return t

# -----------------------------
# Visualization helpers (cooler-style, shared)
# -----------------------------
# for sanity check only
def first_conv_and_parent(model: nn.Module):
    m = model.module if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else model
    for name, mod in m.named_modules():
        if isinstance(mod, nn.Conv2d):
            return name, mod
    return None, None

def conv1_checksum(conv: nn.Conv2d):
    w = conv.weight.detach().float().view(-1)
    return float(w.abs().mean().cpu()), float(w.pow(2).mean().sqrt().cpu())

# -----------------------------

def draw_heatmap_array(matrix: np.ndarray, color_scale: float = 0.0, min_val: float = 1.001):
    m = np.asarray(matrix, dtype=float)
    if color_scale != 0:
        breaks = np.append(np.arange(min_val, color_scale, (color_scale - min_val) / 18), np.max(m))
    elif np.max(m) < 2:
        if np.max(m) <= min_val:
            breaks = np.linspace(min_val, min_val + 1e-3, 20)
        else:
            breaks = np.arange(min_val, np.max(m), (np.max(m) - min_val) / 19)
    else:
        step = (np.quantile(m, q=0.98) - 1) / 18
        up = np.quantile(m, q=0.98) + 0.011
        if up < 2:
            up = 2
            step = 0.999 / 18
        breaks = np.append(np.arange(min_val, up, step), np.max(m) + 0.01)

    colors = ["#FFFFFF", "#FFE4E4", "#FFD7D7", "#FFC9C9", "#FFBCBC", "#FFAEAE", "#FFA1A1", "#FF9494", "#FF8686",
              "#FF7979", "#FF6B6B", "#FF5E5E", "#FF5151", "#FF4343", "#FF3636", "#FF2828", "#FF1B1B", "#FF0D0D",
              "#FF0000"]
    cm = matplotlib.colors.LinearSegmentedColormap.from_list('deeploop', colors, N=20)
    norm = matplotlib.colors.BoundaryNorm(breaks, 20)

    fig, ax = plt.subplots(figsize=(2, 2))
    ax.imshow(m, cmap=cm, norm=norm, interpolation=None)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(pad=0.02)

    fig.canvas.draw()
    try:
        buf = fig.canvas.tostring_rgb()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, -1)[..., :3]
    except Exception:
        buf = fig.canvas.buffer_rgba()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, -1)[..., :3]
    plt.close(fig)
    return img

def grid_panel_sym(v_in, v_out, v_tgt, out_png, color_scale=0.0, min_val=1.001):
    """
    Save panel Input|Pred|Target using cooler-style palette.
    v_in:  [B,R,H,W] -> show first replicate
    v_out: [B,1,H,W] (symmetrized here)
    v_tgt: [B,1,H,W]
    """
    n = min(4, v_in.shape[0])
    fig, axs = plt.subplots(n, 3, figsize=(6, 2*n))
    for i in range(n):
        inp  = v_in[i, 0]
        pred = v_out[i, 0]
        targ = v_tgt[i, 0]
        pred = 0.5 * (pred + pred.T)

        im0 = draw_heatmap_array(inp,  color_scale=color_scale, min_val=min_val)
        im1 = draw_heatmap_array(pred, color_scale=color_scale, min_val=min_val)
        im2 = draw_heatmap_array(targ, color_scale=color_scale, min_val=min_val)

        axs[i,0].imshow(im0); axs[i,0].set_title("Input");  axs[i,0].axis("off")
        axs[i,1].imshow(im1); axs[i,1].set_title("Pred");   axs[i,1].axis("off")
        axs[i,2].imshow(im2); axs[i,2].set_title("Target"); axs[i,2].axis("off")
    plt.tight_layout()
    pathlib.Path(os.path.dirname(out_png)).mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160); plt.close()

# ====== First-conv filter & activation visualization ======
import math

def _unwrap_model(m: nn.Module) -> nn.Module:
    # handle DataParallel/Distributed
    return m.module if isinstance(m, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else m

def get_first_conv(m: nn.Module) -> nn.Conv2d | None:
    m = _unwrap_model(m)
    for _, mod in m.named_modules():
        if isinstance(mod, nn.Conv2d):
            return mod
    return None

def save_first_conv_filters(model: nn.Module, out_png: str, max_cols: int = 8):
    conv = get_first_conv(model)
    if conv is None or conv.weight is None:
        print("[Filters] No Conv2d found; skip.")
        return
    W = conv.weight.detach().cpu().numpy()  # [out_c, in_c, k, k]
    out_c, in_c, kh, kw = W.shape
    # show the slice of the first input channel
    mats = W[:, 0] if in_c >= 1 else W[:, 0]  # [out_c, kh, kw]

    # normalize each filter for display
    def _norm(x):
        x = x.copy()
        lo, hi = x.min(), x.max()
        if hi > lo:
            x = (x - lo) / (hi - lo)
        else:
            x = np.zeros_like(x)
        return x

    mats = np.stack([_norm(m) for m in mats], axis=0)

    # grid layout
    n = mats.shape[0]
    cols = min(max_cols, n)
    rows = math.ceil(n / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(cols*1.2, rows*1.2))
    axs = np.atleast_2d(axs)
    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axs[r, c]
            ax.axis("off")
            if idx < n:
                ax.imshow(mats[idx], cmap="RdGy_r", interpolation="nearest")
            idx += 1
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout(pad=0.1)
    plt.savefig(out_png, dpi=160)
    plt.close(fig)

def save_first_conv_activations(model: nn.Module, x: torch.Tensor, out_png: str, max_maps: int = 32, reduce="mean"):
    """
    x: [B,1,H,W] – we’ll run a forward with a hook on the first conv to capture its output.
    reduce: 'mean' -> average across batch; 'first' -> take feature maps of the first sample.
    """
    conv = get_first_conv(model)
    if conv is None:
        print("[Act] No Conv2d found; skip.")
        return

    acts = {}
    def _hook(_m, _inp, out):
        acts["y"] = out.detach().cpu()  # [B,C,H,W]
    h = conv.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            _ = model(x[:min(4, x.size(0))])  # small forward pass
    finally:
        h.remove()
    if "y" not in acts:
        print("[Act] Hook captured nothing; skip.")
        return

    A = acts["y"]  # [B,C,H,W]
    if reduce == "mean":
        A = A.mean(dim=0)  # [C,H,W]
    else:
        A = A[0]           # [C,H,W]

    C = A.shape[0]
    C = min(C, max_maps)
    mats = A[:C].numpy()

    # normalize each map
    def _norm(x):
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo + 1e-12)

    mats = np.stack([_norm(m) for m in mats], axis=0)

    cols = min(8, C)
    rows = math.ceil(C / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(cols*1.2, rows*1.2))
    axs = np.atleast_2d(axs)
    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axs[r, c]
            ax.axis("off")
            if idx < C:
                ax.imshow(mats[idx], cmap="viridis", interpolation="nearest")
            idx += 1
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout(pad=0.1)
    plt.savefig(out_png, dpi=160)
    plt.close(fig)



def wandb_image_panel(wandb, v_in, v_out, v_tgt, tag="train/preview"):
    imgs = []
    n = min(4, v_in.shape[0])
    for i in range(n):
        panel = np.concatenate([v_in[i,0], v_out[i,0], v_tgt[i,0]], axis=1)
        imgs.append(wandb.Image(panel, caption=f"{tag}_row{i} [Input|Pred|Target]"))
    return imgs

# -----------------------------
# Training / Eval
# -----------------------------
def count_params(model):
    t = sum(p.numel() for p in model.parameters())
    l = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return t, l

def maybe_freeze(model, freeze_all=False, freeze_first_n=0):
    if freeze_all:
        for p in model.parameters():
            p.requires_grad = False
        return
    if freeze_first_n > 0:
        frozen = 0
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.ConvTranspose2d)):
                for p in m.parameters(recurse=False):
                    p.requires_grad = False
                frozen += 1
                if frozen >= freeze_first_n:
                    break

def lr_wsd(step, *, base_lr, warmup, stable, decay, min_lr_ratio=0.05, decay_type="cosine"):
    warmup = max(0, int(warmup))
    stable = max(0, int(stable))
    decay  = max(0, int(decay))
    min_lr = base_lr * float(min_lr_ratio)
    w_end = warmup - 1
    s_end = warmup + stable - 1
    d_end = warmup + stable + decay - 1
    if warmup > 0 and step <= w_end:
        return base_lr * float(step + 1) / float(warmup)
    if stable > 0 and step <= s_end:
        return base_lr
    if decay > 0 and step <= d_end:
        t = step - (warmup + stable)
        if decay_type == "cosine":
            cos_term = 0.5 * (1.0 + math.cos(math.pi * t / max(1, decay)))
            frac = min_lr_ratio + (1.0 - min_lr_ratio) * cos_term
            return base_lr * frac
        elif decay_type == "exponential":
            frac = min_lr_ratio ** (t / max(1.0, float(decay)))
            return base_lr * frac
        else:
            frac = 1.0 - (1.0 - min_lr_ratio) * (t / float(max(1, decay)))
            return base_lr * frac
    return min_lr

def train_one_epoch(model, loader, adapter, device, scaler, opt, cfg, global_step, wandb_run, persistent_it=None):
    model.train()
    running = 0.0
    l1 = nn.L1Loss(reduction="mean")
    steps = cfg["steps_per_epoch"]
    viz_dir = cfg["viz_dir"]
    matrix_size = cfg["matrix_size"]
    local_iter = None if persistent_it is not None else iter(loader)

    for step in range(steps):
        while True:
            try:
                x, y = next(persistent_it) if persistent_it is not None else next(local_iter)
                break
            except StopIteration:
                if persistent_it is None:
                    local_iter = iter(loader)
                else:
                    print("[Coverage] All (item,rep) covered once; reshuffling…")
                    new_loader = DataLoader(loader.dataset, batch_size=loader.batch_size,
                                            shuffle=True, num_workers=4, pin_memory=False, drop_last=False)
                    persistent_it = iter(new_loader)
                    cfg["persistent_it"] = persistent_it

        x, y = adapter(x, y)
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
    
        lr_now = lr_wsd(
            global_step, base_lr=cfg["lr"], warmup=cfg["warmup"],
            stable=cfg["stable"], decay=cfg["decay"],
            min_lr_ratio=cfg["min_lr_ratio"], decay_type=cfg["decay_type"],
        )
        for pg in opt.param_groups:
            pg["lr"] = lr_now

        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=cfg["amp"]):
            pred = model(x)
            pred = fix_pred_layout(pred, expected_hw=matrix_size)
            loss = l1(pred, y)

        if cfg["amp"]:
            scaler.scale(loss).backward()
            if cfg["grad_clip"] > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            if cfg["grad_clip"] > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()

        running += float(loss.item())
        global_step += 1

        # ... after opt.step() / scaler.step(opt)
        if cfg.get("conv_name") is not None:
            # grab conv again (works with DP)
            _, conv1_now = first_conv_and_parent(model)
            if conv1_now is not None:
                mean_abs, rms = conv1_checksum(conv1_now)
                prev = cfg.get("conv_prev_cksum", None)
                if prev is not None:
                    d_mean = abs(mean_abs - prev[0])
                    d_rms  = abs(rms - prev[1])
                else:
                    d_mean = d_rms = float("nan")
                cfg["conv_prev_cksum"] = (mean_abs, rms)

                if wandb_run is not None and (step % cfg["log_every"] == 0):
                    wandb_run.log({
                        "train/conv1_mean_abs": mean_abs,
                        "train/conv1_rms": rms,
                        "train/conv1_d_mean": d_mean,
                        "train/conv1_d_rms": d_rms,
                    }, step=global_step)


        # in train_one_epoch, after `loss.item()`:
        if step % cfg["log_every"] == 0:
            conv = get_first_conv(model)
            if conv is not None and conv.weight.grad is not None:
                grad_norm = float(conv.weight.grad.detach().norm().cpu())
                wandb_run and wandb_run.log({"train/conv1_grad_norm": grad_norm}, step=global_step)


        if wandb_run is not None and (step % cfg["log_every"] == 0):
            wandb_run.log({
                "train/l1": float(loss.item()),
                "train/lr": lr_now,
                "global_step": global_step
            }, step=global_step)

        if (cfg["viz_every"] > 0) and (global_step % cfg["viz_every"] == 0):
            # new (fixed batch)
            fx, fy = cfg["viz_fixed_batch"]
            vx = fx[:min(4, fx.size(0))].to(device, non_blocking=True)
            vy = fy[:min(4, fy.size(0))].to(device, non_blocking=True)
            with torch.no_grad():
                vpred = model(vx)
                vpred = fix_pred_layout(vpred, expected_hw=matrix_size)
            v_in  = vx.detach().cpu().numpy()
            v_out = vpred.detach().cpu().numpy()
            v_tgt = vy.detach().cpu().numpy()
            out_png = f"{viz_dir}/heatmaps/train_step_{global_step}.png"
            grid_panel_sym(v_in, v_out, v_tgt, out_png)

            filt_png = f"{viz_dir}/filters/step_{global_step}.png"
            save_first_conv_filters(model, filt_png)

            act_png = f"{viz_dir}/filters/acts_step_{global_step}.png"
            save_first_conv_activations(model, vx, act_png, max_maps=32, reduce="mean")

            if wandb_run is not None:
                import wandb
                wandb_run.log({
                    "train/images": wandb.Image(out_png),
                    "train/filters": wandb.Image(filt_png),
                    "train/acts": wandb.Image(act_png),
                }, step=global_step)
            
    return running / float(steps), global_step


@torch.no_grad()
def evaluate(model, loader, adapter, device, cfg, global_step, wandb_run):
    model.eval()
    l1 = nn.L1Loss(reduction="mean")
    total = 0.0
    n = 0
    t_iter = iter(loader)
    steps = min(cfg["val_steps"], len(loader))
    matrix_size = cfg["matrix_size"]

    for _ in range(steps):
        try:
            x, y = next(t_iter)
        except StopIteration:
            break
        x, y = adapter(x, y)
        x = x.to(device); y = y.to(device)
        pred = model(x)
        pred = fix_pred_layout(pred, expected_hw=matrix_size)
        total += float(l1(pred, y).item())
        n += 1

    val_loss = total / max(1, n)

    if wandb_run is not None:
        wandb_run.log({"val/l1": val_loss}, step=global_step)
        try:
            import wandb
            # new (fixed batch)
            fx, fy = cfg["viz_fixed_batch"]
            vx = fx[:4].to(device)
            vy = fy[:4].to(device)
            vpred = model(vx)
            vpred = fix_pred_layout(vpred, expected_hw=matrix_size)
            v_in  = vx.detach().cpu().numpy()
            v_out = vpred.detach().cpu().numpy()
            v_tgt = vy.detach().cpu().numpy()

            out_png = f"{cfg['viz_dir']}/heatmaps/val_step_{global_step}.png"
            grid_panel_sym(v_in, v_out, v_tgt, out_png)

            wandb_run.log({"val/images": wandb.Image(out_png)}, step=global_step)
        except Exception as e:
            print("[VAL] W&B image upload failed:", e)

    return val_loss

def save_ckpt(path, model, opt, scaler, meta):
    torch.save({
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "meta": meta
    }, path)

def load_ckpt(path, model, opt=None, scaler=None, map_location="cpu"):
    blob = torch.load(path, map_location=map_location)
    model.load_state_dict(blob["model"], strict=True)
    if opt is not None and "opt" in blob and blob["opt"] is not None:
        opt.load_state_dict(blob["opt"])
    if scaler is not None and "scaler" in blob and blob["scaler"] is not None:
        scaler.load_state_dict(blob["scaler"])
    return blob.get("meta", {})

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser("Fine-tune ONNX-converted Keras denoiser in PyTorch with W&B")
    # Model / IO
    parser.add_argument("--onnx", type=str, required=True, help="Path to cp_gz_denoise.onnx")
    parser.add_argument("--viz-dir", type=str, default="viz_probe")
    parser.add_argument("--ckpt-dir", type=str, default="ckpts")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint .pt to resume")

    # Data
    parser.add_argument("--matrix-size", type=int, default=128, help="Patch size (H=W). Use multiples of 4.")
    parser.add_argument("--step-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--target-root", type=str,
    default="/home/kxt439/DeepLoop2.0/rep_coolers/target_chroms/top_pixels/5p_cap30",
    help="Directory containing {chrom}_target.npz")
    parser.add_argument("--replicate-mode", type=str, default="first",
                    choices=["first", "mean"],
                    help="How to use multiple replicates without fusion: "
                         "'first' uses the first replicate; 'mean' averages all.")



    # Train schedule
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps-per-epoch", type=int, default=500, help="batches per epoch (looping over loader)")
    parser.add_argument("--val-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=0, help="warmup steps (linear ↑)")
    parser.add_argument("--stable", type=int, default=0, help="stable steps (flat)")
    parser.add_argument("--decay",  type=int, default=0, help="decay steps (↓ to min_lr_ratio)")
    parser.add_argument("--min-lr-ratio", type=float, default=0.05,
                        help="final_lr = base_lr * min_lr_ratio after decay")
    parser.add_argument("--decay-type", type=str, default="cosine",
                        choices=["cosine","linear","exponential"])
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--trace", action="store_true", help="attach ShapeTracer for the first forward")

    parser.add_argument("--mode", choices=["pretrain", "finetune"], default="pretrain",
                    help="Training mode. 'pretrain' uses ONNX weights (optionally reset); 'finetune' loads a .pt checkpoint.")
    parser.add_argument("--resume-ckpt", default=None,
                        help="Path to a .pt checkpoint to finetune from (used when --mode finetune).")
    parser.add_argument("--resume-strict", action="store_true",
                        help="Use strict=True when loading --resume-ckpt.")
    parser.add_argument("--reset-params", action="store_true",
                        help="Reinitialize model params before training (useful for fresh pretraining).")
    parser.add_argument("--cover-all-reps", action="store_true",
                        help="Expand each replicate into an individual sample and iterate persistently so all replicates are seen.")


    # Freezing
    parser.add_argument("--freeze-all", action="store_true")
    parser.add_argument("--freeze-first-n", type=int, default=0, help="freeze first N conv/bn/convT modules")

    # Logging / viz
    parser.add_argument("--viz-every", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)

    # W&B
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--project", type=str, default="DeepLoop2_finetune")
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default=None, help="online/offline/disabled")

    # Data parallel
    parser.add_argument("--data-parallel", action="store_true")


    args = parser.parse_args()
    assert args.matrix_size % 4 == 0, "matrix-size must be a multiple of 4."
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 1) Load ONNX → patch → convert
    print(f"[Model] Loading ONNX from: {args.onnx}")
    onnx_model = onnx.load(args.onnx)
    expected_c = get_expected_in_channels_from_onnx(onnx_model)
    print(f"[Model] Expected input channels (from ONNX): {expected_c}")

    onnx_model = fix_same_upper_padding(onnx_model)
    onnx_model = patch_input_chain_to_nchw(onnx_model, expected_hw=args.matrix_size, expected_c=1, verbose=True)

    model = convert(onnx_model).to(device)

    # --- Mode handling ---
    if args.reset_params:
        print("[Init] Resetting model parameters (Kaiming).")
        reset_model_params_(model)

    if args.mode == "finetune":
        if args.resume_ckpt:
            print(f"[Resume] Loading checkpoint: {args.resume_ckpt}")
            blob = torch.load(args.resume_ckpt, map_location=device)
            state = blob.get("model", blob)
            from collections import OrderedDict
            if any(k.startswith("module.") for k in state.keys()):
                state = OrderedDict((k.replace("module.", ""), v) for k, v in state.items())
            model.load_state_dict(state, strict=args.resume_strict)
        else:
            print("[Finetune] No --resume-ckpt provided; starting from ONNX weights (equivalent to pretrain-from-ONNX).")

    if args.freeze_all or args.freeze_first_n > 0:
        maybe_freeze(model, args.freeze_all, args.freeze_first_n)

    tot, trainable = count_params(model)
    print(f"[Model] Params total={tot:,d}, trainable={trainable:,d}")

    # after model is ready
    conv_name, conv1 = first_conv_and_parent(model)
    print(f"[DEBUG] first conv picked for viz: {conv_name or '<none>'}")
    conv_prev_cksum = conv1_checksum(conv1) if conv1 is not None else None

    # 2) Build data loaders
    rep_names = [
        'H9', 'H7', 'HUES1', 'HUES2', 'HUES4', 'HUES8', 'ESI051', 'ESI035', 'ESI017',
        '51213', '51339', 'ACS_1011', 'ACS_1019', 'ACS_1026', 'AG28869', 'CJ-IPS',
        'H1', '51286', 'AG28041', 'DIS1016', 'GM23716', 'GM23720',
        'iPSC_075C', 'iPSC_088B', 'iPSC_1029C', 'iPSC_1106'
    ]
    train_chroms = [f'chr{i}' for i in range(1, 23, 2)]  # odd
    eval_chroms  = [f'chr{i}' for i in range(2, 23, 2)]  # even

    train_loader, train_ds = build_loader(rep_names, train_chroms, args.matrix_size, args.step_size,
                                      args.batch_size, target_root=args.target_root,
                                      shuffle=True, expand_reps=args.cover_all_reps)
    eval_loader,   val_ds   = build_loader(rep_names, eval_chroms,   args.matrix_size, args.step_size,
                                        args.batch_size, target_root=args.target_root,
                                        shuffle=False, expand_reps=False)

    persistent_it = iter(train_loader) if args.cover_all_reps else None


    # 3) Adapter
    adapter = make_batch_adapter(train_loader, expected_c=1)
    print("[Replicates-as-batch] inputs are 1-channel; no fusion layer.")





    if args.data_parallel and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # ---- Fixed viz batch (deterministic across runs) ----
    with torch.no_grad():
        try:
            viz_x0, viz_y0 = next(iter(eval_loader))   # eval has shuffle=False
        except StopIteration:
            viz_x0, viz_y0 = next(iter(train_loader))

    viz_x0, viz_y0 = adapter(viz_x0, viz_y0)
    viz_fixed_batch = (viz_x0.cpu(), viz_y0.cpu())

    # 4) Optimizer / AMP / Resume
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    pathlib.Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)  # FIXED
    ckpt_dir = args.ckpt_dir

    start_epoch = 0
    global_step = 0
    best_val = float("inf")

    if args.resume and os.path.isfile(args.resume):
        meta = load_ckpt(args.resume, model, opt, scaler, map_location=device)
        start_epoch = int(meta.get("epoch", 0))
        global_step = int(meta.get("global_step", 0))
        best_val = float(meta.get("best_val", float("inf")))
        print(f"[Resume] Loaded {args.resume} (epoch={start_epoch}, global_step={global_step}, best_val={best_val:.6f})")

    # 5) W&B
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            if args.wandb_mode: os.environ["WANDB_MODE"] = args.wandb_mode
            wandb_run = wandb.init(project=args.project, entity=args.entity, name=args.run_name,
                                   config=vars(args))
        except Exception as e:
            print("[W&B] init failed, proceeding without W&B:", e)
            wandb_run = None

    # 6) Optional one pass with tracer
    if args.trace:
        tracer = ShapeTracer(max_print=500)
        tracer.attach(model)
        with torch.no_grad():
            x_dbg, y_dbg = next(iter(train_loader))
            x_dbg, y_dbg = adapter(x_dbg, y_dbg)
            _ = model(x_dbg.to(device))
        tracer.detach()

    # 7) Bootstrap visualization to ensure path
    os.makedirs(args.viz_dir, exist_ok=True)
    os.makedirs(os.path.join(args.viz_dir, "heatmaps"), exist_ok=True)
    x0, y0 = next(iter(train_loader))
    x0, y0 = adapter(x0, y0)
    fig, ax = plt.subplots(1, 2, figsize=(4, 2))
    im0 = draw_heatmap_array(x0[0,0].numpy())
    im1 = draw_heatmap_array(y0[0,0].numpy())
    ax[0].imshow(im0); ax[0].set_title("Input0");  ax[0].axis("off")
    ax[1].imshow(im1); ax[1].set_title("Target0"); ax[1].axis("off")
    bootstrap_png = f"{args.viz_dir}/heatmaps/_bootstrap.png"
    plt.tight_layout(); plt.savefig(bootstrap_png, dpi=160); plt.close()
    print(f"[BOOTSTRAP] Wrote {bootstrap_png}")

    # 8) Train / Eval loop
    cfg = dict(
        steps_per_epoch=args.steps_per_epoch,
        val_steps=args.val_steps,
        total_steps=args.epochs * args.steps_per_epoch,
        lr=args.lr,
        min_lr_ratio=args.min_lr_ratio,
        warmup=args.warmup,
        stable=args.stable,
        decay=args.decay,
        decay_type=args.decay_type,
        grad_clip=args.grad_clip,
        amp=args.amp,
        viz_every=args.viz_every,
        log_every=args.log_every,
        viz_dir=args.viz_dir,
        matrix_size=args.matrix_size,
        viz_fixed_batch=viz_fixed_batch,
        conv_name=conv_name,
        conv_prev_cksum=conv_prev_cksum,
    )

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss, global_step = train_one_epoch(
            model, train_loader, adapter, device, scaler, opt, cfg, global_step, wandb_run,
            persistent_it=persistent_it
        )
        if cfg.get("persistent_it", None) is not None:
            persistent_it = cfg["persistent_it"]

        val_loss = evaluate(model, eval_loader, adapter, device, cfg, global_step, wandb_run)

        # after `val_loss = evaluate(...)` in main()
        fx, fy = viz_fixed_batch
        vx = fx[:4].to(device)

        filt_png = f"{cfg['viz_dir']}/filters/val_step_{global_step}.png"
        save_first_conv_filters(model, filt_png)

        act_png = f"{cfg['viz_dir']}/filters/val_acts_step_{global_step}.png"
        save_first_conv_activations(model, vx, act_png, max_maps=32, reduce="mean")

        if wandb_run is not None:
            wandb_run.log({
                "val/filters": wandb.Image(filt_png),
                "val/acts": wandb.Image(act_png),
            }, step=global_step)


        dt = time.time() - t0

        print(f"[Epoch {epoch+1}/{args.epochs}] train_l1={train_loss:.6f} val_l1={val_loss:.6f} ({dt:.1f}s)")

        if wandb_run is not None:
            wandb_run.log({
                "epoch/train_l1": train_loss,
                "epoch/val_l1": val_loss,
                "epoch/time_sec": dt,
                "epoch": epoch+1
            }, step=global_step)

        # Save "last"
        meta = {"epoch": epoch+1, "global_step": global_step, "best_val": best_val}
        save_ckpt(os.path.join(ckpt_dir, "last.pt"), model, opt, scaler, meta)

        # Save "best"
        if val_loss < best_val:
            best_val = val_loss
            meta["best_val"] = best_val
            save_ckpt(os.path.join(ckpt_dir, "best.pt"), model, opt, scaler, meta)
            print(f"[CKPT] New best val {best_val:.6f} → saved best.pt")

    print("✅ Training finished.")

if __name__ == "__main__":
    main()
