import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from scipy.sparse import csr_matrix, coo_matrix, triu

from dense_unet import DenseUNet


def anchor_list_to_dict(anchor_list):
    return {a: i for i, a in enumerate(anchor_list)}

def anchor_to_locus(anchor_dict):
    return lambda a: anchor_dict[a]

def locus_to_anchor(anchor_list):
    return lambda i: anchor_list[int(i)]

def _load_state_dict_robust(weights_path: str):
    """
    Robustly load a pytorch state dict:
      - supports ckpt with {"state_dict": ...}
      - strips 'module.' prefix
    """
    ckpt = torch.load(weights_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    new_state = {}
    for k, v in state.items():
        new_state[k.replace("module.", "")] = v
    return new_state

def sparse_symmetrize(rows, cols, values, shape):
    rows2 = np.concatenate([rows, cols])
    cols2 = np.concatenate([cols, rows])
    data2 = np.concatenate([values, values])
    coo = coo_matrix((data2, (rows2, cols2)), shape=shape)
    coo.sum_duplicates()
    return coo

def sparse_prediction_from_matrix_oomsafe(
    model,
    matrix: csr_matrix,
    anchor_list,
    small_matrix_size: int,
    step_size: int,
    max_dist: int,
    keep_zeros: bool,
    use_float16: bool = False,
):
    device = next(model.parameters()).device
    n = len(anchor_list)

    band_w = int(max_dist) + int(small_matrix_size)
    band_sum = np.zeros((n, band_w), dtype=np.float32)
    band_cnt = np.zeros((n, band_w), dtype=np.uint16)

    tile_indices = [
        (i, j)
        for i in range(0, n, step_size)
        for j in range(i, n, step_size)
        if (j - i) <= max_dist
    ]
    print(f"Number of tiles: {len(tile_indices)}")

    model.eval()
    with torch.no_grad():
        for i, j in tqdm(tile_indices, desc="Predicting tiles", unit="tile", ncols=100):
            i0 = min(i, n - small_matrix_size)
            j0 = min(j, n - small_matrix_size)
            if i0 < 0 or j0 < 0:
                continue

            rows = slice(i0, i0 + small_matrix_size)
            cols = slice(j0, j0 + small_matrix_size)

            tile = matrix[rows, cols].toarray().astype(np.float32, copy=False)
            if tile.shape != (small_matrix_size, small_matrix_size):
                continue

            x = torch.from_numpy(tile)[None, None, :, :].to(device, non_blocking=True)
            if use_float16:
                x = x.half()

            logits, _, _ = model(x)
            if use_float16:
                logits = logits.float()

            d = logits.squeeze().detach().cpu().numpy().astype(np.float32, copy=False)
            d[d < 0] = 0

            rr, cc = np.indices((small_matrix_size, small_matrix_size))
            r = i0 + rr
            c = j0 + cc
            dist = c - r

            m = (dist >= 0) & (dist < band_w)
            if not np.any(m):
                continue

            r_m = r[m].astype(np.int64, copy=False)
            dist_m = dist[m].astype(np.int64, copy=False)
            val_m = d[m]

            np.add.at(band_sum, (r_m, dist_m), val_m)
            np.add.at(band_cnt, (r_m, dist_m), 1)

    covered = band_cnt > 0
    band_avg = np.zeros_like(band_sum, dtype=np.float32)
    band_avg[covered] = band_sum[covered] / band_cnt[covered].astype(np.float32)

    if keep_zeros:
        rows = np.repeat(np.arange(n, dtype=np.int64), band_w)
        ds = np.tile(np.arange(band_w, dtype=np.int64), n)
        cols = rows + ds
        vals = band_avg.reshape(-1)

        m2 = cols < n
        rows, cols, vals = rows[m2], cols[m2], vals[m2]
    else:
        rows, ds = np.nonzero(band_cnt > 0)
        cols = rows + ds
        vals = band_avg[rows, ds]

    shape = (n, n)
    # set diagonal to zero (optional, can be kept if desired)
    m_diag = rows == cols
    vals[m_diag] = 0.0
    sparse_denoised = sparse_symmetrize(rows, cols, vals, shape)
    sparse_denoised = triu(sparse_denoised, format="coo")  # keep upper triangle
    if not keep_zeros:
        sparse_denoised.eliminate_zeros()
    return sparse_denoised


def predict_and_write(
    model,
    full_matrix_dir: str,
    input_name: str,
    out_dir: str,
    anchors: str,
    chromosome: str,
    small_matrix_size: int,
    step_size: int,
    dummy: float,
    max_dist: int,
    val_cols: list[str],
    keep_zeros: bool,
    use_float16: bool,
):
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)

    # anchors bed
    if os.path.isdir(anchors):
        anchor_file = os.path.join(anchors, f"{chromosome}.bed")
    else:
        anchor_file = anchors

    anchor_df = pd.read_csv(
        anchor_file,
        sep="\t",
        header=None,
        names=["chrom", "start", "end", "anchor"],
        usecols=[0, 1, 2, 3],
    )
    anchor_df = anchor_df[anchor_df["chrom"] == chromosome].reset_index(drop=True)
    anchor_list = anchor_df["anchor"].astype(str).tolist()
    anchor_dict = anchor_list_to_dict(anchor_list)

    loop_path = os.path.join(full_matrix_dir, input_name)
    df = pd.read_csv(
        loop_path,
        sep="\t",
        header=None,
        names=["a1", "a2"] + val_cols,
        usecols=[0, 1] + list(range(2, 2 + len(val_cols))),
    )
    df["a1"] = df["a1"].astype(str)
    df["a2"] = df["a2"].astype(str)

    if "ratio" in val_cols:
        ratio = df["ratio"].astype(np.float32).to_numpy()
    else:
        if ("obs" not in val_cols) or ("exp" not in val_cols):
            raise ValueError("val_cols must include either 'ratio' OR both 'obs' and 'exp'.")
        obs = df["obs"].astype(np.float32).to_numpy()
        exp = df["exp"].astype(np.float32).to_numpy()
        ratio = (obs + float(dummy)) / (exp + float(dummy))

    m = df["a1"].isin(anchor_dict) & df["a2"].isin(anchor_dict)
    df = df[m].reset_index(drop=True)
    ratio = ratio[m.to_numpy()]

    rows = np.fromiter((anchor_dict[a] for a in df["a1"]), dtype=np.int64, count=len(df))
    cols = np.fromiter((anchor_dict[a] for a in df["a2"]), dtype=np.int64, count=len(df))
    data = ratio.astype(np.float32, copy=False)

    n = len(anchor_list)
    sparse_matrix = csr_matrix((data, (rows, cols)), shape=(n, n))

    sparse_denoised = sparse_prediction_from_matrix_oomsafe(
        model=model,
        matrix=sparse_matrix,
        anchor_list=anchor_list,
        small_matrix_size=small_matrix_size,
        step_size=step_size,
        max_dist=max_dist,
        keep_zeros=keep_zeros,
        use_float16=use_float16,
    )
    print(f"[DEBUG] chromosome={chromosome}")
    print(f"[DEBUG] anchors={len(anchor_list):,}")
    print(f"[DEBUG] input matched entries={len(df):,}")
    print(f"[DEBUG] input sparse nnz={sparse_matrix.nnz:,}")
    print(f"[DEBUG] output sparse nnz={sparse_denoised.nnz:,}")
    if sparse_denoised.nnz == 0:
        print(f"[WARN] No entries in denoised matrix for {chromosome}.")
        return

    a1_list = np.fromiter((anchor_list[i] for i in sparse_denoised.row), dtype=object, count=len(sparse_denoised.row))
    a2_list = np.fromiter((anchor_list[i] for i in sparse_denoised.col), dtype=object, count=len(sparse_denoised.col))

    out_df = pd.DataFrame(
        {"a1": a1_list, "a2": a2_list, "denoised": sparse_denoised.data.astype(np.float32, copy=False)}
    )
    out_path = os.path.join(out_dir, f"{chromosome}.denoised.anchor.to.anchor")
    out_df.to_csv(out_path, sep="\t", index=False, header=False, float_format="%.4f")
    print(f"[OK] wrote: {out_path} (n={len(out_df):,})")
    print(f"[TIME] {chromosome} done in {time.time() - t0:.1f}s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--full_matrix_dir", required=True)
    p.add_argument("--input_name", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--anchors", required=True)
    p.add_argument("--chromosome", required=True)

    p.add_argument("--small_matrix_size", type=int, default=128)
    p.add_argument("--step_size", type=int, default=64)
    p.add_argument("--max_dist", type=int, default=384)

    p.add_argument("--dummy", type=float, default=5.0,
                   help="dummy value for ratio: (obs+dummy)/(exp+dummy)")

    p.add_argument("--val_cols", nargs="+", default=["obs", "exp"],
                   help="value columns after a1 a2, e.g. obs exp  OR ratio")

    p.add_argument("--keep_zeros", action="store_true",
                   help="if set, output keeps zeros inside the covered band")

    p.add_argument("--use_float16", action="store_true",
                   help="use float16 input for less GPU memory")

    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseUNet(out_channels=1).to(device)
    state = _load_state_dict_robust(args.weights)
    load_result = model.load_state_dict(state, strict=False)

    print("[DEBUG MODEL LOAD]")
    print("missing_keys:", load_result.missing_keys)
    print("unexpected_keys:", load_result.unexpected_keys)
    model.eval()

    predict_and_write(
        model=model,
        full_matrix_dir=args.full_matrix_dir,
        input_name=args.input_name,
        out_dir=args.out_dir,
        anchors=args.anchors,
        chromosome=args.chromosome,
        small_matrix_size=args.small_matrix_size,
        step_size=args.step_size,
        dummy=args.dummy,
        max_dist=args.max_dist,
        val_cols=args.val_cols,
        keep_zeros=args.keep_zeros,
        use_float16=args.use_float16,
    )


if __name__ == "__main__":
    main()
