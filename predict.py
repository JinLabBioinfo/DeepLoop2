import os
import sys
import argparse
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from scipy.sparse import csr_matrix, triu
import torch
import torch.nn as nn
# from unet import UNet
from scipy.sparse import coo_matrix
import onnx
from onnx2torch import convert
from onnx import helper, numpy_helper
import numpy as np
import torch



def anchor_list_to_dict(anchors):
    anchor_dict = {}
    for i, anchor in enumerate(anchors):
        anchor_dict[anchor] = i
    return anchor_dict


def anchor_to_locus(anchor_dict):
    def f(anchor):
        return anchor_dict[anchor]
    return f


def locus_to_anchor(anchor_list):
    def f(locus):
        return anchor_list[locus]

    return f

def sparse_symmetrize(rows, cols, data, shape):
    """
    Given arrays of (row, col, value) for a sparse matrix,
    return a COO sparse matrix symmetrized and averaged exactly
    like (M + M.T) * 0.5 would do in dense land.
    """
    # 1) Duplicate coordinates for (i,j) and (j,i)
    rows2 = np.concatenate([rows, cols])
    cols2 = np.concatenate([cols, rows])
    # 2) Duplicate values and apply the 0.5 factor
    data2 = np.concatenate([data, data]) * 0.5

    # 3) Build COO, then sum any duplicate entries
    coo = coo_matrix((data2, (rows2, cols2)), shape=shape)
    coo.sum_duplicates()
    return coo

def sparse_prediction_from_file(
            model,
            matrix,
            anchor_list,
            small_matrix_size=256,
            step_size=128,
            max_dist=768,
            keep_zeros=True):
    
    input_matrix_size = len(anchor_list)
    # accumulate dense sums and counts
    denoised_matrix = np.zeros_like(matrix.toarray())
    overlap_counts   = np.zeros_like(matrix.toarray())

    # build tile index list
    tile_indices = [
        (i, j)
        for i in range(0, input_matrix_size, step_size)
        for j in range(0, input_matrix_size, step_size)
        if abs(i - j) <= max_dist
    ]
    print(f"Number of tiles: {len(tile_indices)}")

    for i, j in tqdm(tile_indices, desc="Predicting tiles", unit="tile"):
        # define slices
        i0 = min(i, input_matrix_size - small_matrix_size)
        j0 = min(j, input_matrix_size - small_matrix_size)
        rows = slice(i0, i0 + small_matrix_size)
        cols = slice(j0, j0 + small_matrix_size)

        tile = matrix[rows, cols].toarray()
        if tile.shape == (small_matrix_size, small_matrix_size):
            # x = torch.tensor(tile, dtype=torch.float32)\
            #          .unsqueeze(0).unsqueeze(0)
            # denoised, _, _ = model(x)
            # d = denoised.reshape(small_matrix_size, small_matrix_size)\
            #             .detach().cpu().numpy()
            # d[d < 0] = 0

            x = (torch.tensor(tile, dtype=torch.float32)
                    .unsqueeze(0).unsqueeze(0)
                    .to(next(model.parameters()).device))   # move to same device

            with torch.inference_mode():
                y = forward_denoised(model, x)              # <-- use the shim

            d = (y.reshape(small_matrix_size, small_matrix_size)
                .detach().cpu().numpy())
            d[d < 0] = 0


            denoised_matrix[rows, cols] += d
            overlap_counts[rows, cols]   += 1

    # average overlapping predictions
    denoised_matrix = np.divide(
        denoised_matrix,
        overlap_counts,
        out=np.zeros_like(denoised_matrix),
        where=overlap_counts != 0
    )
    # zero the diagonal
    np.fill_diagonal(denoised_matrix, 0)

    # extract upper‐triangular coords & values
    rows, cols = np.triu_indices_from(denoised_matrix)
    values     = denoised_matrix[rows, cols]
    shape      = denoised_matrix.shape

    # build sparse symmetric matrix
    sparse_denoised = sparse_symmetrize(rows, cols, values, shape)
    sparse_denoised = triu(sparse_denoised, format='coo')

    if not keep_zeros:
        sparse_denoised.eliminate_zeros()

    return sparse_denoised


def predict_and_write(model,
                      full_matrix_dir,
                      input_name,
                      out_dir,
                      anchor_dir,
                      chromosome,
                      small_matrix_size,
                      step_size,
                      dummy=5,
                      max_dist=768, # 384
                      val_cols=['obs', 'exp'],
                      keep_zeros=True,
                      matrices_per_tile=8):
    start_time = time.time()
    if os.path.isdir(anchor_dir):
        anchor_file = os.path.join(anchor_dir, chromosome + '.bed')
    else:
        anchor_file = anchor_dir

    anchor_list_df = pd.read_csv(anchor_file, sep='\t', usecols=[0, 1, 2, 3], names=['chr', 'start', 'end', 'anchor'])  # read anchor list file
    anchor_list_df = anchor_list_df[anchor_list_df['chr'] == chromosome].reset_index(drop=True)
    anchor_list = anchor_list_df['anchor'].values.tolist()
    anchor_dict = anchor_list_to_dict(anchor_list)
    # filter to chromosome
    start_time = time.time()
    chr_anchor_file = pd.read_csv(
        os.path.join(full_matrix_dir, input_name),
        delimiter='\t',
        names=['anchor1', 'anchor2'] + val_cols,
        usecols=['anchor1', 'anchor2'] + val_cols)  # read chromosome anchor to anchor file

    if 'ratio' not in val_cols:
        if 'obs' in val_cols and 'exp' in val_cols:
            chr_anchor_file['ratio'] = (chr_anchor_file['obs'] + dummy) / (
                        chr_anchor_file['exp'] + dummy)  # Compute matrix ratio value
        else:
            raise ValueError("Must provide either 'ratio' or both 'obs' and 'exp' columns to compute ratio.")
    else:
        print("Using precomputed 'ratio' column from input file.")

    chr_anchor_file = chr_anchor_file[
        (chr_anchor_file['anchor1'].isin(anchor_dict)) & (chr_anchor_file['anchor2'].isin(anchor_dict))
    ]

    rows = np.vectorize(anchor_to_locus(anchor_dict))(chr_anchor_file['anchor1'].values)
    cols = np.vectorize(anchor_to_locus(anchor_dict))(chr_anchor_file['anchor2'].values)
    data = chr_anchor_file['ratio'].values
    input_matrix_size = len(anchor_list)

    sparse_matrix = csr_matrix((data, (rows, cols)), shape=(input_matrix_size, input_matrix_size))

    sparse_denoised = sparse_prediction_from_file(model,
                                                  sparse_matrix,
                                                  anchor_list,
                                                  small_matrix_size,
                                                  step_size,
                                                  max_dist,
                                                  keep_zeros=keep_zeros)

    if len(sparse_denoised.row) > 0:
        anchor_1_list = np.vectorize(locus_to_anchor(anchor_list))(sparse_denoised.row)
        anchor_2_list = np.vectorize(locus_to_anchor(anchor_list))(sparse_denoised.col)

        output_df = pd.DataFrame({
            'anchor1': anchor_1_list,
            'anchor2': anchor_2_list,
            'denoised': sparse_denoised.data
        }).round({'denoised': 4})

        output_df.to_csv(os.path.join(out_dir, chromosome + '.denoised.anchor.to.anchor'),
                         sep='\t', index=False, header=False)
    else:
        print("No entries in denoised matrix.")

    print('Denoised whole matrix in %d seconds' % (time.time() - start_time))

# added helpers
# --- ONNX -> Torch build helpers ---
def _fix_same_upper_padding(om):
    for node in om.graph.node:
        if node.op_type not in ["Conv","ConvTranspose","MaxPool"]:
            continue
        rm = None
        for i,a in enumerate(node.attribute):
            if a.name=="auto_pad" and getattr(a, "s", b"")==b"SAME_UPPER":
                rm = i; break
        if rm is None: 
            continue
        del node.attribute[rm]
        kshape = None
        for a in node.attribute:
            if a.name=="kernel_shape": 
                kshape = list(a.ints)
        pads = [0,0,0,0]
        if kshape:
            pads2=[]
            for k in kshape:
                p=(k-1)//2
                pads2.extend([p,p])
            pads = pads2 if len(pads2)==4 else [pads2[0], pads2[1], pads2[0], pads2[1]]
        node.attribute.append(helper.make_attribute("pads", pads))
    return om

def _patch_input_chain_to_nchw(m, expected_hw, expected_c=1):
    out2node={}
    for n in m.graph.node:
        for o in n.output: out2node[o]=n
    conv = next((n for n in m.graph.node if n.op_type=="Conv"), None)
    if conv is None: 
        return m
    data_in = conv.input[0]
    chain=[]; cur=data_in
    for _ in range(12):
        prod = out2node.get(cur, None)
        if prod is None: break
        if prod.op_type in ("Reshape","Transpose","Squeeze","Unsqueeze"):
            chain.append(prod); cur = prod.input[0] if prod.input else cur; continue
        break
    if not chain: 
        return m
    init_map = {t.name:t for t in m.graph.initializer}
    n0 = chain[0]
    if n0.op_type=="Transpose":
        found=False
        for a in n0.attribute:
            if a.name=="perm": 
                a.ints[:] = [0,1,2,3]; found=True; break
        if not found: 
            n0.attribute.append(helper.make_attribute("perm",[0,1,2,3]))
    nearest_reshape = next((n for n in chain if n.op_type=="Reshape"), None)
    if nearest_reshape is None or len(nearest_reshape.input)<2: 
        return m
    shape_name = nearest_reshape.input[1]
    target = np.array([-1,int(expected_c),int(expected_hw),int(expected_hw)], dtype=np.int64)
    if shape_name in init_map:
        new_t = numpy_helper.from_array(target, name=shape_name)
        idx = [i for i,t in enumerate(m.graph.initializer) if t.name==shape_name][0]
        m.graph.initializer[idx].CopyFrom(new_t)
    else:
        new_name = shape_name + "__patched_static_nchw"
        m.graph.initializer.append(numpy_helper.from_array(target, name=new_name))
        nearest_reshape.input[1] = new_name
    return m

def build_model_from_onnx(onnx_path, matrix_size=128, expected_c=1, device="cuda"):
    om = onnx.load(onnx_path)
    om = _fix_same_upper_padding(om)
    om = _patch_input_chain_to_nchw(om, expected_hw=matrix_size, expected_c=expected_c)
    model = convert(om).to(device).eval()
    return model

def load_initializers_checkpoint(model, ckpt_path, device="cuda"):
    import torch
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # unwrap wrapper dicts: {'model': state_dict, 'opt': ..., 'scaler': ..., 'meta': ...}
    if isinstance(blob, dict) and "model" in blob and isinstance(blob["model"], dict):
        sd = blob["model"]
    elif isinstance(blob, dict) and all(isinstance(v, torch.Tensor) for v in blob.values()):
        sd = blob
    elif hasattr(blob, "state_dict"):
        sd = blob.state_dict()
    else:
        raise RuntimeError("Unsupported checkpoint format (no tensor state_dict found).")

    # strip DDP prefix
    sd = {k.replace("module.", ""): v for k, v in sd.items()}

    # backfill onnx2torch initializers so strict=False is clean
    fresh = model.state_dict()
    for k in fresh.keys():
        if k.startswith("initializers.") and k not in sd:
            sd[k] = fresh[k]

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print("[load] missing (sample):", missing[:10], " unexpected (sample):", unexpected[:10])

    return model.to(device).eval()

# Forward shim to normalize NHWC → NCHW and pick first channel
def forward_denoised(model, x):
    y = model(x)
    if isinstance(y,(list,tuple)): y = y[0]
    if isinstance(y,dict):
        for k in ("denoised","out","logits"):
            if k in y: y = y[k]; break
    if y.ndim==4 and y.shape[1] not in (1,2,3) and y.shape[-1] in (1,2,3):
        y = y.permute(0,3,1,2).contiguous()
    if y.ndim==4 and y.shape[1] > 1:
        y = y[:, :1]
    return y

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full_matrix_dir', type=str, help='directory containing chromosome interaction files to be used as input')
    # ADD (keep your existing args; this adds the ONNX path)
    parser.add_argument("--onnx", required=True, help="Reference ONNX file matching the new checkpoint architecture")
    parser.add_argument('--input_name', type=str, help='name of file in full_matrix_dir that we want to feed into model')
    parser.add_argument('--weights', type=str, help='path to model weights .pt file')
    parser.add_argument('--out_dir', type=str, help='directory where the output interaction file will be stored')
    parser.add_argument('--anchors', type=str, help='directory/file containing anchor .bed reference files')
    parser.add_argument('--chromosome', type=str, help='chromosome string (e.g chr1, chr20, chrX)')
    parser.add_argument('--small_matrix_size', type=int, default=128, help='size of input tiles (symmetric)')
    parser.add_argument('--step_size', type=int, default=128, help='step size when tiling matrix (overlapping values will be averaged if different)')
    parser.add_argument('--max_dist', type=int, default=384, help='maximum distance from diagonal (in pixels) where we consider interactions (default to ~2Mb)')
    parser.add_argument('--dummy', type=int, default=5, help='dummy value to compute ratio (obs + dummy) / (exp + dummy)')
    parser.add_argument('--val_cols','--list', nargs='+', help='names of value columns in interaction files (not including a1, a2)', default=['obs', 'exp'])
    parser.add_argument('--keep_zeros', action='store_true', help='if provided, the output file will contain all pixels in every tile, even if no value is present')
    args = parser.parse_args()

    full_matrix_dir = args.full_matrix_dir
    input_name = args.input_name
    out_dir = args.out_dir
    anchor_dir = args.anchors
    chromosome = args.chromosome
    small_matrix_size = args.small_matrix_size
    step_size = args.step_size
    dummy = args.dummy
    max_dist = args.max_dist
    val_cols = args.val_cols
    keep_zeros = args.keep_zeros

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(out_dir, exist_ok=True)

    # REPLACE your old model = UNet(...); model.load_state_dict(...)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model_from_onnx(args.onnx, matrix_size=args.small_matrix_size, expected_c=1, device=device)
    model = load_initializers_checkpoint(model, args.weights, device=device)

    predict_and_write(model, full_matrix_dir, input_name, out_dir, anchor_dir, chromosome, small_matrix_size, step_size,
                      dummy, max_dist, val_cols, keep_zeros)