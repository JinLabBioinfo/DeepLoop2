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
from unet import UNet
from scipy.sparse import coo_matrix


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
            x = torch.tensor(tile, dtype=torch.float32)\
                     .unsqueeze(0).unsqueeze(0)
            denoised, _, _ = model(x)
            d = denoised.reshape(small_matrix_size, small_matrix_size)\
                        .detach().cpu().numpy()
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




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full_matrix_dir', type=str, help='directory containing chromosome interaction files to be used as input')
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

    model = UNet(1, 1, init_filters=64,
                bilinear=True, 
                codebook_size=64, 
                rotation_trick=True, 
                straight_through=False).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=torch.device(device)))
    predict_and_write(model, full_matrix_dir, input_name, out_dir, anchor_dir, chromosome, small_matrix_size, step_size,
                      dummy, max_dist, val_cols, keep_zeros)
