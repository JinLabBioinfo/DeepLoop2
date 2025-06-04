import os
import torch
import cooler
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import colors
from torch.utils.data import Dataset
from scipy.sparse import load_npz 


def draw_heatmap(matrix, color_scale, ax=None, return_image=False):
    """
    Display ratio heatmap containing only strong signals (values > 1 or 0.98th quantile)

    Args:
        matrix (:obj:`numpy.array`) : ratio matrix to be displayed
        color_scale (:obj:`int`) : max ratio value to be considered strongest by color mapping
        ax (:obj:`matplotlib.axes.Axes`) : axes which will contain the heatmap.  If None, new axes are created
        return_image (:obj:`bool`) : set to True to return the image obtained from drawing the heatmap with the generated color map

    Returns:
        ``numpy.array`` : if ``return_image`` is set to True, return the heatmap as an array
    """
    if color_scale != 0:
        breaks = np.append(np.arange(1.001, color_scale, (color_scale - 1.001) / 18), np.max(matrix))
    elif np.max(matrix) < 2:
        breaks = np.arange(1.001, np.max(matrix), (np.max(matrix) - 1.001) / 19)
    else:
        step = (np.quantile(matrix, q=0.95) - 1) / 18
        up = np.quantile(matrix, q=0.95) + 0.011
        if up < 2:
            up = 2
            step = 0.999 / 18
        breaks = np.append(np.arange(1.001, up, step), np.max(matrix))

    n_bin = 20  # Discretizes the interpolation into bins
    red_colors = ["#FFFFFF", "#FFE4E4", "#FFD7D7", "#FFC9C9", "#FFBCBC", "#FFAEAE", "#FFA1A1", "#FF9494", "#FF8686",
              "#FF7979", "#FF6B6B", "#FF5E5E", "#FF5151", "#FF4343", "#FF3636", "#FF2828", "#FF1B1B", "#FF0D0D",
              "#FF0000"]
    cmap_name = 'my_list'
    # Create the colormap
    cm = colors.LinearSegmentedColormap.from_list(
        cmap_name, red_colors, N=n_bin)
    norm = colors.BoundaryNorm(breaks, 20)
    # Fewer bins will result in "coarser" colomap interpolation
    if ax is None:
        _, ax = plt.subplots()
    img = ax.imshow(matrix, cmap=cm, norm=norm, interpolation='nearest')
    if return_image:
        plt.close()
        return img.get_array()
    

class MatrixDataset(Dataset):
    def __init__(self, cool_files, target_file, matrix_size=128, step_size=64, max_dist=2000000, n_reps=-1,
                 jitter=True, log_transform=False, symmetric=False):
        self.cool_files = cool_files
        self.mats = [load_npz(c) for c in cool_files]
        chrom = os.path.basename(cool_files[0]).split('.')[0]
        self.target_file = target_file
        self.matrix_size = matrix_size
        self.step_size = step_size
        # get genome size to determine the number of items in the dataset
        # find max distance (in pixels)
        self.max_dist = max_dist
        self.n_reps = n_reps
        self.target_mat = load_npz(target_file)
        self.genome_size = self.target_mat.shape[0]
        self.max_idx = 0
        for i in range(matrix_size, self.genome_size):
            # get mean diagonal value
            diag = self.target_mat.diagonal(i).mean()
            if diag == 0:
                self.max_idx = i - step_size
                print('Max distance:', i)
                break
        
        row_indices = np.arange(0, self.genome_size - self.matrix_size, self.step_size)
        col_indices = np.arange(0, self.genome_size - self.matrix_size, self.step_size)
        if symmetric:
            self.indices = [(i, i) for i in row_indices]
        else:
            # combine row and column indices but exclude pairs that are too far apart
            self.indices = [(i, j) for i in row_indices for j in col_indices if abs(i-j) < self.max_idx]
        drop_indices_file = f'/scratch/users/kxt439/DeepLoop2.0/rep_coolers/{chrom}_drop_indices_symmetric.txt' if symmetric else f'/scratch/users/kxt439/DeepLoop2.0/rep_coolers/{chrom}_drop_indices.txt'
        if not os.path.exists(drop_indices_file):
            drop_indices = []
            for i, j in tqdm(self.indices):
                if np.sum(self.target_mat[i:i+self.matrix_size, j:j+self.matrix_size].data > 1) == 0:
                    drop_indices.append((i, j))
            print('Number of items to drop:', len(drop_indices))
            np.savetxt(drop_indices_file, drop_indices, fmt='%d')
        else:
            drop_indices = np.loadtxt(drop_indices_file, dtype=int)
        try:
            for i, j in drop_indices:
                self.indices.remove((i, j))
        except ValueError as e:
            print(e)
        except TypeError as e:
            print(e)
        self.n_items = len(self.indices)
        print('Number of items in the dataset:', self.n_items)
        self.jitter = jitter
        self.log_transform = log_transform
        


    def __len__(self):
        return self.n_items

    def __getitem__(self, idx):
        reps = []
        i, j = self.indices[idx]
        if self.jitter:
            if self.matrix_size < i < self.genome_size - self.matrix_size:
                i += np.random.randint(-self.step_size, self.step_size)
            if self.matrix_size < j < self.genome_size - self.matrix_size:
                j += np.random.randint(-self.step_size, self.step_size) 
        cooler_list = self.mats
        if self.n_reps > 0:
            cooler_list = np.random.choice(self.mats, self.n_reps)
        for c in cooler_list:
            mat = c[i:i+self.matrix_size, j:j+self.matrix_size].toarray()
            pad_h = self.matrix_size - mat.shape[0]
            pad_w = self.matrix_size - mat.shape[1]
            if pad_h > 0 or pad_w > 0:
                mat = np.pad(mat, ((0, pad_h), (0, pad_w)), mode='constant')
            reps.append(mat)

        #for c in cooler_list
        #    mat = c[i:i+self.matrix_size, j:j+self.matrix_size]
        #    reps.append(mat.toarray())
        reps = np.stack(reps)
        
        
        # get target matrix
        target = self.target_mat[i:i+self.matrix_size, j:j+self.matrix_size].toarray()
        pad_h = self.matrix_size - target.shape[0]
        pad_w = self.matrix_size - target.shape[1]
        if pad_h > 0 or pad_w > 0:
            target = np.pad(target, ((0, pad_h), (0, pad_w)), mode='constant')

        #target = self.target_mat[i:i+self.matrix_size, j:j+self.matrix_size]
        #target = target.toarray()
        #target[target < 1.2] = 0
        reps = torch.tensor(reps, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)
        if self.log_transform:
            reps = torch.log1p(reps)
            target = torch.log1p(target)
        return reps, target
