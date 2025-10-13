import os
import math
import wandb
import numpy as np
from tqdm import tqdm
from tqdm.auto import trange
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from skimage.metrics import structural_similarity as ssim


from unet import UNet
from matrix_generator_chrom import MatrixDataset

global_step = 0
lr = 1e-4
batch_size = 8
matrix_size = 128
step_size = 64
train_iter = 150000
num_codes = 64
seed = 42
rotation_trick = True
device = "cuda" if torch.cuda.is_available() else "cpu"

def stratified_corr(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Compute mean Pearson correlation across all diagonals of the contact map.
    pred, true: 2D arrays of shape (H, H).
    """
    H = pred.shape[0]
    corrs = []
    for d in range(H):
        mask = np.eye(H, k=d, dtype=bool)
        p = pred[mask]
        t = true[mask]
        if p.size:
            corrs.append(np.corrcoef(p, t)[0,1])
    return float(np.nanmean(corrs))

def create_wsd_lambda(warmup_steps, stable_steps, decay_steps, min_lr_ratio=0.001, decay_type='cosine'):
    """
    Create a lambda function for use with PyTorch's LambdaLR scheduler
    
    Args:
        warmup_steps: Number of warmup steps
        stable_steps: Number of stable steps  
        decay_steps: Number of decay steps
        min_lr_ratio: Minimum learning rate as ratio of initial lr
        decay_type: 'cosine' or 'exponential'
    
    Returns:
        Lambda function for LambdaLR scheduler
    """
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Warmup phase
            return current_step / warmup_steps
        elif current_step < warmup_steps + stable_steps:
            # Stable phase
            return 1.0
        else:
            # Decay phase
            decay_step = current_step - warmup_steps - stable_steps
            
            if decay_type == 'exponential':
                decay_rate = min_lr_ratio ** (1 / decay_steps)
                return max(decay_rate ** decay_step, min_lr_ratio)
            elif decay_type == 'cosine':
                progress = min(decay_step / decay_steps, 1.0)
                return max(min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)), min_lr_ratio)
            else:
                raise ValueError(f"Unknown decay_type: {decay_type}")
    
    return lr_lambda
    
wsd_lambda = create_wsd_lambda(
        warmup_steps=1000,
        stable_steps=100000,
        decay_steps=20000,
        min_lr_ratio=0.001,
        decay_type='cosine'
)

def train(model, train_loader, eval_loader, train_iterations=1000, lr=1e-4, alpha=10, viz_dir="viz", start_step=0):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt,
        lr_lambda=wsd_lambda
    )
    # scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10000, gamma=0.1)
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(os.path.join(viz_dir, "heatmaps"), exist_ok=True)
    os.makedirs(os.path.join(viz_dir, "filters"), exist_ok=True)
    def iterate_dataset(data_loader):
        data_iter = iter(data_loader)
        while True:
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                x, y = next(data_iter)
            yield x.to(device), y.to(device)

    viz_x, viz_y = next(iterate_dataset(eval_loader))   
    
    global_step = start_step  # ← initialize local counter
    for _ in (pbar := trange(train_iterations)):        
        x, y = next(iterate_dataset(train_loader))
        rec_loss = 0
        cmt_loss = 0
        opt.zero_grad()
        rep_loss = 0
        preds, indices, cmt_loss = model(x.view(-1, 1, matrix_size, matrix_size))
        # --- codebook perplexity ---
        flat_idx = indices.view(-1)
        counts = torch.bincount(flat_idx, minlength=num_codes).float()
        probs = counts / counts.sum()
        perplexity = torch.exp(-torch.sum(probs * torch.log(probs + 1e-10)))

        preds = preds.view(x.shape)
        rep_target = y.unsqueeze(1)
        for rep_i in range(x.shape[1]):
            out = preds[:, rep_i].unsqueeze(1)
            rec_loss += (out - rep_target).abs().mean()
            for rep_j in range(x.shape[1]):  # output should also be similar across replicates
                if rep_i != rep_j:
                    rep2_target = preds[:, rep_j].unsqueeze(1)
                    rep_loss += torch.nn.functional.mse_loss(out, rep2_target)
        rec_loss /= x.shape[1]
        rep_loss /= x.shape[1] * (x.shape[1] - 1)
        (rec_loss + 0.1 * rep_loss + alpha * cmt_loss).backward()
        opt.step()
        wandb.log({
            "rec_loss": rec_loss.item(),
            "rep_loss": rep_loss.item(),
            "commitment_loss": cmt_loss.item(),
            "vqvae/perplexity": perplexity.item(),
            }, step=global_step)
        pbar.set_description(
            f"rec loss: {rec_loss.item():.3f} | "
            #+ f"cmt loss: {cmt_loss.item():.3f} | "
            #+ f"active %: {indices.unique().numel() / num_codes * 100:.3f}"
        )
        if global_step % 1000 == 0:
            rep = viz_x[:, 0].unsqueeze(1).to(device)
            out, _, _ = model(rep)
            x_in = rep.detach().cpu().numpy()
            out = out.detach().cpu().numpy()
            y = viz_y.detach().cpu().numpy()
            # if train_loader.dataset.log_transform:
            #     # undo log1p transformation
            x_in = np.expm1(x_in)
            out = np.expm1(out)
            y = np.expm1(y)

            # SSIM and stratified-corr
            # convert to 2D arrays
            B = out.shape[0]
            ssim_vals       = []
            strat_corr_vals = []
            for b in range(B):
                pred_mat = out[b,0]
                true_mat = y[b]
                # SSIM
                s = ssim(pred_mat, true_mat,
                        data_range=true_mat.max() - true_mat.min())
                ssim_vals.append(s)
                # Distance-stratified Pearson
                strat_corr_vals.append(stratified_corr(pred_mat, true_mat))
            mean_ssim       = float(np.mean(ssim_vals))
            mean_strat_corr = float(np.mean(strat_corr_vals))
            wandb.log({
            "viz/ssim": mean_ssim,
            "viz/stratified_corr": mean_strat_corr,
            }, step=global_step)

            n = min(8, x_in.shape[0])
            fig, axs = plt.subplots(n, 5, figsize=(8, 2*n))
            for batch_i in range(n):
                axs[batch_i][0].imshow(x_in[batch_i, 0], cmap="Reds", vmin=1, vmax=2, interpolation=None)
                #draw_heatmap(rep[batch_i, 0].cpu().numpy(), 0, ax=axs[batch_i][0])
                axs[batch_i][0].set_xticks([])
                axs[batch_i][0].set_yticks([])
                pred = out[batch_i, 0]
                pred = (pred + pred.T) / 2
                axs[batch_i][1].imshow(pred, cmap="Reds", vmin=1, vmax=2, interpolation=None)
                #draw_heatmap(out[batch_i, 0], 0, ax=axs[batch_i][1])
                axs[batch_i][1].set_xticks([])
                axs[batch_i][1].set_yticks([])
                axs[batch_i][2].imshow(pred, cmap="Reds", interpolation=None)
                #draw_heatmap(y[batch_i], 0, ax=axs[batch_i][2])
                axs[batch_i][2].set_xticks([])
                axs[batch_i][2].set_yticks([])
                axs[batch_i][3].imshow(y[batch_i], cmap="Reds", interpolation=None)
                axs[batch_i][3].set_xticks([])
                axs[batch_i][3].set_yticks([])
                axs[batch_i][4].imshow(y[batch_i], cmap="Reds", vmin=1, vmax=2, interpolation=None)
                axs[batch_i][4].set_xticks([])
                axs[batch_i][4].set_yticks([])
            plt.tight_layout()
            plt.savefig(f'{viz_dir}/heatmaps/vqae_train_{global_step // 100}.png')
            plt.close()

            # visualize the first convolutional layer filters
            filters = model.parameters().__next__().detach().cpu().numpy()
            n_filters = model.init_filters
            n_rows = int(math.sqrt(n_filters))
            n_cols = math.ceil(n_filters / n_rows)
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
            for i in range(n_filters):
                axs[i // n_cols, i % n_cols].imshow(filters[i, 0], cmap="RdGy_r")
                axs[i // n_cols, i % n_cols].set_xticks([])
                axs[i // n_cols, i % n_cols].set_yticks([])
            plt.tight_layout()
            plt.savefig(f'{viz_dir}/filters/vqae_train_{global_step // 100}.png')
            plt.close()

            wandb.log({"reconstruction": wandb.Image(f'{viz_dir}/heatmaps/vqae_train_{global_step // 100}.png')})
            torch.save(model.state_dict(), f'{viz_dir}/vqae_train_5p_cap3.pt')
        scheduler.step()
        global_step += 1 

    return global_step

if __name__ == '__main__':
    torch.manual_seed(seed)
    wandb.init(project="hic_vqvae")

    rep_names = [
        'H9', 'H7', 'HUES1', 'HUES2', 'HUES4', 'HUES8', 'ESI051', 'ESI035', 'ESI017',
        '51213', '51339', 'ACS_1011', 'ACS_1019', 'ACS_1026', 'AG28869', 'CJ-IPS',
        'H1', '51286', 'AG28041', 'DIS1016', 'GM23716', 'GM23720',
        'iPSC_075C', 'iPSC_088B', 'iPSC_1029C', 'iPSC_1106'
    ]

    train_chroms = [f'chr{i}' for i in range(1, 23, 2)]  # odd
    eval_chroms = [f'chr{i}' for i in range(2, 23, 2)]   # even

    # Build evaluation loader just once
    eval_datasets = []
    for chrom in eval_chroms:
        cool_files = [f'/home/kxt439/DeepLoop2.0/rep_coolers/chroms/{chrom}/{chrom}_{rep}.npz' for rep in rep_names]
        target_file = f'/home/kxt439/DeepLoop2.0/rep_coolers/target_chroms/top_pixels/5p_cap3/{chrom}_target.npz'
        if all(os.path.exists(f) for f in cool_files + [target_file]):
            eval_data = MatrixDataset(
                cool_files, target_file,
                matrix_size=matrix_size, step_size=step_size,
                log_transform=True,
                jitter=False, symmetric=True
            )
            eval_datasets.append(eval_data)
    eval_loader = DataLoader(
        torch.utils.data.ConcatDataset(eval_datasets),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=False
    )

    # Build model once
    model = UNet(
        n_channels=1, n_classes=1, init_filters=64,
        bilinear=True,
        codebook_size=num_codes,
        rotation_trick=rotation_trick,
        straight_through=False
    ).to(device)
    print('Device:', device)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # Train chromosome by chromosome
    print("Starting chromosome-wise training...")
    chrom_count = 0
    for chrom in train_chroms:
        cool_files = [f'/home/kxt439/DeepLoop2.0/rep_coolers/chroms/{chrom}/{chrom}_{rep}.npz' for rep in rep_names]
        target_file = f'/home/kxt439/DeepLoop2.0/rep_coolers/target_chroms/top_pixels/5p_cap3/{chrom}_target.npz'

        if not all(os.path.exists(f) for f in cool_files + [target_file]):
            print(f"[Skipping] {chrom} due to missing files")
            continue

        print(f"[Loading] {chrom} for training...")
        dataset = MatrixDataset(
            cool_files, target_file,
            matrix_size=matrix_size, step_size=step_size,
            log_transform=True
        )

        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=False
        )

        # Split training iterations equally across chromosomes
        chrom_train_iters = train_iter // len(train_chroms)
        # pass in the current global_step and re-capture it
        global_step = train(
            model,
            train_loader,
            eval_loader,
            train_iterations=chrom_train_iters,
            lr=lr,
            start_step=global_step
        )
        # train(model, train_loader, eval_loader, train_iterations=chrom_train_iters, lr=lr)
        chrom_count += 1

    print(f"Finished training on {chrom_count} chromosomes.")

    # Optional: final reconstruction plots
    x, y = next(iter(train_loader))
    rep = x[:, 0].unsqueeze(1).to(device)
    out, _, _ = model(rep)
    x = np.expm1(rep.detach().cpu().numpy())
    out = np.expm1(out.detach().cpu().numpy())
    y = np.expm1(y.detach().cpu().numpy())

    for batch_i in range(min(4, x.shape[0])):
        fig, axs = plt.subplots(1, 3, figsize=(6, 2))
        axs[0].imshow(x[batch_i, 0], cmap="Reds")
        axs[0].set_title("Input")
        axs[1].imshow(out[batch_i, 0], cmap="Reds")
        axs[1].set_title("Reconstruction")
        axs[2].imshow(y[batch_i], cmap="Reds")
        axs[2].set_title("Target")
        plt.tight_layout()
        plt.savefig(f'vqae_train_post_{batch_i}.png')
        plt.close()
