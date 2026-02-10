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

from dense_unet import DenseUNet
from matrix_generator_chrom import MatrixDataset

lr = 1e-4
batch_size = 8
matrix_size = 128
step_size = 64
train_iter = 150000
num_codes = 64
seed = 42
rotation_trick = True
device = "cuda" if torch.cuda.is_available() else "cpu"

def wsd_lr(step, total_steps, base_lr, warm_frac=0.05, stable_frac=0.45, final_lr_frac=0.1):
    warm_steps = int(total_steps * warm_frac)
    stable_steps = int(total_steps * stable_frac)
    decay_steps = max(1, total_steps - warm_steps - stable_steps)

    if step < warm_steps:
        return base_lr * (step + 1) / max(1, warm_steps)

    if step < warm_steps + stable_steps:
        return base_lr

    t = (step - warm_steps - stable_steps) / decay_steps  # 0..1
    min_lr = base_lr * final_lr_frac
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))

def train(model, opt, train_loader, eval_loader, train_iterations, base_lr, total_steps,
          global_step, alpha=10, viz_dir="/home/kxt439/DeepLoop2.0/denseunet_train_loopenhance/ds_1.0/viz"):
    # opt = torch.optim.AdamW(model.parameters(), lr=lr)
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

    train_iter = iterate_dataset(train_loader)
    eval_iter = iterate_dataset(eval_loader)

    viz_x, viz_y = next(eval_iter)

    for iter_i in (pbar := trange(train_iterations)):
        x, y = next(train_iter)

        cur_lr = wsd_lr(global_step, total_steps, base_lr, warm_frac=0.05, stable_frac=0.45, final_lr_frac=0.1)
        for pg in opt.param_groups:
            pg["lr"] = cur_lr

        opt.zero_grad(set_to_none=True)

        rep_loss = 0.0
        rec_loss = 0.0
        # preds, indices, cmt_loss = model(x.view(-1, 1, matrix_size, matrix_size))
        preds, indices, _ = model(x.view(-1, 1, matrix_size, matrix_size))
        # cmt_loss_log = cmt_loss.detach() 
        preds = preds.view(x.shape)
        rep_target = y.unsqueeze(1)
        for rep_i in range(x.shape[1]):
            out = preds[:, rep_i].unsqueeze(1)
            # rec_loss += (out - rep_target).abs().mean()
            rec_loss += torch.nn.functional.mse_loss(out, rep_target)
            for rep_j in range(x.shape[1]):  # output should also be similar across replicates
                if rep_i != rep_j:
                    rep2_target = preds[:, rep_j].unsqueeze(1).detach()
                    rep_loss += torch.nn.functional.mse_loss(out, rep2_target)

                    # rep2_target = preds[:, rep_j].unsqueeze(1)
                    # rep_loss += torch.nn.functional.mse_loss(out, rep2_target)
        rec_loss /= x.shape[1]
        rep_loss /= x.shape[1] * (x.shape[1] - 1)
        # (rec_loss + 0.1 * rep_loss + alpha * cmt_loss).backward()
        rec_loss.backward()
        opt.step()
        wandb.log({"lr": cur_lr, "rec_loss": rec_loss.item(), "rep_loss": rep_loss.item()})
        global_step += 1
        pbar.set_description(
            f"rec loss: {rec_loss.item():.3f} | "
            #+ f"cmt loss: {cmt_loss.item():.3f} | "
            #+ f"active %: {indices.unique().numel() / num_codes * 100:.3f}"
        )
        if iter_i % 1000 == 0:
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

            fig, axs = plt.subplots(8, 5, figsize=(8, 16))
            for batch_i in range(8):
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
            plt.savefig(f'{viz_dir}/heatmaps/dense_unet_{iter_i // 100}.jpg')
            plt.close()

            # visualize the first convolutional layer filters
            filters = next(model.parameters()).detach().cpu().numpy()
            n_filters = filters.shape[0]
            n_rows = int(math.sqrt(n_filters))
            n_cols = math.ceil(n_filters / n_rows)
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
            for i in range(n_filters):
                axs[i // n_cols, i % n_cols].imshow(filters[i, 0], cmap="RdGy_r")
                axs[i // n_cols, i % n_cols].set_xticks([])
                axs[i // n_cols, i % n_cols].set_yticks([])
            plt.tight_layout()
            plt.savefig(f'{viz_dir}/filters/dense_unet_{iter_i // 100}.jpg')
            plt.close()

            wandb.log({"reconstruction": wandb.Image(f'{viz_dir}/heatmaps/dense_unet_{iter_i // 100}.jpg')})
            torch.save(model.state_dict(), f'{viz_dir}/dense_unet_train_wsd.pt')
        # scheduler.step()
    return global_step

if __name__ == '__main__':
    torch.manual_seed(seed)
    wandb.init(project="dense_unet")

    rep_names = [
        'H9', 'H7', 'HUES1', 'HUES2', 'HUES4', 'HUES8', 'ESI051', 'ESI035', 'ESI017',
        '51213', '51339', 'ACS_1011', 'ACS_1019', 'ACS_1026', 'AG28869', 'CJ-IPS',
        'H1', '51286', 'AG28041', 'DIS1016', 'GM23716', 'GM23720',
        'iPSC_075C', 'iPSC_088B', 'iPSC_1029C', 'iPSC_1106'
    ]

    train_chroms = [f'chr{i}' for i in range(1, 23, 2)]  # odd
    eval_chroms  = [f'chr{i}' for i in range(2, 23, 2)]  # even

    eval_datasets = []
    for chrom in eval_chroms:
        cool_files = [f'/scratch/pioneer/users/kxt439/DeepLoop2.0/rep_coolers/chroms/depth_1.0/{chrom}_downsample_{rep}_1.0.npz'
                      for rep in rep_names]
        target_file = f'/home/kxt439/DeepLoop2.0/rep_coolers/target_chroms/combined/{chrom}_reproducible_combined.npz'

        if all(os.path.exists(f) for f in cool_files + [target_file]):
            eval_data = MatrixDataset(
                cool_files, target_file,
                matrix_size=matrix_size, step_size=step_size,
                log_transform=True,
                jitter=False, symmetric=True
            )
            eval_datasets.append(eval_data)

    if len(eval_datasets) == 0:
        raise RuntimeError("No eval datasets found. Check eval_chroms paths.")

    eval_loader = DataLoader(
        torch.utils.data.ConcatDataset(eval_datasets),
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True
    )

    model = DenseUNet(out_channels=1).to(device)
    print('Device:', device)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    available_train_chroms = []
    for chrom in train_chroms:
        cool_files = [f'/scratch/pioneer/users/kxt439/DeepLoop2.0/rep_coolers/chroms/depth_1.0/{chrom}_downsample_{rep}_1.0.npz'
                      for rep in rep_names]
        target_file = f'/home/kxt439/DeepLoop2.0/rep_coolers/target_chroms/combined/{chrom}_reproducible_combined.npz'
        if all(os.path.exists(f) for f in cool_files + [target_file]):
            available_train_chroms.append(chrom)
        else:
            print(f"[Skipping] {chrom} due to missing files")

    if len(available_train_chroms) == 0:
        raise RuntimeError("No train chromosomes found. Check train paths.")

    chrom_train_iters = train_iter // len(available_train_chroms)
    total_steps = chrom_train_iters * len(available_train_chroms)

    print(f"Train chroms available: {len(available_train_chroms)}")
    print(f"chrom_train_iters = {chrom_train_iters}")
    print(f"total_steps (global WSD) = {total_steps}")

    print("Starting chromosome-wise training with GLOBAL WSD...")
    global_step = 0

    for chrom in available_train_chroms:
        print(f"[Loading] {chrom} for training...")

        cool_files = [f'/scratch/pioneer/users/kxt439/DeepLoop2.0/rep_coolers/chroms/depth_1.0/{chrom}_downsample_{rep}_1.0.npz'
                      for rep in rep_names]
        target_file = f'/home/kxt439/DeepLoop2.0/rep_coolers/target_chroms/combined/{chrom}_reproducible_combined.npz'

        dataset = MatrixDataset(
            cool_files, target_file,
            matrix_size=matrix_size, step_size=step_size,
            log_transform=True
        )

        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=16,
            pin_memory=True
        )

        global_step = train(
            model=model,
            opt=opt,
            train_loader=train_loader,
            eval_loader=eval_loader,
            train_iterations=chrom_train_iters,
            base_lr=lr,
            total_steps=total_steps,
            global_step=global_step,
            alpha=10,
            viz_dir=f"/home/kxt439/DeepLoop2.0/denseunet_train_loopenhance/ds_1.0/viz"
        )

    print(f"Finished training on {len(available_train_chroms)} chromosomes.")
