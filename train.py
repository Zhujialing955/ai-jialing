import os
import argparse
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.lf_dataset import LFSRDataset
from models.LFTransMamba import LFTransMamba
from losses import MultiScaleFreqAngularLoss


def calc_psnr(sr, hr):
    mse = nn.functional.mse_loss(sr, hr)
    if mse == 0:
        return 100
    return 10 * torch.log10(1.0 / mse)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    psnr_list = []
    for lr, hr in loader:
        lr = lr.to(device)
        hr = hr.to(device)
        # 使用标准 bicubic 从 HR 生成 LR，与训练一致
        lr = torch.nn.functional.interpolate(hr, scale_factor=0.25, mode='bicubic', align_corners=False)
        uh, vw = lr.shape[2], lr.shape[3]
        size = min(uh, vw)
        size = (size // 5) * 5
        lr = lr[:, :, :size, :size]
        hr = hr[:, :, :size * 4, :size * 4]
        sr = model(lr)
        sr = torch.clamp(sr, 0, 1)
        psnr = calc_psnr(sr, hr)
        psnr_list.append(psnr.item())
    model.train()
    return sum(psnr_list) / len(psnr_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='ablation_baseline', help='Experiment name')
    parser.add_argument('--use_gaam', action='store_true', help='Use GAAM module')
    parser.add_argument('--use_cvesa', action='store_true', help='Use CVESA module')
    parser.add_argument('--use_mfpm', action='store_true', help='Use MFPM module')
    parser.add_argument('--use_frem', action='store_true', help='Use FREM module')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--C', type=int, default=64, help='Channel width')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print(f"Experiment: {args.name}")
    print(f"  GAAM={args.use_gaam}  CVESA={args.use_cvesa}  MFPM={args.use_mfpm}  FREM={args.use_frem}")

    USE_FREQ_ANGULAR = True

    train_set = LFSRDataset(
        root="/root/autodl-tmp/ZJL/h5_data/train",
        list_file="train_list.txt",
        angRes=5,
        patch_size=32,
        scale_factor=4,
        mode="train"
    )
    val_set = LFSRDataset(
        root="/root/autodl-tmp/ZJL/h5_data/train",
        list_file="test_list.txt",
        angRes=5,
        patch_size=32,
        scale_factor=4,
        mode="val"
    )

    print("Train scenes:", len(train_set))
    print("Val scenes:", len(val_set))

    train_loader = DataLoader(
        train_set, batch_size=1, shuffle=True,
        num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False,
        num_workers=2, pin_memory=True
    )

    model = LFTransMamba(angRes_in=5, scale_factor=4, D=16, R=1, K=4, C=args.C, T=3,
                         use_gaam=args.use_gaam, use_cvesa=args.use_cvesa, use_mfpm=(args.use_mfpm or args.use_frem)).to(device)

    if args.resume:
        state_dict = torch.load(args.resume, map_location=device)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        print(f"  Resumed from {args.resume}")


    criterion = nn.L1Loss()
    freq_angular_criterion = MultiScaleFreqAngularLoss(
        angRes=5, num_scales=3, freq_weight=0.01, angular_weight=0.005
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=2e-4, betas=(0.9, 0.999)
    )

    # Cosine annealing learning rate scheduler
    epochs = args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    scaler = torch.cuda.amp.GradScaler()

    os.makedirs("checkpoints", exist_ok=True)

    best_psnr = 0

    for epoch in range(epochs):
        epoch_loss = 0

        for i, (lr, hr) in enumerate(train_loader):
            lr = lr.to(device)
            hr = hr.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                sr = model(lr)
                if sr.shape != hr.shape:
                    sr = torch.nn.functional.interpolate(
                        sr, size=hr.shape[-2:],
                        mode="bilinear", align_corners=False
                    )
                loss_recon = criterion(sr, hr)
                if USE_FREQ_ANGULAR:
                    loss_freq = freq_angular_criterion(sr, hr)
                    loss = loss_recon + loss_freq
                else:
                    loss = loss_recon
                    loss_freq = torch.tensor(0.0)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if i % 10 == 0:
                print(
                    "[Epoch %d] [Iter %d] Loss: %.6f "
                    "(recon=%.6f, freq_angular=%.6f)"
                    % (epoch + 1, i, loss.item(),
                       loss_recon.item(), loss_freq.item())
                )

        # Step the learning rate scheduler
        scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        val_psnr = evaluate(model, val_loader, device)
        current_lr = optimizer.param_groups[0]['lr']

        print("\nEpoch %d" % (epoch + 1))
        print("Avg Loss: %.6f" % avg_loss)
        print("Val PSNR: %.3f dB" % val_psnr)
        print("LR: %.6f\n" % current_lr)

        torch.save(model.state_dict(), f"checkpoints/{args.name}_last.pth")

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), f"checkpoints/{args.name}.pth")
            print("Best model updated!")

    print("Training Finished!")


if __name__ == "__main__":
    main()
