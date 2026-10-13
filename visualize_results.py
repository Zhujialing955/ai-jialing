import os, sys, numpy as np
import h5py
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__) or ".")
from models.LFTransMamba import LFTransMamba
from skimage.metrics import peak_signal_noise_ratio as PSNR

angRes, scale = 5, 4
device = "cuda" if torch.cuda.is_available() else "cpu"
TEST_ROOT = "/root/autodl-tmp/ZJL/h5_data/test"
CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "results/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONFIGS = [
    ("ablation_baseline_v4", "Baseline", False, False, False),
    ("ablation_gaam_v4", "+GAAM", True, False, False),
    ("ablation_gaam_cvesa_v4", "+GAAM+CVESA", True, True, False),
]

cdict = {
    "red": [(0.0, 0.4, 0.4), (0.4, 0.6, 0.6), (0.7, 0.9, 0.9), (1.0, 1.0, 1.0)],
    "green": [(0.0, 0.4, 0.4), (0.4, 0.5, 0.5), (0.7, 0.3, 0.3), (1.0, 0.0, 0.0)],
    "blue": [(0.0, 0.4, 0.4), (0.4, 0.5, 0.5), (0.7, 0.2, 0.2), (1.0, 0.0, 0.0)],
}
gray_red_cmap = LinearSegmentedColormap("gray_red", cdict, 256)

def load_h5_scene(scene_key):
    parts = scene_key.split("/")
    h5_path = os.path.join(TEST_ROOT, parts[0], parts[1] + ".h5")
    with h5py.File(h5_path, "r") as f:
        hr = np.array(f["Hr_SAI_y"], dtype=np.float32)
        lr = np.array(f["Lr_SAI_y"], dtype=np.float32)
    return hr, lr

def center_view(sai, H, W):
    return sai[2*H:3*H, 2*W:3*W]

SCENES = ["HCI/bedroom", "HCI/herbs", "Stanford/buildings_1", "Stanford/bikes_9"]

models = {}
for ckpt_name, label, g, c, m in CONFIGS:
    ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name + ".pth")
    if not os.path.exists(ckpt_path):
        print(f"[SKIP] {ckpt_path}")
        continue
    print(f"Loading: {label}")
    model = LFTransMamba(angRes_in=angRes, scale_factor=scale, D=16, R=1, K=4, C=64, T=3,
                         use_gaam=g, use_cvesa=c, use_mfpm=m).to(device)
    sd = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in sd: sd = sd["state_dict"]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    models[label] = model

active = [c for c in CONFIGS if c[1] in models]
n_cols = len(active) + 2

for scene_key in SCENES:
    print(f"\n=== {scene_key} ===")
    hr_y, lr_y = load_h5_scene(scene_key)
    H = hr_y.shape[0] // 5
    W = hr_y.shape[1] // 5
    # 限制显示区域大小
    cs = min(H, W, 360)
    cy, cx = H // 2 - cs // 2, W // 2 - cs // 2
    gt_c = center_view(hr_y, H, W)[cy:cy+cs, cx:cx+cs]
    # 裁剪 LR 5-grid 以避免 OOM
    lr_H = lr_y.shape[0] // 5
    lr_W = lr_y.shape[1] // 5
    Lcs = cs // 4
    l_cy, l_cx = lr_H // 2 - Lcs // 2, lr_W // 2 - Lcs // 2
    lr_cropped = lr_y[l_cy*5:(l_cy+Lcs)*5, l_cx*5:(l_cx+Lcs)*5]
    lr_t = torch.from_numpy(lr_cropped).float().unsqueeze(0).unsqueeze(0).to(device)
    lr_disp = center_view(lr_cropped, Lcs, Lcs)
    lr_up = np.array(Image.fromarray((lr_disp*255).astype(np.uint8)).resize((cs, cs), Image.BICUBIC)) / 255.0
    srs = {}
    for label, model in models.items():
        with torch.no_grad():
            sr = model(lr_t)
        sr_c = center_view(sr.squeeze().cpu().numpy(), cs, cs)
        srs[label] = sr_c
        print(f"  {label}: {PSNR(gt_c, sr_c, data_range=1.0):.2f} dB")

    labels = ["LR(x4)"] + [c[1] for c in active] + ["GT"]
    imgs = [lr_up] + [srs[c[1]] for c in active] + [gt_c]

    fig, axes = plt.subplots(2, n_cols, figsize=(n_cols*3.8, 7.5))
    for i in range(n_cols):
        axes[0, i].imshow(imgs[i], cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        if i == n_cols - 1:
            axes[0, i].set_title(labels[i], fontsize=11, fontweight="bold")
        elif i == 0:
            axes[0, i].set_title(labels[i], fontsize=10, fontweight="bold")
        else:
            axes[0, i].set_title("{} {:.2f}dB".format(labels[i], PSNR(gt_c, imgs[i], data_range=1.0)), fontsize=10, fontweight="bold")
        ax1 = axes[1, i]
        if i == 0 or i == n_cols - 1:
            ax1.imshow(imgs[i], cmap="gray", vmin=0, vmax=1)
        else:
            err = np.abs(imgs[i] - gt_c)
            c = ax1.imshow(err, cmap=gray_red_cmap, vmin=0, vmax=0.15)
            plt.colorbar(c, ax=ax1, fraction=0.046, pad=0.04)
        ax1.axis("off")
    plt.tight_layout(pad=0.4)
    spath = os.path.join(OUTPUT_DIR, scene_key.replace("/", "_") + "_comp.png")
    plt.savefig(spath, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {spath}")

    z = 96
    zy, zx = cs // 2 - z // 2, cs // 2 - z // 2
    fig2, axes2 = plt.subplots(2, n_cols, figsize=(n_cols*3.2, 6.5))
    for i in range(n_cols):
        axes2[0, i].imshow(imgs[i][zy:zy+z, zx:zx+z], cmap="gray", vmin=0, vmax=1)
        axes2[0, i].axis("off")
        if i == 0:
            axes2[0, i].set_title("LR", fontsize=9, fontweight="bold")
        elif i == n_cols - 1:
            axes2[0, i].set_title("GT", fontsize=9, fontweight="bold")
        else:
            axes2[0, i].set_title("{} {:.2f}dB".format(labels[i], PSNR(gt_c[zy:zy+z, zx:zx+z], imgs[i][zy:zy+z, zx:zx+z], data_range=1.0)), fontsize=8, fontweight="bold")
        ax2 = axes2[1, i]
        if i == 0 or i == n_cols - 1:
            ax2.imshow(imgs[i][zy:zy+z, zx:zx+z], cmap="gray", vmin=0, vmax=1)
        else:
            zerr = np.abs(imgs[i][zy:zy+z, zx:zx+z] - gt_c[zy:zy+z, zx:zx+z])
            c = ax2.imshow(zerr, cmap=gray_red_cmap, vmin=0, vmax=0.15)
            plt.colorbar(c, ax=ax2, fraction=0.046, pad=0.04)
        ax2.axis("off")
    plt.tight_layout(pad=0.3)
    spath2 = os.path.join(OUTPUT_DIR, scene_key.replace("/", "_") + "_zoom.png")
    plt.savefig(spath2, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {spath2}")

print(f"\nDone! Figures in {OUTPUT_DIR}")
