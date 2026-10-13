import os, sys, numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import h5py

sys.path.insert(0, os.path.dirname(__file__) or ".")
from models.LFTransMamba import LFTransMamba
from skimage.metrics import peak_signal_noise_ratio as PSNR, structural_similarity as SSIM

angRes, scale = 5, 4
device = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "results/figures_rgb"
STANFORD_RGB_DIR = "stanford_rgb_refs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONFIGS = [
    ("ablation_baseline_v4", "Baseline", False, False, False),
    ("ablation_gaam_v4", "+GAAM", True, False, False),
    ("ablation_gaam_cvesa_v4", "+GAAM+CVESA", True, True, False),
]

SELECT = [2, 3, 4, 5, 6]

cdict = {
    "red": [(0.0, 0.4, 0.4), (0.4, 0.6, 0.6), (0.7, 0.9, 0.9), (1.0, 1.0, 1.0)],
    "green": [(0.0, 0.4, 0.4), (0.4, 0.5, 0.5), (0.7, 0.3, 0.3), (1.0, 0.0, 0.0)],
    "blue": [(0.0, 0.4, 0.4), (0.4, 0.5, 0.5), (0.7, 0.2, 0.2), (1.0, 0.0, 0.0)],
}
gray_red_cmap = LinearSegmentedColormap("gray_red", cdict, 256)

def rgb2y(img):
    img = img.astype(np.float32) / 255.0
    y = 65.481*img[...,0] + 128.553*img[...,1] + 24.966*img[...,2] + 16.0
    return np.clip(y/255.0, 0.0, 1.0)

def rgb_to_ycbcr(img):
    img = img.astype(np.float32)
    y  = (65.481*img[...,0] + 128.553*img[...,1] + 24.966*img[...,2] + 16.0) / 255.0
    cb = -37.797*img[...,0] - 74.203*img[...,1] + 112.000*img[...,2] + 128.0
    cr = 112.000*img[...,0] - 93.786*img[...,1] - 18.214*img[...,2] + 128.0
    return y, cb, cr

def ycbcr_to_rgb(y, cb, cr):
    yy = (y * 255.0).astype(np.float32)
    r = yy + 1.402 * (cr - 128.0)
    g = yy - 0.344136 * (cb - 128.0) - 0.714136 * (cr - 128.0)
    b = yy + 1.772 * (cb - 128.0)
    return np.clip(np.stack([r, g, b], axis=-1), 0, 255).astype(np.uint8)

def load_h5_for_sai(dset, scene_name):
    h5_path = os.path.join("/root/autodl-tmp/ZJL/h5_data/test", dset, scene_name + ".h5")
    with h5py.File(h5_path, "r") as f:
        hr_y = np.array(f["Hr_SAI_y"], dtype=np.float32)
        lr_y = np.array(f["Lr_SAI_y"], dtype=np.float32)
    return hr_y, lr_y

SCENES = [
    ("HCI/antinous", "antinous", "png"),
    ("HCI/boardgames", "boardgames", "png"),
    ("HCI/rosemary", "rosemary", "png"),
    ("HCI/table", "table", "png"),
    ("HCI/bedroom", "bedroom", "png"),
    ("HCI/herbs", "herbs", "png"),
    ("Stanford/buildings_1", "buildings_1", "stanford"),
    ("Stanford/bikes_9", "bikes_9", "stanford"),
    ("Stanford/cars_1", "cars_1", "stanford"),
    ("Stanford/general_29", "general_29", "stanford"),
    ("Stanford/flowers_plants_1", "flowers_plants_1", "stanford"),
]

models = {}
for ckpt_name, label, g, c, m in CONFIGS:
    ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name + ".pth")
    if not os.path.exists(ckpt_path):
        print(f"[SKIP] {ckpt_path}"); continue
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

for scene_display, scene_dir, mode in SCENES:
    print(f"\n=== {scene_display} ===")
    
    if mode == "png":
        scene_path = os.path.join("/root/autodl-tmp/ZJL/train_ready_all/test", scene_dir)
        if not os.path.isdir(scene_path):
            print(f"[SKIP] no dir"); continue
        imgs = sorted(f for f in os.listdir(scene_path) if f.startswith("input_Cam") and f.endswith(".png"))
        if len(imgs) != 81:
            sai_p = os.path.join(scene_path, "SAI")
            if os.path.isdir(sai_p):
                imgs = sorted(f for f in os.listdir(sai_p) if f.startswith("view_u") and f.endswith(".png"))
                vrgb = [np.array(Image.open(os.path.join(sai_p, fn)).convert("RGB")) for fn in imgs]
                H, W = vrgb[0].shape[0], vrgb[0].shape[1]
                vs_rgb = np.stack(vrgb, 0).reshape(5, 5, H, W, 3)
                vs_y = np.stack([rgb2y(v) for v in vrgb], 0).reshape(5, 5, H, W)
            else:
                print(f"[SKIP] no valid images"); continue
        else:
            vrgb = [np.array(Image.open(os.path.join(scene_path, f)).convert("RGB")) for f in imgs]
            H, W = vrgb[0].shape[0], vrgb[0].shape[1]
            vs_rgb = np.stack(vrgb, 0).reshape(9, 9, H, W, 3)[np.ix_(SELECT, SELECT)]
            vs_y = np.stack([rgb2y(v) for v in vrgb], 0).reshape(9, 9, H, W)[np.ix_(SELECT, SELECT)]
        gt_y = vs_y[2, 2, :, :]
        gt_rgb = vs_rgb[2, 2, :, :, :]
        cs = min(H, W, 512)
        cy, cx = H // 2 - cs // 2, W // 2 - cs // 2
        gt_y = gt_y[cy:cy+cs, cx:cx+cs]
        gt_rgb = gt_rgb[cy:cy+cs, cx:cx+cs, :]
        _, gt_cb, gt_cr = rgb_to_ycbcr(gt_rgb)
        hr_sai = vs_y.transpose(0, 2, 1, 3).reshape(5*H, 5*W)
        hr_sai_t = torch.from_numpy(hr_sai).float().unsqueeze(0).unsqueeze(0).to(device)
        lr_t = F.interpolate(hr_sai_t, scale_factor=0.25, mode="bicubic", align_corners=False)
        lr_sai = lr_t.squeeze().cpu().numpy()
        lr_center = lr_sai[2*(H//4):3*(H//4), 2*(W//4):3*(W//4)]
        lr_up = np.array(Image.fromarray((lr_center[cy//4:(cy+cs)//4, cx//4:(cx+cs)//4]*255).astype(np.uint8)).resize((cs, cs), Image.BICUBIC)) / 255.0
        lr_rgb = ycbcr_to_rgb(lr_up, gt_cb, gt_cr)
    else:
        hr_y, lr_y = load_h5_for_sai("Stanford", scene_dir)
        H = hr_y.shape[0] // 5
        W = hr_y.shape[1] // 5
        
        # 限制显示区域避免 OOM
        cs = min(H, W, 320)
        cy, cx = H // 2 - cs // 2, W // 2 - cs // 2
        
        # 裁剪 LR 5-grid
        lr_H = lr_y.shape[0] // 5
        lr_W = lr_y.shape[1] // 5
        Lcs = cs // 4
        l_cy, l_cx = lr_H // 2 - Lcs // 2, lr_W // 2 - Lcs // 2
        lr_cropped_full = lr_y.reshape(5, lr_H, 5, lr_W).transpose(0, 2, 1, 3).reshape(5*lr_H, 5*lr_W)
        lr_cropped = lr_cropped_full[l_cy*5:(l_cy+Lcs)*5, l_cx*5:(l_cx+Lcs)*5]
        lr_t = torch.from_numpy(lr_cropped).float().unsqueeze(0).unsqueeze(0).to(device)
        
        # GT
        gt_y = hr_y.reshape(5, H, 5, W).transpose(0, 2, 1, 3).reshape(5*H, 5*W)
        gt_y = gt_y[2*H:3*H, 2*W:3*W][cy:cy+cs, cx:cx+cs]
        
        # RGB 参考
        rgb_ref = np.array(Image.open(os.path.join(STANFORD_RGB_DIR, scene_dir + ".png")).convert("RGB"))
        rgb_ref = np.array(Image.fromarray(rgb_ref).resize((gt_y.shape[1], gt_y.shape[0]), Image.LANCZOS))
        gt_rgb = rgb_ref
        _, gt_cb, gt_cr = rgb_to_ycbcr(rgb_ref)
        
        # LR 显示
        lr_disp = lr_cropped_full[l_cy*5:(l_cy+Lcs)*5, l_cx*5:(l_cx+Lcs)*5]
        lr_disp = lr_disp.reshape(5, Lcs, 5, Lcs).transpose(0, 2, 1, 3).reshape(5*Lcs, 5*Lcs)
        lr_center = lr_disp[2*Lcs:3*Lcs, 2*Lcs:3*Lcs]
        lr_up = np.array(Image.fromarray((lr_center*255).astype(np.uint8)).resize((cs, cs), Image.BICUBIC)) / 255.0
        lr_rgb = ycbcr_to_rgb(lr_up, gt_cb, gt_cr)
    
    srs_y = {}
    srs_rgb = {}
    for label, model in models.items():
        with torch.no_grad():
            sr_sai = model(lr_t)
        sr_np = sr_sai.squeeze().cpu().numpy()
        sr_center = sr_np[2*cs:3*cs, 2*cs:3*cs]
        srs_y[label] = sr_center
        srs_rgb[label] = ycbcr_to_rgb(sr_center, gt_cb, gt_cr)
    
    labels = ["LR (x4)"] + [c[1] for c in active] + ["GT"]
    imgs_y = [lr_up] + [srs_y[c[1]] for c in active] + [gt_y]
    imgs_rgb = [lr_rgb] + [srs_rgb[c[1]] for c in active] + [gt_rgb]
    
    z = 100
    zy, zx = cs//2 - z//2, cs//2 - z//2
    
    fig, axes = plt.subplots(2, n_cols, figsize=(n_cols*4.0, 8.0))
    for i in range(n_cols):
        axes[0, i].imshow(imgs_rgb[i])
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("LR\n(x4 Bicubic)", fontsize=11, fontweight="bold")
        elif i == n_cols-1:
            axes[0, i].set_title("GT\n(HR)", fontsize=11, fontweight="bold")
        else:
            ps = PSNR(gt_y, imgs_y[i], data_range=1.0)
            ss = SSIM(gt_y, imgs_y[i], data_range=1.0)
            axes[0, i].set_title("{}\n{:.2f}dB/{:.4f}".format(labels[i], ps, ss), fontsize=10, fontweight="bold")
    for i in range(n_cols):
        ax = axes[1, i]
        if i == 0 or i == n_cols-1:
            ax.imshow(imgs_rgb[i])
        else:
            err = np.abs(imgs_y[i] - gt_y)
            c = ax.imshow(err, cmap=gray_red_cmap, vmin=0, vmax=0.15)
            plt.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
        ax.axis("off")
    plt.tight_layout(pad=0.4)
    sp = os.path.join(OUTPUT_DIR, scene_display.replace("/", "_") + "_comp.png")
    plt.savefig(sp, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved: {sp}")
    
    fig2, axes2 = plt.subplots(2, n_cols, figsize=(n_cols*3.5, 7.0))
    for i in range(n_cols):
        axes2[0, i].imshow(imgs_rgb[i][zy:zy+z, zx:zx+z, :])
        axes2[0, i].axis("off")
        if i == n_cols-1:
            axes2[0, i].set_title("GT", fontsize=10, fontweight="bold")
        elif i == 0:
            axes2[0, i].set_title("LR", fontsize=10, fontweight="bold")
        else:
            ps = PSNR(gt_y[zy:zy+z, zx:zx+z], imgs_y[i][zy:zy+z, zx:zx+z], data_range=1.0)
            axes2[0, i].set_title("{}\n{:.2f}dB".format(labels[i], ps), fontsize=9, fontweight="bold")
    for i in range(n_cols):
        ax2 = axes2[1, i]
        if i == 0 or i == n_cols-1:
            ax2.imshow(imgs_rgb[i][zy:zy+z, zx:zx+z, :])
        else:
            zerr = np.abs(imgs_y[i][zy:zy+z, zx:zx+z] - gt_y[zy:zy+z, zx:zx+z])
            c = ax2.imshow(zerr, cmap=gray_red_cmap, vmin=0, vmax=0.15)
            plt.colorbar(c, ax=ax2, fraction=0.046, pad=0.04)
        ax2.axis("off")
    plt.tight_layout(pad=0.3)
    sp2 = os.path.join(OUTPUT_DIR, scene_display.replace("/", "_") + "_zoom.png")
    plt.savefig(sp2, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved: {sp2}")

print(f"\nDone! Figures in {OUTPUT_DIR}")
