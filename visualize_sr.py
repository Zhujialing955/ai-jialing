import os, sys, numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
sys.path.insert(0, os.path.dirname(__file__) or ".")
from models.LFTransMamba import LFTransMamba
from skimage.metrics import peak_signal_noise_ratio as PSNR

angRes, scale = 5, 4
device = "cuda" if torch.cuda.is_available() else "cpu"
SELECT = [2, 3, 4, 5, 6]
TEST_ROOT = "/root/autodl-tmp/ZJL/train_ready_all/test"
CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "results/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONFIGS = [
    ("ablation_baseline_v2",    "Baseline",      False, False, False),
    ("ablation_gaam_v2",        "+GAAM",          True,  False, False),
    ("ablation_gftransmamba_v2","Full (Ours)",    True,  True, True),
]

def rgb2y(img):
    img = img.astype(np.float32) / 255.0
    y = 65.481*img[...,0] + 128.553*img[...,1] + 24.966*img[...,2] + 16.0
    return np.clip(y/255.0, 0.0, 1.0)

def ycbcr2rgb(y, cb, cr):
    y = y.astype(np.float32)
    cb = cb.astype(np.float32) - 128.0
    cr = cr.astype(np.float32) - 128.0
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return np.clip(np.stack([r,g,b], axis=-1), 0, 255).astype(np.uint8)

def rgb2ycbcr(img):
    img = img.astype(np.float32)
    y  = 65.481*img[...,0] + 128.553*img[...,1] +  24.966*img[...,2] + 16.0
    cb = -37.797*img[...,0] - 74.203*img[...,1] + 112.000*img[...,2] + 128.0
    cr = 112.000*img[...,0] - 93.786*img[...,1] -  18.214*img[...,2] + 128.0
    return y/255.0, cb, cr

def load_test(sp):
    imgs = sorted(f for f in os.listdir(sp) if f.startswith("input_Cam") and f.endswith(".png"))
    vrgb = [np.array(Image.open(os.path.join(sp,f)).convert("RGB")) for f in imgs]
    vy = [rgb2y(v) for v in vrgb]
    H, W = vy[0].shape[0], vy[0].shape[1]
    vs = np.stack(vy, 0).reshape(9,9,H,W)[np.ix_(SELECT, SELECT)]
    vs_rgb = np.stack(vrgb, 0).reshape(9,9,H,W,3)[np.ix_(SELECT, SELECT)]
    return H, W, vs, vs_rgb

def sai_from_views(views):
    u, v = 5, 5
    H, W = views.shape[2], views.shape[3]
    return torch.from_numpy(views.transpose(0,2,1,3).reshape(u*H,v*W)).float().unsqueeze(0).unsqueeze(0)

def extract_center(sai, H, W):
    a = sai.squeeze().cpu().numpy()
    return a[2*H:3*H, 2*W:3*W]

def extract_center_rgb(vs_rgb, H, W):
    return vs_rgb[2, 2, :, :, :]

def safe_crop(img, y, x, size):
    if img.ndim < 2 or img.shape[0] == 0 or img.shape[1] == 0:
        return img
    H, W = img.shape[0], img.shape[1]
    y = max(0, min(y, H-size))
    x = max(0, min(x, W-size))
    if img.ndim == 2:
        return img[y:y+size, x:x+size]
    else:
        return img[y:y+size, x:x+size, :]

# Custom colormap: gray background (low error) -> red highlight (high error)
# Matches the reference paper style
cdict = {
    "red":   [(0.0, 0.4, 0.4), (0.4, 0.6, 0.6), (0.7, 0.9, 0.9), (1.0, 1.0, 1.0)],
    "green": [(0.0, 0.4, 0.4), (0.4, 0.5, 0.5), (0.7, 0.3, 0.3), (1.0, 0.0, 0.0)],
    "blue":  [(0.0, 0.4, 0.4), (0.4, 0.5, 0.5), (0.7, 0.2, 0.2), (1.0, 0.0, 0.0)],
}
gray_red_cmap = LinearSegmentedColormap("gray_red", cdict, 256)

SCENES = ["antinous", "boardgames", "rosemary"]

models = {}
for ckpt_name, label, g, c, m in CONFIGS:
    print("Loading model:", label)
    model = LFTransMamba(angRes_in=angRes, scale_factor=scale, D=16, R=1, K=4, C=64, T=3,
                         use_gaam=g, use_cvesa=c, use_mfpm=m).to(device)
    sd = torch.load("checkpoints/{}.pth".format(ckpt_name), map_location="cpu")
    if "state_dict" in sd: sd = sd["state_dict"]
    sd = {k.replace("module.",""): v for k,v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    models[label] = model

for scene_name in SCENES:
    print("\nProcessing:", scene_name)
    sp = os.path.join(TEST_ROOT, scene_name)
    H, W, views_y, views_rgb = load_test(sp)

    gt_rgb = extract_center_rgb(views_rgb, H, W)
    gt_y, gt_cb, gt_cr = rgb2ycbcr(gt_rgb)

    hr_sai = sai_from_views(views_y).to(device)
    lr = F.interpolate(hr_sai, scale_factor=0.25, mode="bicubic", align_corners=False)
    gt_center_y = extract_center(hr_sai, H, W)
    H_lr, W_lr = H // 4, W // 4
    lr_center_y = extract_center(lr, H_lr, W_lr)
    lr_display = np.array(Image.fromarray((lr_center_y*255).astype(np.uint8)).resize((W, H), Image.BICUBIC)) / 255.0

    def y_to_rgb(y_channel):
        return ycbcr2rgb(y_channel*255.0, gt_cb, gt_cr)

    lr_rgb = y_to_rgb(lr_display)

    srs_y = {}
    srs_rgb = {}
    for label, model in models.items():
        with torch.no_grad():
            sr = model(lr)
        sr_y = extract_center(sr, H, W)
        srs_y[label] = sr_y
        srs_rgb[label] = y_to_rgb(sr_y)

    n = len(CONFIGS) + 2
    fig, axes = plt.subplots(2, n, figsize=(n * 3.8, 7.5))

    all_titles = ["LR\n(Bicubic)"] + [c[1] for c in CONFIGS] + ["GT\n(HR)"]
    all_images_rgb = [lr_rgb] + [srs_rgb[c[1]] for c in CONFIGS] + [gt_rgb]
    all_images_y   = [lr_display] + [srs_y[c[1]] for c in CONFIGS] + [gt_y]

    # Row 0: RGB display
    for i, (ax, title, img_rgb, img_y) in enumerate(zip(axes[0], all_titles, all_images_rgb, all_images_y)):
        ax.imshow(img_rgb)
        if i == 0: ax.set_title("LR\n(Bicubic)", fontsize=11, fontweight="bold")
        elif i == n - 1: ax.set_title("GT\n(HR)", fontsize=11, fontweight="bold")
        else: ax.set_title("{}\n{:.2f} dB".format(title, PSNR(gt_y, img_y, data_range=1.0)), fontsize=11, fontweight="bold")
        ax.axis("off")

    # Row 1: Error maps - gray background + red highlights
    for i in range(n):
        ax = axes[1, i]
        if i == 0:
            ax.imshow(lr_rgb)
            ax.set_title("", fontsize=10)
        elif i == n - 1:
            ax.imshow(gt_rgb)
            ax.set_title("RGB GT", fontsize=10, fontweight="bold")
        else:
            err = np.abs(all_images_y[i] - gt_y)
            c = ax.imshow(err, cmap=gray_red_cmap, vmin=0, vmax=0.15)
            psnr_val = PSNR(gt_y, all_images_y[i], data_range=1.0)
            ax.set_title("Error\n{:.2f} dB".format(psnr_val), fontsize=10, fontweight="bold")
            plt.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
        ax.axis("off")

    plt.tight_layout(pad=0.4)
    p = os.path.join(OUTPUT_DIR, "{}_comparison.png".format(scene_name))
    plt.savefig(p, dpi=250, bbox_inches="tight"); plt.close()
    print("  Saved:", p)

    # Zoom
    zoom = 100
    ys, xs = H//2 - zoom//2, W//2 - zoom//2
    fig2, axes2 = plt.subplots(2, n, figsize=(n * 3.2, 6.5))

    for i, (ax, title, img_rgb, img_y) in enumerate(zip(axes2[0], all_titles, all_images_rgb, all_images_y)):
        ax.imshow(safe_crop(img_rgb, ys, xs, zoom))
        if i == 0:
            ax.set_title("LR\n(Bicubic)", fontsize=10, fontweight="bold")
            ax.add_patch(Rectangle((xs, ys), zoom, zoom, linewidth=2, edgecolor="red", facecolor="none", linestyle="--"))
        elif i == n - 1: ax.set_title("GT", fontsize=10, fontweight="bold")
        else: ax.set_title("{}\n{:.2f}dB".format(title, PSNR(gt_y, img_y, data_range=1.0)), fontsize=10, fontweight="bold")
        ax.axis("off")

    for i in range(n):
        ax = axes2[1, i]
        if i == 0: ax.imshow(safe_crop(lr_rgb, ys, xs, zoom))
        elif i == n - 1: ax.imshow(safe_crop(gt_rgb, ys, xs, zoom))
        else:
            c = ax.imshow(safe_crop(np.abs(all_images_y[i] - gt_y), ys, xs, zoom), cmap=gray_red_cmap, vmin=0, vmax=0.15)
            plt.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
        ax.axis("off")

    plt.tight_layout(pad=0.3)
    p2 = os.path.join(OUTPUT_DIR, "{}_zoom.png".format(scene_name))
    plt.savefig(p2, dpi=250, bbox_inches="tight"); plt.close()
    print("  Saved:", p2)

print("\nAll figures saved to", OUTPUT_DIR)
