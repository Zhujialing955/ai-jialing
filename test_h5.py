import os, re, math, numpy as np, h5py, torch, torch.nn.functional as F, csv, argparse
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
from models.LFTransMamba import LFTransMamba

angRes, scale_factor = 5, 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model_from_name(model_name, angRes=5, scale=4):
    parts = model_name.split("_")
    params = {}
    for p in parts:
        if p.startswith("D"): params["D"] = int(p[1:])
        elif p.startswith("R"): params["R"] = int(p[1:])
        elif p.startswith("K"): params["K"] = int(p[1:])
        elif p.startswith("C"): params["C"] = int(p[1:])
        elif p.startswith("T"): params["T"] = int(p[1:])
    use_gaam = "gaam" in model_name.lower()
    use_cvesa = "cvesa" in model_name.lower()
    use_mfpm = "mfpm" in model_name.lower() or "full" in model_name.lower()
    D = params.get("D", 16); R = params.get("R", 1)
    K = params.get("K", 4); C = params.get("C", 64); T = params.get("T", 3)
    return LFTransMamba(angRes, scale, D, R, K, C, T,
        use_gaam=use_gaam, use_cvesa=use_cvesa, use_mfpm=use_mfpm)

def load_h5_scene(h5_path, u=5, v=5, max_lr_size=320):
    with h5py.File(h5_path, "r") as f:
        hr_y = f["Hr_SAI_y"][()]; lr_y = f["Lr_SAI_y"][()]
    hr = torch.from_numpy(hr_y).float().unsqueeze(0).unsqueeze(0)
    lr = torch.from_numpy(lr_y).float().unsqueeze(0).unsqueeze(0)
    h, w = lr.shape[2], lr.shape[3]
    h = (h//u)*u; w = (w//v)*v; lr = lr[:,:,:h,:w]
    if h > max_lr_size or w > max_lr_size:
        h = min(h,max_lr_size); w = min(w,max_lr_size)
        h = (h//u)*u; w = (w//v)*v; lr = lr[:,:,:h,:w]
    h_hr, w_hr = hr.shape[2], hr.shape[3]
    h_hr = (h_hr//u)*u; w_hr = (w_hr//v)*v
    if h_hr > max_lr_size*4 or w_hr > max_lr_size*4:
        h_hr = min(h_hr,max_lr_size*4); w_hr = min(w_hr,max_lr_size*4)
        h_hr = (h_hr//u)*u; w_hr = (w_hr//v)*v
    hr = hr[:,:,:h_hr,:w_hr]
    return lr, hr

def center_view(sai, u=5, v=5):
    a = sai.squeeze().cpu().numpy()
    H = a.shape[0]//u; W = a.shape[1]//v; ci = u//2; cj = v//2
    return a[ci*H:(ci+1)*H, cj*W:(cj+1)*W]

def load_checkpoint(model, path):
    if not os.path.exists(path): return False
    sd = torch.load(path, map_location="cpu")
    if "state_dict" in sd: sd = sd["state_dict"]
    sd = {k.replace("module.",""): v for k,v in sd.items()}
    f = {}
    for k,v in sd.items():
        if k in model.state_dict() and v.shape == model.state_dict()[k].shape:
            f[k] = v
    model.load_state_dict(f, strict=False)
    print(f"  Loaded: {path}")
    return True

TEST_ROOT = "/root/autodl-tmp/ZJL/h5_data/test"
def discover_scenes():
    s = []
    for ds in sorted(os.listdir(TEST_ROOT)):
        dp = os.path.join(TEST_ROOT, ds)
        if not os.path.isdir(dp): continue
        for f in sorted(os.listdir(dp)):
            if f.endswith(".h5"):
                s.append((ds, f.replace(".h5",""), os.path.join(dp, f)))
    return s

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--name", type=str, default="D16_R1_K4_C64_T3")
    args = parser.parse_args()
    print(f"\n{'='*60}")
    model = build_model_from_name(args.name).to(device)
    if not load_checkpoint(model, args.checkpoint): return
    model.eval()
    scenes = discover_scenes()
    print(f"  Scenes: {len(scenes)}")
    results = []
    for ds, label, fpath in scenes:
        try:
            lr, hr = load_h5_scene(fpath)
            lr, hr = lr.to(device), hr.to(device)
            with torch.no_grad():
                sr = model(lr)
            p = PSNR(center_view(hr), center_view(sr), data_range=1.0)
            s = SSIM(center_view(hr), center_view(sr), data_range=1.0)
            results.append((ds, label, p, s))
            print(f"  [{ds}/{label}] PSNR={p:.3f}  SSIM={s:.4f}")
        except Exception as e:
            print(f"  [ERR {ds}/{label}] {str(e)[:100]}")
    if results:
        dm = {}
        for ds,_,p,s in results: dm.setdefault(ds,[]).append((p,s))
        print(f"\n  ==== {args.name}: {len(results)} scenes ====")
        ap, aas = [], []
        for ds in sorted(dm.keys()):
            v = dm[ds]
            mp = np.mean([x[0] for x in v]); ms = np.mean([x[1] for x in v])
            print(f"    {ds}: PSNR={mp:.4f} SSIM={ms:.4f} (n={len(v)})")
            ap.extend([x[0] for x in v]); aas.extend([x[1] for x in v])
        print(f"    AVG: PSNR={np.mean(ap):.4f} SSIM={np.mean(aas):.4f} (n={len(ap)})")

if __name__ == "__main__":
    main()
