import os, sys, re, numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
import csv
sys.path.insert(0, os.path.dirname(__file__) or '.')
from models.LFTransMamba import LFTransMamba

angRes, scale = 5, 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
SELECT = [2, 3, 4, 5, 6]
TEST_ROOT = '/root/autodl-tmp/ZJL/train_ready_all/test'

EXPERIMENTS = [
    ('ablation_baseline_v2',    False, False, False),
    ('ablation_gaam_v2',        True,  False, False),
    ('ablation_gaam_cvesa_v2',  True,  True,  False),
    ('ablation_gftransmamba_v2', True,  True,  True),
]

def rgb2y(img):
    img = img.astype(np.float32) / 255.0
    y = 65.481 * img[..., 0] + 128.553 * img[..., 1] + 24.966 * img[..., 2] + 16.0
    return np.clip(y / 255.0, 0.0, 1.0)

def load_test(scene_path):
    imgs = sorted(f for f in os.listdir(scene_path) if f.startswith('input_Cam') and f.endswith('.png'))
    views = np.stack([rgb2y(np.array(Image.open(os.path.join(scene_path, f)).convert('RGB'))) for f in imgs], 0)
    H, W = views.shape[1], views.shape[2]
    views = views.reshape(9, 9, H, W)[np.ix_(SELECT, SELECT)]
    return H, W, views

def sai_from_views(views):
    u, v = 5, 5
    H, W = views.shape[2], views.shape[3]
    sai = views.transpose(0, 2, 1, 3).reshape(u * H, v * W)
    return torch.from_numpy(sai).float().unsqueeze(0).unsqueeze(0)

def center_view(sai, H, W):
    arr = sai.squeeze().cpu().numpy()
    return arr[2 * H:3 * H, 2 * W:3 * W]

for name, use_gaam, use_cvesa, use_mfpm in EXPERIMENTS:
    print('\n' + '=' * 60)
    print('Evaluating:', name)
    print('Config: gaam={} cvesa={} mfpm={}'.format(use_gaam, use_cvesa, use_mfpm))

    model = LFTransMamba(angRes_in=angRes, scale_factor=scale, D=16, R=1, K=4, C=64, T=3,
                         use_gaam=use_gaam, use_cvesa=use_cvesa, use_mfpm=use_mfpm).to(device)

    ckpt_path = 'checkpoints/{}.pth'.format(name)
    if not os.path.exists(ckpt_path):
        print('  [SKIP] Checkpoint not found:', ckpt_path)
        continue

    sd = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in sd:
        sd = sd['state_dict']
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print('  Params: {:.2f}M'.format(n_params))

    scenes = sorted(s for s in os.listdir(TEST_ROOT)
                    if os.path.isdir(os.path.join(TEST_ROOT, s)) and not s.startswith('.'))
    print('  Scenes:', len(scenes))

    results = []
    for scene in scenes:
        try:
            sp = os.path.join(TEST_ROOT, scene)
            H, W, views = load_test(sp)
            hr_sai = sai_from_views(views).to(device)
            lr = F.interpolate(hr_sai, scale_factor=0.25, mode='bicubic', align_corners=False)

            with torch.no_grad():
                sr = model(lr)

            hr_cv = center_view(hr_sai, H, W)
            sr_cv = center_view(sr, H, W)

            p = PSNR(hr_cv, sr_cv, data_range=1.0)
            s = SSIM(hr_cv, sr_cv, data_range=1.0)
            results.append([scene, p, s])
            print('  [{}] PSNR={:.3f}  SSIM={:.4f}'.format(scene, p, s))
        except Exception as e:
            import traceback; traceback.print_exc()
            print('  [Skip {}] {}'.format(scene, e))

    if results:
        psnrs = [r[1] for r in results]
        ssims = [r[2] for r in results]
        print('  ==== {}: PSNR={:.4f} SSIM={:.4f} (n={}) ===='.format(
            name, np.mean(psnrs), np.mean(ssims), len(results)))

    os.makedirs('results', exist_ok=True)
    with open('results/{}_png.csv'.format(name), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['scene', 'PSNR', 'SSIM'])
        w.writerows(results)
    print('  Saved: results/{}_png.csv'.format(name))

print('\nAll experiments complete!')
