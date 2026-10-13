import os
import glob
import numpy as np
import h5py
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
import csv
from models.LFTransMamba import LFTransMamba

angRes = 5
scale_factor = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/ablation_gftransmamba_v2.pth"
MODEL_NAME = "LFTransMamba_D16_R1_K4_C64_T3"
TEST_ROOT = "/root/autodl-tmp/ZJL/h5_data/test"
USE_TTA = True


def build_model(model_name, angRes, scale):
    params = {}
    for part in model_name.split("_"):
        if part.startswith("D"):   params["D"] = int(part[1:])
        elif part.startswith("R"): params["R"] = int(part[1:])
        elif part.startswith("K"): params["K"] = int(part[1:])
        elif part.startswith("C"): params["C"] = int(part[1:])
        elif part.startswith("T"): params["T"] = int(part[1:])
    return LFTransMamba(angRes_in=angRes, scale_factor=scale, **params)


def inference_tta(model, lr):
    preds = []
    with torch.no_grad():
        preds.append(model(lr))
        preds.append(model(lr.rot90(1, [2, 3]).flip([2])).flip([2]).rot90(3, [2, 3]))
        preds.append(model(lr.flip([2])).flip([2]))
        preds.append(model(lr.rot90(3, [2, 3])).rot90(1, [2, 3]))
        preds.append(model(lr.rot90(2, [2, 3]).flip([2])).flip([2]).rot90(2, [2, 3]))
        preds.append(model(lr.rot90(1, [2, 3])).rot90(3, [2, 3]))
        preds.append(model(lr.rot90(2, [2, 3])).rot90(2, [2, 3]))
        preds.append(model(lr.rot90(3, [2, 3]).flip([2])).flip([2]).rot90(1, [2, 3]))
    return torch.stack(preds, dim=0).mean(dim=0)


def main():
    model = build_model(MODEL_NAME, angRes, scale_factor).to(device)

    if os.path.exists(CHECKPOINT_PATH):
        state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        clean = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(clean, strict=False)
        print("Loaded: " + CHECKPOINT_PATH)
    else:
        print("[WARNING] Not found: " + CHECKPOINT_PATH)

    model.eval()

    h5_files = sorted(glob.glob(os.path.join(TEST_ROOT, "**/*.h5"), recursive=True))
    print("Test scenes: %d | TTA: %s" % (len(h5_files), "ON" if USE_TTA else "OFF"))

    results = []

    for h5_path in h5_files:
        scene_name = os.path.splitext(os.path.basename(h5_path))[0]
        dataset = h5_path.split("/")[-2]  # EPFL / HCI / Stanford

        try:
            with h5py.File(h5_path, "r") as hf:
                lr_np = np.array(hf["Lr_SAI_y"], dtype=np.float32)
                hr_np = np.array(hf["Hr_SAI_y"], dtype=np.float32)

            lr = torch.from_numpy(lr_np).unsqueeze(0).unsqueeze(0).to(device)
            hr = torch.from_numpy(hr_np).unsqueeze(0).unsqueeze(0).to(device)

            # Crop to square and multiple of 5
            uh, vw = lr.shape[2], lr.shape[3]
            side = min(uh, vw)
            side = (side // angRes) * angRes
            lr = lr[:, :, :side, :side]
            hr = hr[:, :, :side * scale_factor, :side * scale_factor]

            # Inference
            if USE_TTA:
                sr = inference_tta(model, lr)
            else:
                with torch.no_grad():
                    sr = model(lr)

            sr = torch.clamp(sr, 0, 1)

            # PSNR / SSIM
            sr_np = sr.squeeze().cpu().numpy()
            hr_np = hr.squeeze().cpu().numpy()

            if sr_np.shape != hr_np.shape:
                from PIL import Image as PILImage
                sr_np = np.array(PILImage.fromarray(
                    (np.clip(sr_np, 0, 1) * 255).astype(np.uint8)
                ).resize((hr_np.shape[1], hr_np.shape[0]))).astype(np.float32) / 255.0

            psnr_val = PSNR(hr_np, sr_np, data_range=1.0)
            ssim_val = SSIM(hr_np, sr_np, data_range=1.0)

            print("[%s/%s] PSNR=%.3f  SSIM=%.4f" % (dataset, scene_name, psnr_val, ssim_val))
            results.append([dataset, scene_name, psnr_val, ssim_val])

        except Exception as e:
            import traceback
            print("[Skip %s/%s] %s" % (dataset, scene_name, str(e)))
            traceback.print_exc()

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/test_results_h5.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "scene", "PSNR", "SSIM"])
        writer.writerows(results)

    if results:
        psnrs = [r[2] for r in results]
        ssims = [r[3] for r in results]
        print("\n===== FINAL RESULT =====")
        print("Scenes: %d" % len(results))
        print("AVG PSNR: %.4f" % np.mean(psnrs))
        print("AVG SSIM: %.4f" % np.mean(ssims))

        for prefix in ["EPFL", "HCI", "Stanford"]:
            sub = [r for r in results if r[0] == prefix]
            if sub:
                print("  %s: PSNR=%.4f SSIM=%.4f (n=%d)" % (
                    prefix,
                    np.mean([r[2] for r in sub]),
                    np.mean([r[3] for r in sub]),
                    len(sub)))
    else:
        print("No valid results!")


if __name__ == "__main__":
    main()
