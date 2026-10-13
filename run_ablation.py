import os, re

LFTRANSMAMBA = "models/LFTransMamba.py"
TRAIN_PY = "train.py"
TEST_PY = "test.py"
BACKUP = ".backup_ablation"

CONFIGS = [
    ("baseline",     False, False, False),
    ("gaam",         True,  False, False),
    ("gaam_cvesa",   True,  True,  False),
    ("gftransmamba", True,  True,  True),
]

def rd(p):
    with open(p, "r", encoding="utf-8") as f: return f.read()
def wr(p, c):
    with open(p, "w", encoding="utf-8") as f: f.write(c)

def toggle_fwd(code, gaam, cvesa, mfpm):
    out = []
    for line in code.split('\n'):
        if 'buffer_5d = self.gaam(buffer_5d' in line:
            out.append(line if gaam else '    pass  # GAAM off')
        elif 'self.cvesa(buffer_temp' in line:
            out.append(line if cvesa else '    pass  # CVESA off')
        elif 'buffer_5d = self.mfpm(buffer_5d' in line:
            out.append(line if mfpm else '    pass  # MFPM off')
        else:
            out.append(line)
    return '\n'.join(out)

def main():
    for f in [LFTRANSMAMBA, TRAIN_PY, TEST_PY]:
        if not os.path.exists(f + BACKUP): wr(f + BACKUP, rd(f))
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    for name, gaam, cvesa, mfpm in CONFIGS:
        print("\n" + "="*50)
        print("Experiment: %s (GAAM=%s CVESA=%s MFPM=%s)" % (name, gaam, cvesa, mfpm))
        print("="*50)
        for f in [LFTRANSMAMBA, TRAIN_PY, TEST_PY]: wr(f, rd(f + BACKUP))
        os.system("python patch_gftransmamba.py")
        wr(LFTRANSMAMBA, toggle_fwd(rd(LFTRANSMAMBA), gaam, cvesa, mfpm))
        wr(TRAIN_PY, rd(TRAIN_PY).replace(
            'torch.save(model.state_dict(), "checkpoints/best_model.pth")',
            'torch.save(model.state_dict(), "checkpoints/ablation_%s.pth")' % name))
        print("Training %s ..." % name)
        if os.system("python train.py") != 0:
            print("FAILED: %s" % name); continue
        wr(TEST_PY, rd(TEST_PY).replace(
            re.findall(r'CHECKPOINT_PATH = "checkpoints/[^"]*"', rd(TEST_PY))[0],
            'CHECKPOINT_PATH = "checkpoints/ablation_%s.pth"' % name))
        print("Testing %s ..." % name)
        os.system("python test.py | tee results/ablation_%s.txt" % name)

    for f in [LFTRANSMAMBA, TRAIN_PY, TEST_PY]: wr(f, rd(f + BACKUP))
    print("\nDone! Results in results/ablation_*.txt")

if __name__ == "__main__":
    main()
