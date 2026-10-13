import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

ROOT_PATH = '/root/autodl-tmp/ZJL/train_ready'

SAVE_VIS = './sanity_vis'

os.makedirs(SAVE_VIS, exist_ok=True)

angRes = 5
scale = 4

h5_files = sorted([
    f for f in os.listdir(ROOT_PATH)
    if f.endswith('.h5')
])

print('=' * 60)
print('Found H5 files:')
print(h5_files)
print('=' * 60)

for h5_name in h5_files:

    path = os.path.join(ROOT_PATH, h5_name)

    print(f'\nChecking: {h5_name}')

    with h5py.File(path, 'r') as hf:

        # -------------------------
        # load
        # -------------------------
        Hr_SAI_y = hf['Hr_SAI_y'][:]
        Lr_SAI_y = hf['Lr_SAI_y'][:]
        Sr_SAI_cbcr = hf['Sr_SAI_cbcr'][:]

        # -------------------------
        # basic info
        # -------------------------
        print('HR shape:', Hr_SAI_y.shape)
        print('LR shape:', Lr_SAI_y.shape)
        print('CbCr shape:', Sr_SAI_cbcr.shape)

        # -------------------------
        # NaN check
        # -------------------------
        assert not np.isnan(Hr_SAI_y).any(), 'HR contains NaN'
        assert not np.isnan(Lr_SAI_y).any(), 'LR contains NaN'

        # -------------------------
        # empty check
        # -------------------------
        assert Hr_SAI_y.size > 0, 'Empty HR'
        assert Lr_SAI_y.size > 0, 'Empty LR'

        # -------------------------
        # scale check
        # -------------------------
        assert Hr_SAI_y.shape[0] == Lr_SAI_y.shape[0] * scale, \
            'Height scale mismatch'

        assert Hr_SAI_y.shape[1] == Lr_SAI_y.shape[1] * scale, \
            'Width scale mismatch'

        # -------------------------
        # angular resolution check
        # -------------------------
        assert Hr_SAI_y.shape[0] % angRes == 0, \
            'HR height not divisible by angRes'

        assert Hr_SAI_y.shape[1] % angRes == 0, \
            'HR width not divisible by angRes'

        # -------------------------
        # recover single view size
        # -------------------------
        H = Hr_SAI_y.shape[0] // angRes
        W = Hr_SAI_y.shape[1] // angRes

        print(f'Single view size: {H} x {W}')

        # -------------------------
        # split into views
        # -------------------------
        views = []

        for u in range(angRes):
            for v in range(angRes):

                view = Hr_SAI_y[
                    u * H:(u + 1) * H,
                    v * W:(v + 1) * W
                ]

                views.append(view)

        # -------------------------
        # view count check
        # -------------------------
        assert len(views) == angRes * angRes

        # -------------------------
        # visualization
        # -------------------------
        fig = plt.figure(figsize=(10, 10))

        plt.subplot(2, 2, 1)
        plt.imshow(Hr_SAI_y, cmap='gray')
        plt.title('HR SAI')

        plt.subplot(2, 2, 2)
        plt.imshow(Lr_SAI_y, cmap='gray')
        plt.title('LR SAI')

        plt.subplot(2, 2, 3)
        plt.imshow(views[0], cmap='gray')
        plt.title('Top-left View')

        plt.subplot(2, 2, 4)
        plt.imshow(views[len(views)//2], cmap='gray')
        plt.title('Center View')

        save_path = os.path.join(
            SAVE_VIS,
            h5_name.replace('.h5', '.png')
        )

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        print(f'Saved visualization: {save_path}')

print('\n' + '=' * 60)
print('ALL H5 FILES PASSED SANITY CHECK')
print('=' * 60)
