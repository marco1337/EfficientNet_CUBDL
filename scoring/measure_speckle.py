import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import hdf5storage
from datasets.PWDataLoaders import load_data
from scoring.metrics import snr


nrois = 6


def measure_speckle(beamformer, moniker, center_angle=False, verbose=True):
    snrs = []
    bimgs = []
    for idx in range(nrois):
        torch.cuda.empty_cache()
        # Load ROI information
        mdict = hdf5storage.loadmat(os.path.join("scoring","rois","speckle","roi%02d.mat" % idx))
        data_source = mdict["data_source"]
        acq = mdict["acq"]
        img_grid = mdict["grid"]
        extent = mdict["extent"]

        # Load plane wave channel data
        P, _, _ = load_data(data_source, acq)
        if center_angle:
            # Grab only the center angle's worth of data
            aidx = len(P.angles) // 2
            P.idata = P.idata[[aidx]]
            P.qdata = P.qdata[[aidx]]
            P.angles = P.angles[[aidx]]
            P.time_zero = P.time_zero[[aidx]]

        # Normalize input to [-1, 1] range
        maxval = np.maximum(np.abs(P.idata).max(), np.abs(P.qdata).max())
        P.idata /= maxval
        P.qdata /= maxval

        # Beamform the ROI
        bimg = beamformer(P, img_grid)
        bimgs.append(bimg)

        # Compute statistics
        snrs.append(snr(bimg))

        if verbose:
            print("roi%02d SNR: %f" % (idx, snrs[idx]))

    hdf5storage.savemat(os.path.join("results",moniker,"speckle"), {"snrs": snrs, "bimgs": bimgs})

    plt.figure(figsize=[10, 6])
    for idx in range(nrois):
        # Display images via matplotlib
        plt.subplot(2, 3, idx + 1)
        plt.imshow(
            20 * np.log10(bimgs[idx]),
            vmin=-40,
            cmap="gray",
            extent=extent,
            origin="upper",
        )

    plt.suptitle("%s: Speckle Targets" % moniker)
    plt.pause(0.01)
    outdir = os.path.join("results",moniker)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    plt.savefig(os.path.join(outdir,"speckle.jpg"))

