import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import hdf5storage
from datasets.PWDataLoaders import load_data
from scoring.metrics import contrast, cnr, gcnr


def measure_lesion(beamformer, moniker, center_angle=False, verbose=True):
    contrasts = []
    cnrs = []
    gcnrs = []
    brois = []
    bimgs = []
    for idx in range(8):
        torch.cuda.empty_cache()
        # Load ROI information
        mdict = hdf5storage.loadmat(os.path.join("scoring","rois","lesion","roi%02d.mat" % idx))
        data_source = mdict["data_source"]
        acq = mdict["acq"]
        # roi_grid = np.concatenate((mdict["grid_i"], mdict["grid_o"]), axis=0)
        roi_i = mdict["roi_i"]
        roi_o = mdict["roi_o"]
        img_grid = mdict["grid"]
        extent = mdict["extent"]
        # ni = mdict["grid_i"].shape[0]

        # Load plane wave channel data
        P, _, _ = load_data(data_source, acq)
        # Normalize input to [-1, 1] range
        maxval = np.maximum(np.abs(P.idata).max(), np.abs(P.qdata).max())
        P.idata /= maxval
        P.qdata /= maxval

        if center_angle:
            # Grab only the center angle's worth of data
            aidx = len(P.angles) // 2
            P.idata = P.idata[[aidx]]
            P.qdata = P.qdata[[aidx]]
            P.angles = P.angles[[aidx]]
            P.time_zero = P.time_zero[[aidx]]

        # Beamform the ROI and image
        # broi = beamformer(P, roi_grid)
        bimg = beamformer(P, img_grid)
        # brois.append(broi)
        bimgs.append(bimg)

        # Compute statistics
        # b_inner, b_outer = broi[:ni], broi[ni:]
        b_inner = bimgs[idx][roi_i]
        b_outer = bimgs[idx][roi_o]
        contrasts.append(contrast(b_inner, b_outer))
        cnrs.append(cnr(b_inner, b_outer))
        gcnrs.append(gcnr(b_inner, b_outer))

        if verbose:
            print("roi%02d Contrast: %f" % (idx, contrasts[idx]))
            print("roi%02d CNR: %f" % (idx, cnrs[idx]))
            print("roi%02d gCNR: %f" % (idx, gcnrs[idx]))

    hdf5storage.savemat(
        os.path.join("results",moniker,"lesion"),
        {
            "contrasts": contrasts,
            "cnrs": cnrs,
            "gcnrs": gcnrs,
            "brois": brois,
            "bimgs": bimgs,
        },
    )
    plt.figure(figsize=[10, 6])
    for idx in range(8):
        # Display images via matplotlib
        plt.subplot(2, 4, idx + 1)
        plt.imshow(
            20 * np.log10(bimgs[idx]),
            vmin=-40,
            cmap="gray",
            extent=extent,
            origin="upper",
        )

    plt.suptitle("%s: Anechoic Lesions" % moniker)
    plt.pause(0.01)
    outdir = os.path.join("results",moniker)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    plt.savefig(os.path.join(outdir,"lesion.jpg"))
