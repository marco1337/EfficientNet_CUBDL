import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import hdf5storage
from datasets.PWDataLoaders import load_data
from scoring.metrics import l1loss, l2loss, psnr, ncc, wopt_mae, wopt_mse


def measure_additional(beamformer, moniker, center_angle=False, verbose=True):
    nrois = 11
    l1_lins, l2_lins, l1_logs, l2_logs, = [], [], [], []
    psnrs, nccs, bimgs, exts = [], [], [], []
    for idx in range(nrois):
        torch.cuda.empty_cache()
        # Load ROI information
        mdict = hdf5storage.loadmat(os.path.join("scoring","rois","additional","roi%02d.mat" % idx))
        data_source = mdict["data_source"]
        acq = mdict["acq"]
        img_grid = mdict["grid"]
        extent = mdict["extent"]
        yimg = mdict["ground_truth"]
        exts.append(extent)

        # Load plane wave channel data
        P, _, _ = load_data(data_source, acq)
        if center_angle:
            # Grab only the center angle's worth of data
            aidx = np.where(P.angles == 0) #len(P.angles) // 2
            P.idata = P.idata[aidx]
            P.qdata = P.qdata[aidx]
            P.angles = P.angles[aidx]
            P.time_zero = P.time_zero[aidx]

        # Normalize input to [-1, 1] range
        maxval = np.maximum(np.abs(P.idata).max(), np.abs(P.qdata).max())
        P.idata /= maxval
        P.qdata /= maxval

        # Beamform the ROI
        bimg = beamformer(P, img_grid)

        # Statistics will only be computed using image values within -40dB of max value
        mask = (20 * np.log10(yimg / np.amax(yimg))) >= -40
        mask2 = (20 * np.log10(bimg / np.amax(bimg))) < -40
        bimg[mask & mask2] = 10 ** (-40 / 20) * np.amax(bimg)

        # Find L1, L2 optimal image scaling factors
        bimg_l1 = bimg * wopt_mae(yimg[mask], bimg[mask])
        bimg_l2 = bimg * wopt_mse(yimg[mask], bimg[mask])
        bimgs.append(bimg_l2)

        # Compute statistics
        l1_lins.append(l1loss(yimg[mask], bimg_l1[mask]))
        l2_lins.append(l2loss(yimg[mask], bimg_l2[mask]))
        l1_logs.append(l1loss(np.log(yimg[mask]), np.log(bimg_l1[mask])))
        l2_logs.append(l2loss(np.log(yimg[mask]), np.log(bimg_l2[mask])))
        psnrs.append(psnr(yimg[mask], bimg_l2[mask]))
        nccs.append(ncc(yimg[mask], bimg[mask]))

        if verbose:
            print("roi%02d L1 lin: %f" % (idx, l1_lins[idx]))
            print("roi%02d L1 log: %f" % (idx, l1_logs[idx]))
            print("roi%02d L2 lin: %f" % (idx, l2_lins[idx]))
            print("roi%02d L2 log: %f" % (idx, l2_logs[idx]))
            print("roi%02d PSNR:   %f" % (idx, psnrs[idx]))
            print("roi%02d NCC:    %f" % (idx, nccs[idx]))

    hdf5storage.savemat(
        os.path.join("results",moniker,"additional"),
        {
            "l1_lins": l1_lins,
            "l1_logs": l1_logs,
            "l2_lins": l2_lins,
            "l2_logs": l2_logs,
            "psnrs": psnrs,
            "nccs": nccs,
            "bimgs": bimgs,
            "mask": mask,
        },
    )

    plt.figure(figsize=[10, 6])
    for idx in range(nrois):
        # Display images via matplotlib
        plt.subplot(3, 4, idx + 1)
        plt.imshow(
            20 * np.log10(bimgs[idx]),
            vmin=-60,
            cmap="gray",
            extent=exts[idx],
            origin="upper",
        )

    plt.suptitle("%s: Additional Targets" % moniker)
    plt.pause(0.01)
    outdir = os.path.join("results",moniker)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    plt.savefig(os.path.join(outdir,"additional.jpg"))
