import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import hdf5storage
from datasets.PWDataLoaders import load_data
from scoring.metrics import res_FWHM
from scipy.interpolate import RectBivariateSpline as interp2d


def measure_point(beamformer, moniker, center_angle=False, verbose=True):
    x_fwhms = []
    z_fwhms = []
    bimgs = []
    exts = []
    for idx in range(4):
        torch.cuda.empty_cache()
        # Load ROI information
        mdict = hdf5storage.loadmat(os.path.join("scoring","rois","point","roi%02d.mat" % idx))
        data_source = mdict["data_source"]
        acq = mdict["acq"]
        img_grid = mdict["grid"]
        extent = mdict["extent"]
        pts = mdict["pts"]
        exts.append(extent)

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

        # Make grid based on ROIs
        bimg = beamformer(P, img_grid)
        bimgs.append(bimg)

        x_fwhm = []
        z_fwhm = []
        for i, pt in enumerate(pts):
            # One option is to directly beamform a line of pixels over the point targets
            # at high pixel density. However, this may be unfair to methods that rely on
            # a consistent pixel spacing, or to those that require 2D images to operate
            # upon.
            # Instead, we choose to interpolate the original beamformed image.
            ax = img_grid[:, 0, 2]  # Axial pixel positions [m]
            az = img_grid[0, :, 0]  # Azimuthal pixel positions [m]
            f = interp2d(ax, az, bimg)

            # Define pixel grid limits (assume y == 0)
            dp = P.c / P.fc / 3 / 100  # Interpolate 100-fold
            roi = np.arange(-1e-3, 1e-3, dp)
            roi -= np.mean(roi)
            zeros = np.zeros_like(roi)
            # Horizontal grid
            xroi = f(pt[2] + zeros, pt[0] + roi)[0]
            zroi = f(pt[2] + roi, pt[0] + zeros)[:, 0]

            x_fwhm.append(res_FWHM(xroi) * dp)
            z_fwhm.append(res_FWHM(zroi) * dp)

        x_fwhms.append(np.mean(x_fwhm))
        z_fwhms.append(np.mean(z_fwhm))
        if verbose:
            print("ROI %d, x-FWHM: %dum" % (idx, np.round(x_fwhms[idx] * 1e6)))
            print("ROI %d, z-FWHM: %dum" % (idx, np.round(z_fwhms[idx] * 1e6)))

    hdf5storage.savemat(
        os.path.join("results",moniker,"point"),
        {"x_fwhms": x_fwhms, "z_fwhms": z_fwhms, "bimgs": bimgs, "exts": exts},
    )

    # Display images via matplotlib
    plt.figure(figsize=[10, 6])
    plt.subplot(2, 3, 1)
    plt.imshow(_dB(bimgs[0]), vmin=-40, cmap="gray", extent=exts[0], origin="upper")
    plt.subplot(2, 3, 2)
    plt.imshow(_dB(bimgs[1]), vmin=-40, cmap="gray", extent=exts[1], origin="upper")
    plt.subplot(1, 3, 3)
    plt.imshow(_dB(bimgs[2]), vmin=-40, cmap="gray", extent=exts[2], origin="upper")
    plt.subplot(2, 2, 3)
    plt.imshow(_dB(bimgs[3]), vmin=-40, cmap="gray", extent=exts[3], origin="upper")
    plt.suptitle("Point Targets")
    plt.pause(0.01)
    outdir = os.path.join("results",moniker)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    plt.savefig(os.path.join(outdir,"point.jpg"))


def _dB(x):
    return 20 * np.log10(x)

