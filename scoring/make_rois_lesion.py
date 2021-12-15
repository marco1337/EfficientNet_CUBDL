# File:       make_rois_lesion.py
# Author:     Dongwoon Hyun (dongwoon.hyun@stanford.edu)
# Created on: 2020-07-27
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from datasets.PWDataLoaders import load_data
from cubdl.PixelGrid import make_pixel_grid
from cubdl.das_torch import DAS_PW
import hdf5storage


device = torch.device("cuda:0")


def rois_lesion(idx, outdir=os.path.join("scoring","rois","lesion")):
    if idx == 0:
        data_source, acq = "UFL", 1
        xctr, zctr, r0, r1 = -0.8e-3, 30.3e-3, 2.8e-3, 4.5e-3
        P, _, _ = load_data(data_source, acq)
    elif idx == 1:
        data_source, acq = "UFL", 5
        xctr, zctr, r0, r1 = 0.2e-3, 16.4e-3, 2.2e-3, 3.5e-3
        P, _, _ = load_data(data_source, acq)
    elif idx == 2:
        data_source, acq = "UFL", 5
        xctr, zctr, r0, r1 = -0.5e-3, 45e-3, 2.2e-3, 3.5e-3
        P, _, _ = load_data(data_source, acq)
    elif idx == 3:
        data_source, acq = "OSL", 7
        xctr, zctr, r0, r1 = -8.2e-3, 39.1e-3, 2.5e-3, 4.5e-3
        P, _, _ = load_data(data_source, acq)
    elif idx == 4:
        data_source, acq = "MYO", 1
        xctr, zctr, r0, r1 = 15.5e-3, 16.5e-3, 1.3e-3, 3e-3
        P, _, _ = load_data(data_source, acq)
    elif idx == 5:
        data_source, acq = "MYO", 4
        xctr, zctr, r0, r1 = 12.2e-3, 25.8e-3, 1.3e-3, 3e-3
        P, _, _ = load_data(data_source, acq)
    elif idx == 6:
        data_source, acq = "INS", 8
        xctr, zctr, r0, r1 = 11.4e-3, 42.3e-3, 2.8e-3, 4.5e-3
        P, _, _ = load_data(data_source, acq)
    elif idx == 7:
        data_source, acq = "INS", 21
        xctr, zctr, r0, r1 = -7e-3, 41.2e-3, 2.8e-3, 4.5e-3
        P, _, _ = load_data(data_source, acq)
    else:
        raise NotImplementedError

    xlims = [-6e-3 + xctr, 6e-3 + xctr]
    zlims = [-6e-3 + zctr, 6e-3 + zctr]
    r2 = np.sqrt(r0 ** 2 + r1 ** 2)

    # Define pixel grid limits (assume y == 0)
    wvln = P.c / P.fc
    dx = wvln / 3
    dz = dx  # Use square pixels
    grid = make_pixel_grid(xlims, zlims, dx, dz)
    fnum = 1

    # Normalize input to [-1, 1] range
    maxval = np.maximum(np.abs(P.idata).max(), np.abs(P.qdata).max())
    P.idata /= maxval
    P.qdata /= maxval

    # Make ROI
    dist = np.sqrt((grid[:, :, 0] - xctr) ** 2 + (grid[:, :, 2] - zctr) ** 2)
    roi_i = dist <= r0
    roi_o = (r1 <= dist) * (dist <= r2)

    # Make data torch tensors
    x = (P.idata, P.qdata)

    # Make 75-angle image
    dasN = DAS_PW(P, grid, rxfnum=fnum)
    idasN, qdasN = dasN(x)
    idasN, qdasN = idasN.detach().cpu().numpy(), qdasN.detach().cpu().numpy()
    iqN = idasN + 1j * qdasN
    bimgN = 20 * np.log10(np.abs(iqN))  # Log-compress
    bimgN -= np.amax(bimgN)  # Normalize by max value

    # Display images via matplotlib
    xext = (np.array([-0.5, grid.shape[1] - 0.5]) * dx + xlims[0]) * 1e3
    zext = (np.array([-0.5, grid.shape[0] - 0.5]) * dz + zlims[0]) * 1e3
    extent = [xext[0], xext[1], zext[1], zext[0]]
    plt.clf()
    opts = {"extent": extent, "origin": "upper"}
    plt.imshow(bimgN, vmin=-40, cmap="gray", **opts)
    plt.contour(roi_i, [0.5], colors="c", **opts)
    plt.contour(roi_o, [0.5], colors="m", **opts)
    plt.suptitle("%s%03d (c = %d m/s)" % (data_source, acq, np.round(P.c)))
    # Check to make sure save directory exists, if not, create it
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    plt.savefig(os.path.join(outdir, "roi%02d.png" % (idx)))

    # Save ROI locations
    xgrid = grid[:, :, 0]
    ygrid = grid[:, :, 1]
    zgrid = grid[:, :, 2]
    grid_i = np.stack([xgrid[roi_i], ygrid[roi_i], zgrid[roi_i]], axis=1)
    grid_o = np.stack([xgrid[roi_o], ygrid[roi_o], zgrid[roi_o]], axis=1)
    grid_i = np.expand_dims(grid_i, 1)
    grid_o = np.expand_dims(grid_o, 1)
    mdict = {
        "grid": grid,
        "grid_i": grid_i,
        "grid_o": grid_o,
        "data_source": data_source,
        "acq": acq,
        "xctr": xctr,
        "zctr": zctr,
        "r0": r0,
        "r1": r1,
        "r2": r2,
        "extent": extent,
        "roi_i": roi_i,
        "roi_o": roi_o,
    }
    hdf5storage.savemat(os.path.join(outdir, "roi%02d" % (idx)), mdict)


if __name__ == "__main__":
    for i in range(8):
        rois_lesion(i)

