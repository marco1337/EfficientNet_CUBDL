# File:       make_rois_additional.py
# Author:     Alycen Wiacek (awiacek1@jhu.edu)
# Created on: 2020-08-25
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from datasets.PWDataLoaders import load_data
from cubdl.PixelGrid import make_pixel_grid
from cubdl.das_torch import DAS_PW
import hdf5storage


def rois_additional(idx, outdir=os.path.join("scoring","rois","additional")):
    images = [
        ("JHU", 24),
        ("JHU", 25),
        ("JHU", 26),
        ("JHU", 27),
        ("JHU", 28),
        ("JHU", 29),
        ("JHU", 30),
        ("JHU", 31),
        ("JHU", 32),
        ("JHU", 33),
        ("JHU", 34),
    ]
    data_source, acq = images[idx]
    P, xl, zl = load_data(data_source, acq)

    # Define pixel grid limits (assume y == 0)
    wvln = P.c / P.fc
    dx = wvln / 3
    dz = dx  # Use square pixels
    desired_grid = [400, 300]
    xlims = np.array([-0.5, 0.5]) * (desired_grid[1] - 1) * dx + (xl[0] + xl[1]) / 2
    zlims = np.array([-0.5, 0.5]) * (desired_grid[0] - 1) * dz + (zl[0] + zl[1]) / 2
    xlims = [np.maximum(xlims[0], xl[0]), np.minimum(xlims[1], xl[1])]
    zlims = [np.maximum(zlims[0], zl[0]), np.minimum(zlims[1], zl[1])]
    grid = make_pixel_grid(xlims, zlims, dx, dz)
    fnum = 1

    # Normalize input to [-1, 1] range
    maxval = np.maximum(np.abs(P.idata).max(), np.abs(P.qdata).max())
    P.idata /= maxval
    P.qdata /= maxval

    # Make data torch tensors
    x = (P.idata, P.qdata)

    # Make 75-angle image
    dasN = DAS_PW(P, grid, rxfnum=fnum)
    idasN, qdasN = dasN(x)
    idasN, qdasN = idasN.detach().cpu().numpy(), qdasN.detach().cpu().numpy()
    iqN = idasN + 1j * qdasN
    ground_truth = np.abs(iqN)
    ground_truth /= np.amax(ground_truth)

    # Normalize to have RMS = 1
    # ground_truth /= np.sqrt(np.mean(ground_truth ** 2))

    # Display images via matplotlib
    xext = (np.array([-0.5, grid.shape[1] - 0.5]) * dx + xlims[0]) * 1e3
    zext = (np.array([-0.5, grid.shape[0] - 0.5]) * dz + zlims[0]) * 1e3
    extent = [xext[0], xext[1], zext[1], zext[0]]
    plt.clf()
    bimgN = 20 * np.log10(ground_truth)  # Log-compress
    bimgN -= np.amax(bimgN)  # Normalize by max value
    plt.imshow(bimgN, vmin=-60, cmap="gray", extent=extent, origin="upper")
    plt.suptitle("%s%03d (c = %d m/s)" % (data_source, acq, np.round(P.c)))
    # Check to make sure save directory exists, if not, create it
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    plt.savefig(os.path.join(outdir,"roi%02d.png" % (idx)))

    # Save
    mdict = {
        "grid": grid,
        "data_source": data_source,
        "acq": acq,
        "extent": extent,
        "ground_truth": ground_truth,
    }
    hdf5storage.savemat(os.path.join(outdir,"roi%02d" % (idx)), mdict)


if __name__ == "__main__":
    for i in range(11):
        rois_additional(i)

