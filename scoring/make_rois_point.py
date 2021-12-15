# File:       make_rois_point.py
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


def rois_point(idx, outdir=os.path.join("scoring","rois","point")):
    if idx == 0:
        data_source, acq = "MYO", 3
        P, _, _ = load_data(data_source, acq)
        xlims = [0e-3, 20e-3]
        zlims = [22e-3, 32e-3]
        pts = [[13.4e-3, 0, 27.75e-3], [14.55e-3, 0, 26.8e-3], [15.15e-3, 0, 25.85e-3]]
    elif idx == 1:
        data_source, acq = "UFL", 4
        P, _, _ = load_data(data_source, acq)
        xlims = [-6e-3, 10e-3]
        zlims = [22e-3, 32e-3]
        pts = [[-2.8e-3, 0, 27.65e-3], [-3.95e-3, 0, 26.7e-3], [-4.56e-3, 0, 25.7e-3]]
    elif idx == 2:
        data_source, acq = "UFL", 2
        P, _, _ = load_data(data_source, acq)
        xlims = [-3e-3, 3e-3]
        zlims = [12e-3, 38e-3]
        pts = [
            [-0.29e-3, 0, 15.5e-3],
            [-0.31e-3, 0, 20.6e-3],
            [-0.43e-3, 0, 25.4e-3],
            [-0.20e-3, 0, 30.8e-3],
            [-0.12e-3, 0, 35.7e-3],
        ]
    elif idx == 3:
        data_source, acq = "MYO", 2
        P, _, _ = load_data(data_source, acq)
        xlims = [-15e-3, 11e-3]
        zlims = [38e-3, 42e-3]
        pts = [[-11.95e-3, 0, 39.4e-3], [-2e-3, 0, 39.55e-3], [8e-3, 0, 39.6e-3]]
    else:
        raise NotImplementedError

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
    plt.imshow(bimgN, vmin=-40, cmap="gray", extent=extent, origin="upper")
    plt.suptitle("%s%03d (c = %d m/s)" % (data_source, acq, np.round(P.c)))
    bar = np.array([-1e-3, 1e-3])
    tmp = np.array([0, 0])
    for pt in pts:
        plt.plot((pt[0] + bar) * 1e3, (pt[2] + tmp) * 1e3, "c-")
        plt.plot((pt[0] + tmp) * 1e3, (pt[2] + bar) * 1e3, "c-")
    # Check to make sure save directory exists, if not, create it
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    plt.savefig(os.path.join(outdir, "roi%02d.png" % (idx)))

    # Save
    mdict = {
        "grid": grid,
        "data_source": data_source,
        "acq": acq,
        "extent": extent,
        "pts": pts,
    }
    hdf5storage.savemat(os.path.join(outdir, "roi%02d" % (idx)), mdict)


if __name__ == "__main__":
    for i in range(4):
        rois_point(i)

