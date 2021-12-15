# File:       make_rois_speckle.py
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


def rois_speckle(idx, outdir=os.path.join("scoring","rois","speckle")):
    if idx == 0:
        data_source, acq = "UFL", 4
        P, _, _ = load_data(data_source, acq)
        xlims = [0e-3, 12e-3]
        zlims = [14e-3, 26e-3]
    elif idx == 1:
        data_source, acq = "OSL", 7
        P, _, _ = load_data(data_source, acq)
        xlims = [-6e-3, 6e-3]
        zlims = [14e-3, 26e-3]
    elif idx == 2:
        data_source, acq = "MYO", 2
        P, _, _ = load_data(data_source, acq)
        xlims = [-6e-3, 6e-3]
        zlims = [14e-3, 26e-3]
    elif idx == 3:
        data_source, acq = "EUT", 3
        P, _, _ = load_data(data_source, acq)
        xlims = [-6e-3, 6e-3]
        zlims = [25e-3, 37e-3]
    elif idx == 4:
        data_source, acq = "INS", 4
        P, _, _ = load_data(data_source, acq)
        xlims = [-6e-3, 6e-3]
        zlims = [25e-3, 37e-3]
    elif idx == 5:
        data_source, acq = "INS", 16
        P, _, _ = load_data(data_source, acq)
        xlims = [-6e-3, 6e-3]
        zlims = [25e-3, 37e-3]
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
    # Check to make sure save directory exists, if not, create it
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    plt.savefig(os.path.join(outdir, "roi%02d.png" % (idx)))

    # Save
    mdict = {"grid": grid, "data_source": data_source, "acq": acq, "extent": extent}
    hdf5storage.savemat(os.path.join(outdir, "roi%02d" % (idx)), mdict)


if __name__ == "__main__":
    for i in range(6):
        rois_speckle(i)

