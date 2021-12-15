import os
import matplotlib.pyplot as plt
import numpy as np
import hdf5storage
from cubdl.PlaneWaveData import PICMUSData
from cubdl.PixelGrid import make_pixel_grid


def measure_picmus(beamformer, moniker, center_angle=False, verbose=True):
    # Load PICMUS dataset
    database_path = os.path.join("datasets","picmus")
    acq = "simulation"
    target = "contrast_speckle"
    dtype = "rf"
    P = PICMUSData(database_path, acq, target, dtype)

    # Define pixel grid limits (assume y == 0)
    xlims = [P.ele_pos[0, 0], P.ele_pos[-1, 0]]
    zlims = [5e-3, 55e-3]
    wvln = P.c / P.fc
    dx = wvln / 3
    dz = dx  # Use square pixels
    grid = make_pixel_grid(xlims, zlims, dx, dz)
    xext = (np.array([-0.5, grid.shape[1] - 0.5]) * dx + xlims[0]) * 1e3
    zext = (np.array([-0.5, grid.shape[0] - 0.5]) * dz + zlims[0]) * 1e3
    extent = [xext[0], xext[1], zext[1], zext[0]]

    if center_angle:
        # Grab only the center angle's worth of data
        aidx = len(P.angles) // 2
        P.idata = P.idata[[aidx]]
        P.qdata = P.qdata[[aidx]]
        P.angles = P.angles[[aidx]]
        P.time_zero = P.time_zero[[aidx]]

    # Beamform the ROI
    bimg = beamformer(P, grid)

    plt.figure(figsize=[5, 6])
    plt.imshow(
        20 * np.log10(bimg), vmin=-60, cmap="gray", extent=extent, origin="upper"
    )
    plt.suptitle("%s: PICMUS" % moniker)
    plt.pause(0.01)
    outdir = os.path.join("results",moniker)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    plt.savefig(os.path.join(outdir,"picmus.jpg"))

