# File:       example_PICMUS.py
# Author:     Dongwoon Hyun (dongwoon.hyun@stanford.edu)
# Created on: 2020-03-12
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from cubdl.das_torch import DAS_PW
from datasets.PWDataLoaders import load_data, get_filelist
from cubdl.PixelGrid import make_pixel_grid

device = torch.device("cuda:0")
# Find all the CUBDL Task 1 Data
# Optional other data subsets:
#   "all", "phantom", "invivo", "postcubdl", "simulation", "task1"
filelist = get_filelist(data_type="task1")

for data_source in filelist:
    for acq in filelist[data_source]:
        P, xlims, zlims = load_data(data_source, acq)

        # Define pixel grid limits (assume y == 0)
        wvln = P.c / P.fc
        dx = wvln / 2.5
        dz = dx  # Use square pixels
        grid = make_pixel_grid(xlims, zlims, dx, dz)
        fnum = 1

        # Make data torch tensors
        x = (P.idata, P.qdata)

        # Make 75-angle image
        dasN = DAS_PW(P, grid, rxfnum=fnum)
        idasN, qdasN = dasN(x)
        idasN, qdasN = idasN.detach().cpu().numpy(), qdasN.detach().cpu().numpy()
        iqN = idasN + 1j * qdasN
        bimgN = 20 * np.log10(np.abs(iqN))  # Log-compress
        bimgN -= np.amax(bimgN)  # Normalize by max value

        # Make 1-angle image
        idx = len(P.angles) // 2  # Choose center angle
        das1 = DAS_PW(P, grid, idx, rxfnum=fnum)
        idas1, qdas1 = das1(x)
        idas1, qdas1 = idas1.detach().cpu().numpy(), qdas1.detach().cpu().numpy()
        iq1 = idas1 + 1j * qdas1
        bimg1 = 20 * np.log10(np.abs(iq1))  # Log-compress
        bimg1 -= np.amax(bimg1)  # Normalize by max value

        # Display images via matplotlib
        extent = [xlims[0] * 1e3, xlims[1] * 1e3, zlims[1] * 1e3, zlims[0] * 1e3]
        plt.clf()
        plt.subplot(121)
        plt.imshow(bimg1, vmin=-60, cmap="gray", extent=extent, origin="upper")
        plt.title("1 angle")
        plt.subplot(122)
        plt.imshow(bimgN, vmin=-60, cmap="gray", extent=extent, origin="upper")
        plt.title("%d angles" % len(P.angles))
        plt.suptitle("%s%03d (c = %d m/s)" % (data_source, acq, np.round(P.c)))
        plt.pause(0.01)

        # Check to make sure save directory exists, if not, create it
        savedir = os.path.join("datasets", "test_pw_images", data_source)
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        print(os.path.join("datasets", "test_pw_images", data_source, "%s%03d.jpg" % (data_source, acq)))
        plt.savefig(
            os.path.join(savedir, "%s%03d.jpg" % (data_source, acq))
        )
