# File:       test_FT.py
# Author:     Dongwoon Hyun (dongwoon.hyun@stanford.edu)
# Created on: 2020-03-12
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from cubdl.das_torch import DAS_FT
from datasets.FTDataLoaders import OSLData, JHUData, get_filelist
from cubdl.PixelGrid import make_foctx_grid, make_pixel_grid
from scipy.interpolate import griddata


# Find all the Focused Data
# Optional other data subsets:
#   "all", "phantom", "invivo"
filelist = get_filelist(data_type="all")

for data_source in filelist:
    for acq in filelist[data_source]:
        database_path = os.path.join("datasets", "data")
        if data_source == "OSL":
            # Load OSL dataset
            F = OSLData(database_path, acq)
            rmax = 38e-3 if (acq == 8 or acq ==9) else 125e-3
            scan_convert = False if (acq == 8 or acq ==9) else True
            drange = 60
        elif data_source == "JHU":
            # Load JHU dataset
            F = JHUData(database_path, acq)
            if acq <= 2:
                rmax = 40e-3
                scan_convert = False
            else:
                rmax = 60e-3
                scan_convert = True
            drange = 50
        else:
            raise NotImplementedError

        # Define pixel grid limits (assume y == 0)
        wvln = F.c / F.fc
        dr = wvln / 4
        rlims = [0, rmax]
        grid = make_foctx_grid(rlims, dr, F.tx_ori, F.tx_dir)
        fnum = 0

        # Make data torch tensors
        idata = torch.tensor(F.idata, dtype=torch.float, device=torch.device("cuda:0"))
        qdata = torch.tensor(F.qdata, dtype=torch.float, device=torch.device("cuda:0"))
        x = (idata, qdata)

        # Make focused transmit
        das = DAS_FT(F, grid, rxfnum=fnum)
        idas, qdas = das(x)
        idas, qdas = idas.detach().cpu().numpy(), qdas.detach().cpu().numpy()
        iq = idas + 1j * qdas
        bimg = np.abs(iq).T

        # Scan convert if necessary
        if scan_convert:
            xlims = rlims[1] * np.array([-0.7, 0.7])
            zlims = rlims[1] * np.array([0, 1])
            img_grid = make_pixel_grid(xlims, zlims, wvln / 2, wvln / 2)
            grid = np.transpose(grid, (1, 0, 2))
            g1 = np.stack((grid[:, :, 2], grid[:, :, 0]), -1).reshape(-1, 2)
            g2 = np.stack((img_grid[:, :, 2], img_grid[:, :, 0]), -1).reshape(-1, 2)
            bsc = griddata(g1, bimg.reshape(-1), g2, "linear", 1e-10)
            bimg = np.reshape(bsc, img_grid.shape[:2])
            grid = img_grid.transpose(1, 0, 2)

        bimg = 20 * np.log10(bimg)  # Log-compress
        bimg -= np.amax(bimg)  # Normalize by max value

        # Display images via matplotlib
        extent = [grid[0, 0, 0], grid[-1, 0, 0], grid[0, -1, 2], grid[0, 0, 2]]
        extent = np.array(extent) * 1e3  # Convert to mm
        plt.clf()
        plt.imshow(bimg, vmin=-drange, cmap="gray", extent=extent, origin="upper")
        plt.title("%s%03d (c = %d m/s)" % (data_source, acq, np.round(F.c)))
        plt.colorbar()
        plt.pause(0.01)

        # Check to make sure save directory exists, if not, create it
        savedir = os.path.join("datasets", "test_ft_images", data_source)
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        plt.savefig(
            os.path.join(savedir, "%s%03d.jpg" % (data_source, acq))
        )
