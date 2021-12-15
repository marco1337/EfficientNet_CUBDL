# File:       find_soundspeed.py
# Author:     Dongwoon Hyun (dongwoon.hyun@stanford.edu)
# Created on: 2020-03-12
import os
import pathlib
import torch
import matplotlib.pyplot as plt
import numpy as np
from cubdl.das_torch import DAS_PW
from cubdl.PixelGrid import make_pixel_grid
import hdf5storage
from datasets.PWDataLoaders import load_data, get_filelist


# Create a DAS_PW where the speed of sound is trainable
class DAS_PW_c(DAS_PW):
    def __init__(
        self,
        P,
        grid,
        ang_list=None,
        ele_list=None,
        rxfnum=0,
        dtype=torch.float,
        device=torch.device("cuda:0"),
    ):
        super().__init__(P, grid, ang_list, ele_list, rxfnum, dtype, device)
        self.c = torch.nn.Parameter(self.c)


filelist = get_filelist(data_type="phantom")
nepochs = 30

for data_source in filelist:
    for acq in filelist[data_source]:

        # Create the output directory if it does not already exist
        savedir = os.path.join("datasets", "soundspeed", "%s%03d" % (data_source, acq))
        pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)

        # If the output file already exists, skip
        fname = "%s/data.mat" % savedir
        if os.path.exists(fname):
            print("%s exists. Skipping..." % fname)
            continue
        else:
            print("Processing %s..." % fname)

        xlims = [-10e-3, 10e-3]
        zlims = [20e-3, 40e-3]

        P, _, _ = load_data(data_source, acq)
        P.c = 1540

        if data_source == "MYO":
            zlims = [15e-3, 35e-3]
        elif data_source == "EUT":
            zlims = [15e-3, 35e-3]
        elif data_source == "INS":
            if acq == 24:
                # Acqs 10, 11, 23, 24 are challenging to find a good speckle ROI
                xlims = [-12e-3, 12e-3]
                zlims = [14e-3, 24e-3]

        # Define pixel grid limits (assume y == 0)
        wvln = P.c / P.fc
        dx = wvln / 3
        dz = dx  # Use square pixels
        grid = make_pixel_grid(xlims, zlims, dx, dz)

        # Make DAS PW
        das = DAS_PW_c(P, grid, range(0, 75, 15))

        # Make data torch tensors
        x = (P.idata, P.qdata)

        # Make 75-angle image
        rms = np.mean(P.idata ** 2 + P.qdata ** 2)
        optimizer = torch.optim.Adam(das.parameters(), lr=10)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

        # Initialize empty lists to store B-mode images, sound speeds, image extents
        bimgs = []
        cs = []
        exts = []

        for epoch in range(nepochs + 1):
            optimizer.zero_grad()
            idas, qdas = das(x)
            loss = -(idas ** 2 + qdas ** 2).mean() / rms
            print(loss)
            print(list(das.parameters())[0])
            idas, qdas = idas.detach().cpu().numpy(), qdas.detach().cpu().numpy()
            iq = idas + 1j * qdas
            bimg = 20 * np.log10(np.abs(iq))  # Log-compress
            bimg -= np.amax(bimg)  # Normalize by max value

            # Update
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Display images
            extent = [xlims[0] * 1e3, xlims[1] * 1e3, zlims[1] * 1e3, zlims[0] * 1e3]
            plt.clf()
            plt.imshow(bimg, vmin=-60, cmap="gray", extent=extent, origin="upper")
            plt.title(
                "Epoch %d, sound speed = %dm/s"
                % (epoch, np.round(das.c.detach().cpu().numpy()))
            )
            plt.pause(0.1)

            # Save results
            bimgs.append(bimg)
            cs.append(das.c.detach().cpu().numpy())
            exts.append(extent)

        hdf5storage.savemat(
            "%s/data" % savedir,
            {"bimgs": np.stack(bimgs), "cs": np.stack(cs), "extent": exts},
        )

