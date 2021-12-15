# File:       test_metrics.py
# Author:     Dongwoon Hyun (dongwoon.hyun@stanford.edu)
# Created on: 2020-07-23
import torch
import os
import sys
sys.path.append(os.getcwd())
import tensorflow as tf
import numpy as np
from cubdl.das_torch import DAS_PW
from submissions.load_submission import load_submission
from scoring.measure_image import measure_image
from scoring.measure_point import measure_point
from scoring.measure_lesion import measure_lesion
from scoring.measure_speckle import measure_speckle
from scoring.measure_picmus import measure_picmus
from PIL import Image
from scipy.interpolate import RectBivariateSpline as interp2

device = torch.device("cuda:0")


def make_bmode_all(P, grid, fnum=1):
    # Make data torch tensors
    x = (P.idata, P.qdata)
    # Make inner ROI high quality image
    das = DAS_PW(P, grid, rxfnum=fnum)
    idas, qdas = das(x)
    idas, qdas = idas.detach().cpu().numpy(), qdas.detach().cpu().numpy()
    iq = idas + 1j * qdas
    bimg = np.abs(iq)
    bimg /= np.amax(bimg)
    return bimg


def make_bmode_one(P, grid, fnum=1):
    # Make data torch tensors
    x = (P.idata, P.qdata)
    # Make 1-angle image
    idx = len(P.angles) // 2  # Choose center angle
    window_size = 32
    nelems = 128
    iq = torch.zeros(2, window_size, nelems)
    for ele in range(nelems):
        das = DAS_PW(P, grid, idx, rxfnum=fnum)
        idas, qdas = das(x)
    idas, qdas = idas.detach().cpu().numpy(), qdas.detach().cpu().numpy()
    iq = idas + 1j * qdas
    bimg = np.abs(iq)
    bimg /= np.amax(bimg)
    return bimg


def make_bmode_goudarzi(P, grid, fnum=1):
    from submissions.goudarzi.das_torch import DAS_PW as DAS_PW1

    # Make data torch tensors
    x = (P.idata, P.qdata)
    # Make 1-angle image
    idx = len(P.angles) // 2  # Choose center angle
    das = DAS_PW1(P, grid, idx)
    idas, qdas, ch1, apod = das(x)
    idas = idas.detach().cpu().numpy()
    qdas = qdas.detach().cpu().numpy()
    ch1 = ch1.detach().cpu().numpy()
    apod = apod.detach().cpu().numpy()
    iq = idas + 1j * qdas
    bimg = 20 * np.log10(np.abs(iq))
    bimg -= np.amax(bimg)

    ch2 = np.resize(ch1, [grid.shape[0], grid.shape[1], 2, P.idata.shape[1]])

    input1 = np.zeros([np.size(ch1, 0), 2, 32, P.idata.shape[1]], dtype=np.float32)
    for i in range(np.size(ch1, 0)):
        idxc = i % grid.shape[1]
        idxr = i // grid.shape[1]
        if idxr - 14 > 0 and idxr + 16 < grid.shape[0] - 1:
            input1[i, :, :, :] = np.transpose(
                ch2[idxr - 15 : idxr + 17, idxc, :, :], (1, 0, 2)
            )
        elif idxr - 14 <= 0:
            input1[i, :, 15 - idxr : 32, :] = np.transpose(
                ch2[0 : idxr + 17, idxc, :, :], (1, 0, 2)
            )
        else:
            input1[i, :, 0 : 16 + grid.shape[0] - 1 - idxr, :] = np.transpose(
                ch2[idxr - 15 : grid.shape[0], idxc, :, :], (1, 0, 2)
            )

    torch.cuda.empty_cache()
    model = load_submission("goudarzi").to(device)
    model.eval()

    # ch2 = np.resize(ch, [508, 387, 2, 128])
    output1 = np.zeros([np.size(input1, 0), 2], dtype=np.float32)
    batch_size = grid.shape[0] + 8
    count = 0

    for count in range(int(np.size(input1, 0) / batch_size)):
        imgs1 = torch.tensor(
            input1[count * batch_size : (count + 1) * batch_size],
            dtype=torch.float,
            device="cuda:0",
        )
        outputt = model(imgs1)
        output1[count * batch_size : (count + 1) * batch_size] = (
            outputt.cpu().detach().numpy()
        )
        del outputt, imgs1
    iq = output1[:, 0] + 1j * output1[:, 1]
    iq = iq.reshape(grid.shape[:2])
    bimg = 20 * np.log10(np.abs(iq))  # Log-compress
    bimg -= np.amax(bimg)  # Normalize by max value
    bimg = 10 ** (bimg / 20)
    return bimg


def make_bmode_rothlubbers(P, grid):
    das = load_submission("rothlubbers").to(device)
    das = das.eval()
    bimg = das(P, grid, device=device).detach().cpu().numpy()
    bimg = 10 ** (bimg / 20)
    return bimg


if __name__ == "__main__":
    submissions = [
        [make_bmode_all, "ground_truth", False],
        [make_bmode_one, "single_plane", True],
        [make_bmode_rothlubbers, "rothlubbers", True],
        [make_bmode_goudarzi, "goudarzi", True],
    ]

    for func, name, flag in submissions:
        # measure_picmus(func, name, flag)
        measure_image(func, name, flag)
        measure_lesion(func, name, flag)
        measure_speckle(func, name, flag)
        measure_point(func, name, flag)
