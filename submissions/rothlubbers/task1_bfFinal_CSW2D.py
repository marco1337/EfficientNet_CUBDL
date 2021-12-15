import torch
import torch.nn as nn
from enum import Enum

import numpy as np
from torch.nn.functional import grid_sample
import copy

"""
This file contains a contribution to the 2020 challenge on ultrasound beamforming using deep learning (cubdl)
In order to use the mode from another file use the following code with the path to the supplied model pth-file
  
from cubdlSubmission.modelWrapper.task1_bfFinal_CSW2D import Task1_bf_final_CSW2D

model = Task1_bf_final_CSW2D()
model.load_state_dict(torch.load(modelPath))
model.eval()

"""


##################################################
### PREPROCESSING: BEAMFORMING + APODIZATION #####
##################################################


def delay_focus(grid, ele_pos):
    dist = torch.norm(grid - ele_pos.unsqueeze(0), dim=-1)
    return dist


def delay_plane(grid, angles):
    x = grid[:, 0].unsqueeze(0)
    z = grid[:, 2].unsqueeze(0)
    dist = x * torch.sin(angles) + z * torch.cos(angles)
    return dist


def _complex_rotate(I, Q, theta):
    Ir = I * torch.cos(theta) - Q * torch.sin(theta)
    Qr = Q * torch.cos(theta) + I * torch.sin(theta)
    return Ir, Qr


class TimeDelayBeamformer:
    def __init__(self, apodizationAlgorithms=None):

        if apodizationAlgorithms is None:
            apodizationAlgorithms = []
        assert isinstance(apodizationAlgorithms, list)
        self.apodizationAlgorithms = apodizationAlgorithms

    def __call__(
        self, P, grid, eventID, dtype=torch.float, device=torch.device("cuda:0")
    ):

        """ Forward pass for DAS_PW neural network.
    P           A PlaneWaveData object that describes the acquisition
    grid        A [ncols, nrows, 3] numpy array of the reconstruction grid
    ang_list    List of angle indices which should be used in the geometric beamforming
    """
        for apodAlgo in self.apodizationAlgorithms:
            apodAlgo.setUp(P, grid)

        rdel_list = [0]
        ang_list = [eventID]

        angles = torch.tensor(P.angles, dtype=dtype, device=device)
        ele_pos = torch.tensor(P.ele_pos, dtype=dtype, device=device)
        ele_list = torch.tensor(
            range(P.ele_pos.shape[0]), dtype=torch.long, device=device
        )
        fs = torch.tensor(P.fs, dtype=dtype, device=device)
        fdemod = torch.tensor(P.fdemod, dtype=dtype, device=device)
        c = torch.tensor(P.c, dtype=dtype, device=device)
        time_zero = torch.tensor(P.time_zero, dtype=dtype, device=device)

        idata = torch.tensor(P.idata, dtype=torch.float, device=torch.device("cuda:0"))
        qdata = torch.tensor(P.qdata, dtype=torch.float, device=torch.device("cuda:0"))

        img_shape = grid.shape[:-1]
        tgrid = torch.tensor(grid, dtype=dtype, device=device).reshape(-1, 3)

        nrdelays = len(rdel_list)
        nangles = len(ang_list)
        nelems = len(ele_list)
        npixels = tgrid.shape[0]
        txdel = torch.zeros((nangles, npixels), dtype=dtype, device=device)
        rxdel = torch.zeros((nelems, npixels), dtype=dtype, device=device)
        for i, tx in enumerate(ang_list):
            txdel[i] = delay_plane(tgrid, angles[tx])
            txdel[i] += time_zero[tx] * c
        for j, rx in enumerate(ele_list):
            rxdel[j] = delay_focus(tgrid, ele_pos[rx])
        txdel *= fs / c
        rxdel *= fs / c

        idas = torch.zeros(
            nangles, npixels, nrdelays, nelems, dtype=dtype, device=device
        )
        qdas = torch.zeros(
            nangles, npixels, nrdelays, nelems, dtype=dtype, device=device
        )
        for rdelidx, rawdelay in enumerate(rdel_list):
            for angleidx, (t, td) in enumerate(zip(ang_list, txdel)):
                for r, rd in zip(ele_list, rxdel):
                    iq = torch.stack((idata[t, r], qdata[t, r]), dim=0).view(
                        1, 2, 1, -1
                    )
                    delays = td + rd + rawdelay * fs
                    dgs = (delays.view(1, 1, -1, 1) * 2 + 1) / idata.shape[-1] - 1
                    dgs = torch.cat((dgs, 0 * dgs), axis=-1)
                    ifoc, qfoc = grid_sample(iq, dgs, align_corners=False).view(2, -1)
                    if fdemod != 0:
                        tshift = delays.view(-1) / fs - tgrid[:, 2] * 2 / c
                        theta = 2 * np.pi * fdemod * tshift
                        ifoc, qfoc = _complex_rotate(ifoc, qfoc, theta)

                    idas[angleidx, :, rdelidx, r] = ifoc
                    qdas[angleidx, :, rdelidx, r] = qfoc

        idas = idas.view((nangles,) + img_shape + (nelems, nrdelays))
        qdas = qdas.view((nangles,) + img_shape + (nelems, nrdelays))
        npidas = idas.cpu().numpy()
        npqdas = qdas.cpu().numpy()

        beamformed = np.stack((npidas, npqdas), axis=-1)
        for apodAlgo in self.apodizationAlgorithms:
            apod = apodAlgo(eventID)
            for i in range(nrdelays):
                beamformed = (
                    beamformed * apod[np.newaxis, :, :, :, np.newaxis, np.newaxis]
                )

        return beamformed


class ApodizationPWTransmit:
    def __init__(self, fnum=1):
        self.fnum = fnum
        self.jsonDict = self.__dict__.copy()

    def toDict(self):
        return self.jsonDict

    def setUp(self, data, grid):
        self.ele_pos = data.ele_pos
        self.angles = data.angles
        self.grid1D = grid.reshape(-1, 3)
        self.outputGridShape = grid.shape

    def __call__(self, eventID):
        xlims = (self.ele_pos[0, 0], self.ele_pos[-1, 0])
        txapo = self.apod_plane(self.angles[eventID], xlims=xlims)
        txapo = np.tile(txapo, (len(self.ele_pos), 1))
        apods3D = np.reshape(txapo, ((-1,) + self.outputGridShape[:2]))
        apods3DTranspose = np.transpose(apods3D, (1, 2, 0))
        return apods3DTranspose

    def apod_plane(self, angles, xlims):
        pix = self.grid1D[np.newaxis, :]
        ang = angles.reshape(-1, 1, 1)
        x_proj = pix[:, :, 0] - pix[:, :, 2] * np.tan(ang)
        mask = (x_proj >= xlims[0] * 1.2) & (x_proj <= xlims[1] * 1.2)
        apod = mask.astype(float)
        return apod


class ApodizationPWReceiveOrtho:
    def __init__(self, fnum=1):
        self.fnum = fnum
        self.jsonDict = self.__dict__.copy()

    def toDict(self):
        return self.jsonDict

    def setUp(self, data, grid):
        self.data = data
        self.grid1D = np.reshape(copy.deepcopy(grid), (-1, 3))
        self.ele_pos = data.ele_pos
        self.rxApo = self.apod_focus()
        apods3D = np.reshape(self.rxApo, ((-1,) + grid.shape[:2]))
        self.apods3DTranspose = np.transpose(apods3D, (1, 2, 0))

    def __call__(self, eventID):
        return self.apods3DTranspose

    def apod_focus(self, min_width=1e-3):
        ppos = self.grid1D[np.newaxis, :].copy()
        epos = np.reshape(copy.deepcopy(self.ele_pos), (-1, 1, 3))
        v = ppos - epos
        mask = np.abs(v[:, :, 2] / v[:, :, 0]) > self.fnum
        mask = mask | (np.abs(v[:, :, 0]) <= min_width)
        mask = mask | ((v[:, :, 0] >= min_width) & (ppos[:, :, 0] <= epos[0, 0, 0]))
        mask = mask | ((v[:, :, 0] <= -min_width) & (ppos[:, :, 0] >= epos[-1, 0, 0]))
        apod = mask.astype(float)
        return apod


# The following code is copyrighted
# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.

##################################################
### DEFINITIONS FOR DEEP LEARNING  ###############
##################################################


def ConvNd(n):
    return {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[n]


def BatchNormNd(n):
    return {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}[n]


class Activation(Enum):
    relu = 0
    leakyRelu = 1

    def get_module(self):
        if self == Activation.relu:
            return nn.ReLU()
        elif self == Activation.leakyRelu:
            return nn.LeakyReLU()


class ConvBatchNormActivate(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dim=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        activation=Activation.relu,
    ):
        super(self.__class__, self).__init__()

        if dim is None:
            dim = len(kernel_size)

        self.conv = ConvNd(dim)(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=True,
        )

        norm = BatchNormNd(dim)
        self.norm = norm(num_features=out_channels)

        self.activation = activation.get_module()

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)

        return x


##################################################
### MODEL FOR CHANNEL COMPOUNDING ALGORITHM ######
##################################################


class Task1_bf_final_CSW2D(nn.Module):
    def __init__(self):

        super(Task1_bf_final_CSW2D, self).__init__()

        levels = 4
        kernels = [(1, 65), (1, 15), (1, 15), (1, 3)]
        channels = [8, 8, 8, 1]
        activation = Activation.relu

        blocks = []
        curInChannel = 2
        for level in range(levels):
            curOutChannel = channels[level]
            kernel_size = kernels[level]
            paddingHeight = int((kernel_size[0] - 1) / 2)
            paddingWidth = int((kernel_size[1] - 1) / 2)
            convLayer = ConvBatchNormActivate(
                curInChannel,
                curOutChannel,
                kernel_size=kernel_size,
                activation=activation,
                padding=(paddingHeight, paddingWidth),
            )
            blocks.append(convLayer)
            curInChannel = curOutChannel

        self.convLayers = nn.ModuleList(blocks)
        self.beamformingAlgorithm = TimeDelayBeamformer(
            apodizationAlgorithms=[ApodizationPWReceiveOrtho(), ApodizationPWTransmit()]
        )

    def forward(self, P, grid, dtype=torch.float, device=torch.device("cpu:0")):

        norm = np.max(np.sqrt(P.idata ** 2 + P.qdata ** 2))
        P.idata /= norm
        P.qdata /= norm

        beamformed = self.beamformingAlgorithm(P, grid, 0)

        inputData = torch.from_numpy(beamformed).type(dtype).to(device)
        inputData = inputData.squeeze(-2)
        batchSize, height, width, numChannels, _ = inputData.shape
        inputPermuted = inputData.permute(0, 4, 3, 1, 2).contiguous()
        inputReordered = inputPermuted.view(batchSize, 2, numChannels, -1)
        inputMagSwap = torch.transpose(inputReordered, -1, -2)

        x = inputMagSwap
        for block in self.convLayers:
            x = block(x)

        x = x.squeeze(dim=1)
        x = torch.transpose(x, -1, -2)
        weights = x.view(batchSize, numChannels, height, width)
        singleWeight = torch.sum(weights, dim=1)

        eventCompoundedData = torch.mean(
            inputData * singleWeight[:, :, :, None, None], dim=-2
        )
        yPredMagnitude = torch.sqrt(torch.sum(eventCompoundedData ** 2, dim=-1))
        yPredCompressed = 20 * torch.log10(yPredMagnitude + 1e-20)
        yPred = yPredCompressed.squeeze(0)
        yPred = yPred - torch.max(yPred)

        return yPred

