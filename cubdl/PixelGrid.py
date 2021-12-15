# File:       PixelGrid.py
# Author:     Dongwoon Hyun (dongwoon.hyun@stanford.edu)
# Created on: 2020-04-03
import numpy as np

eps = 1e-10


def make_pixel_grid(xlims, zlims, dx, dz):
    """
    Generate a Cartesian pixel grid based on input parameters.
    The output has shape (nx, nz, 3).
    
    INPUTS
    xlims   Azimuthal limits of pixel grid ([xmin, xmax])
    zlims   Depth limits of pixel grid ([zmin, zmax])
    dx      Pixel spacing in azimuth
    dz      Pixel spacing in depth

    OUTPUTS
    grid    Pixel grid of size (nx, nz, 3)
    """
    x = np.arange(xlims[0], xlims[1] + eps, dx)
    z = np.arange(zlims[0], zlims[1] + eps, dz)
    zz, xx = np.meshgrid(z, x, indexing="ij")
    yy = 0 * xx
    grid = np.stack((xx, yy, zz), axis=-1)
    return grid


def make_foctx_grid(rlims, dr, oris, dirs):
    """
    Generate a focused pixel grid based on input parameters.
    To accommodate the multitude of ways of defining a focused transmit grid, we define
    pixel "rays" or "lines" according to their origins (oris) and directions (dirs).
    The position along the ray is defined by its limits (rlims) and spacing (dr).

    INPUTS
    rlims   Radial limits of pixel grid ([rmin, rmax])
    dr      Pixel spacing in radius
    oris    Origin of each ray in Cartesian coordinates (x, y, z) with shape (nrays, 3)
    dirs    Steering direction of each ray in azimuth, in units of radians (nrays, 2)

    OUTPUTS
    grid    Pixel grid of size (nr, nrays, 3) in Cartesian coordinates (x, y, z)
    """
    # Get focusing positions in rho-theta coordinates
    r = np.arange(rlims[0], rlims[1] + eps, dr)  # Depth rho
    t = dirs[:, 0]  # Use azimuthal angle theta (ignore elevation angle)
    tt, rr = np.meshgrid(t, r, indexing="ij")

    # Convert the focusing grid to Cartesian coordinates
    xx = rr * np.sin(tt) + oris[:, [0]]
    zz = rr * np.cos(tt) + oris[:, [2]]
    yy = 0 * xx
    grid = np.stack((xx, yy, zz), axis=-1)
    return grid
