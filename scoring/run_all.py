import numpy as np
from scoring.make_rois_point import rois_point
from scoring.make_rois_image import rois_image
from scoring.make_rois_lesion import rois_lesion
from scoring.make_rois_speckle import rois_speckle
from scoring.make_rois_additional import rois_additional
from cubdl.das_torch import DAS_PW
from scoring.measure_picmus import measure_picmus
from scoring.measure_point import measure_point
from scoring.measure_image import measure_image
from scoring.measure_lesion import measure_lesion
from scoring.measure_speckle import measure_speckle
from scoring.measure_additional import measure_additional

# Make ROIs
for idx in range(4):
    rois_point(idx)
for idx in range(8):
    rois_lesion(idx)
for idx in range(6):
    rois_speckle(idx)
for idx in range(12):
    rois_image(idx)
for idx in range(11):
    rois_additional(idx)


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
    # idx = len(P.angles) // 2  # Choose center angle
    idx = np.where(P.angles == 0)
    das = DAS_PW(P, grid, idx, rxfnum=fnum)
    idas, qdas = das(x)
    idas, qdas = idas.detach().cpu().numpy(), qdas.detach().cpu().numpy()
    iq = idas + 1j * qdas
    bimg = np.abs(iq)
    bimg /= np.amax(bimg)
    return bimg


# Make results for the ground truth (all plane wave angles)
measure_picmus(make_bmode_all, "ground_truth")
measure_point(make_bmode_all, "ground_truth")
measure_image(make_bmode_all, "ground_truth")
measure_lesion(make_bmode_all, "ground_truth")
measure_speckle(make_bmode_all, "ground_truth")
measure_additional(make_bmode_all, "ground_truth")

# Also make results for the single plane wave case
measure_picmus(make_bmode_one, "single_plane", center_angle=True)
measure_point(make_bmode_one, "single_plane", center_angle=True)
measure_image(make_bmode_one, "single_plane", center_angle=True)
measure_lesion(make_bmode_one, "single_plane", center_angle=True)
measure_speckle(make_bmode_one, "single_plane", center_angle=True)
measure_additional(make_bmode_one, "single_plane", center_angle=True)
