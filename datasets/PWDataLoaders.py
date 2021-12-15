import os
import h5py
import numpy as np
from glob import glob
from scipy.signal import hilbert, convolve
from cubdl.PlaneWaveData import PlaneWaveData


def load_data(data_source, acq):
    datadir1 = "1_CUBDL_Task1_Data"
    datadir3 = os.path.join("3_Additional_CUBDL_Data", "Plane_Wave_Data")
    if data_source == "MYO":
        # Choose correct path for data
        if acq in [1, 2, 3, 4, 5]:
            database_path = os.path.join("datasets", datadir1)
        elif acq == 6:
            database_path = os.path.join("datasets", datadir3)
        else:
            raise ValueError("MYO%03d is not a valid plane wave acquistion." % acq)
        # Load Mayo Clinic dataset
        P = MYOData(database_path, acq)
        xlims = [P.ele_pos[0, 0], P.ele_pos[-1, 0]]
        zlims = [5e-3, 55e-3]
    elif data_source == "UFL":
        # Choose correct path for data
        if acq in [1, 2, 4, 5]:
            database_path = os.path.join("datasets", datadir1)
        elif acq == 3:
            database_path = os.path.join("datasets", datadir3)
        else:
            raise ValueError("UFL%03d is not a valid plane wave acquistion." % acq)
        # Load UNIFI dataset
        P = UFLData(database_path, acq)
        xlims = [P.ele_pos[0, 0], P.ele_pos[-1, 0]]
        zlims = [10e-3, 50e-3]
    elif data_source == "EUT":
        # Choose correct path for data
        if acq in [3, 6]:
            database_path = os.path.join("datasets", datadir1)
        elif acq in [1, 2, 4, 5]:
            database_path = os.path.join("datasets", datadir3)
        else:
            raise ValueError("UFL%03d is not a valid plane wave acquistion." % acq)
        # Load Eindhoven dataset
        P = EUTData(database_path, acq)
        xlims = [P.ele_pos[0, 0], P.ele_pos[-1, 0]]
        zlims = [10e-3, 80e-3]
    elif data_source == "INS":
        # Choose correct path for data
        if acq in [4, 6, 8, 15, 16, 19, 21]:
            database_path = os.path.join("datasets", datadir1)
        elif acq >= 1 and acq <= 26:
            database_path = os.path.join("datasets", datadir3)
        else:
            raise ValueError("INS%03d is not a valid plane wave acquistion." % acq)
        # Load INSERM dataset
        P = INSData(database_path, acq)
        xlims = [P.ele_pos[0, 0], P.ele_pos[-1, 0]]
        zlims = [10e-3, 60e-3]
        if acq >= 13:
            zlims = [10e-3, 50e-3]
    elif data_source == "OSL":
        # Choose correct path for data
        if acq in [2, 3, 4, 5, 6]:
            database_path = os.path.join("datasets", datadir3)
        elif acq in [7]:
            database_path = os.path.join("datasets", datadir1)
        elif acq in [10]:
            database_path = os.path.join("datasets", datadir1, "OSL010")
        else:
            raise ValueError("OSL%03d is not a valid plane wave acquistion." % acq)
        # Load University of Oslo dataset
        P = OSLData(database_path, acq)
        xlims = [P.ele_pos[0, 0], P.ele_pos[-1, 0]]
        zlims = [10e-3, 65e-3]
        if acq == 10:
            zlims = [5e-3, 50e-3]
    elif data_source == "TSH":
        # Choose correct path for data
        if acq in [2]:
            database_path = os.path.join("datasets", datadir1, "TSH002")
        elif acq >= 3 and acq <= 501:
            database_path = os.path.join("datasets", datadir3, "TSH")
        else:
            raise ValueError("OSL%03d is not a valid plane wave acquistion." % acq)
        # Load Tsinghua University dataset
        P = TSHData(database_path, acq)
        xlims = [P.ele_pos[0, 0], P.ele_pos[-1, 0]]
        zlims = [10e-3, 45e-3]
    elif data_source == "JHU":
        # Load Johns Hopkins University dataset
        P = JHUData(database_path, acq)
        xlims = [P.ele_pos[0, 0], P.ele_pos[-1, 0]]
        zlims = [0e-3, 30e-3]
    else:
        raise NotImplementedError
    return P, xlims, zlims


class TSHData(PlaneWaveData):
    """ Load data from Tsinghua University. """

    def __init__(self, database_path, acq):
        # Make sure the selected dataset is valid
        moniker = "TSH{:03d}".format(acq) + "*.hdf5"
        fname = [
            y for x in os.walk(database_path) for y in glob(os.path.join(x[0], moniker))
        ]
        assert fname, "File not found."

        # Load dataset
        f = h5py.File(fname[0], "r")

        # Get data
        self.angles = np.array(f["angles"])
        self.idata = np.array(f["channel_data"], dtype="float32")
        self.idata = np.reshape(self.idata, (128, len(self.angles), -1))
        self.idata = np.transpose(self.idata, (1, 0, 2))
        self.qdata = np.imag(hilbert(self.idata, axis=-1))
        self.fc = np.array(f["modulation_frequency"]).item()
        self.fs = np.array(f["sampling_frequency"]).item()
        self.c = 1540  # np.array(f["sound_speed"]).item()
        self.time_zero = np.zeros((len(self.angles),), dtype="float32")
        self.fdemod = 0

        # Make the element positions based on L11-4v geometry
        pitch = 0.3e-3
        nelems = self.idata.shape[1]
        xpos = np.arange(nelems) * pitch
        xpos -= np.mean(xpos)
        self.ele_pos = np.stack([xpos, 0 * xpos, 0 * xpos], axis=1)

        # For this dataset, time zero is the center point
        for i, a in enumerate(self.angles):
            self.time_zero[i] = self.ele_pos[-1, 0] * np.abs(np.sin(a)) / self.c

        # Validate that all information is properly included
        super().validate()


class MYOData(PlaneWaveData):
    """ Load data from Mayo Clinic. """

    def __init__(self, database_path, acq):
        # Make sure the selected dataset is valid
        moniker = "MYO{:03d}".format(acq) + "*.hdf5"
        fname = [
            y for x in os.walk(database_path) for y in glob(os.path.join(x[0], moniker))
        ]
        assert fname, "File not found."

        # Load dataset
        f = h5py.File(fname[0], "r")

        # Phantom-specific parameters
        if acq == 1:
            sound_speed = 1580
        elif acq == 2:
            sound_speed = 1583
        elif acq == 3:
            sound_speed = 1578
        elif acq == 4:
            sound_speed = 1572
        elif acq == 5:
            sound_speed = 1562
        else:
            sound_speed = 1581

        # Get data
        self.idata = np.array(f["channel_data"], dtype="float32")
        self.qdata = np.imag(hilbert(self.idata, axis=-1))
        self.angles = np.array(f["angles"])
        self.fc = np.array(f["modulation_frequency"]).item()
        self.fs = np.array(f["sampling_frequency"]).item()
        self.c = sound_speed  # np.array(f["sound_speed"]).item()
        self.time_zero = np.zeros((len(self.angles),), dtype="float32")
        self.fdemod = 0

        # Make the element positions based on L11-4v geometry
        pitch = 0.3e-3
        nelems = self.idata.shape[1]
        xpos = np.arange(nelems) * pitch
        xpos -= np.mean(xpos)
        self.ele_pos = np.stack([xpos, 0 * xpos, 0 * xpos], axis=1)

        # For this dataset, time zero is the center point
        for i, a in enumerate(self.angles):
            self.time_zero[i] = self.ele_pos[-1, 0] * np.abs(np.sin(a)) / self.c

        # Validate that all information is properly included
        super().validate()


class UFLData(PlaneWaveData):
    """ Load data from UNIFI. """

    def __init__(self, database_path, acq):
        moniker = "UFL{:03d}".format(acq) + "*.hdf5"
        fname = [
            y for x in os.walk(database_path) for y in glob(os.path.join(x[0], moniker))
        ]
        assert fname, "File not found."

        # Load dataset
        f = h5py.File(fname[0], "r")

        # Phantom-specific parameters
        if acq == 1:
            sound_speed = 1526
        elif acq == 2 or acq == 4 or acq == 5:
            sound_speed = 1523
        else:
            sound_speed = 1525

        # Get data
        self.idata = np.array(f["channel_data"], dtype="float32")
        self.qdata = np.imag(hilbert(self.idata, axis=-1))
        self.angles = np.array(f["angles"]) * np.pi / 180
        self.fc = np.array(f["modulation_frequency"]).item()
        self.fs = np.array(f["channel_data_sampling_frequency"]).item()
        self.c = sound_speed  # np.array(f["sound_speed"]).item()
        self.time_zero = -1 * np.array(f["channel_data_t0"], dtype="float32")
        self.fdemod = self.fc

        # Make the element positions based on LA533 geometry
        pitch = 0.245e-3
        nelems = self.idata.shape[1]
        xpos = np.arange(nelems) * pitch
        xpos -= np.mean(xpos)
        self.ele_pos = np.stack([xpos, 0 * xpos, 0 * xpos], axis=1)

        # Make sure that time_zero is an array of size [nangles]
        if self.time_zero.size == 1:
            self.time_zero = np.ones_like(self.angles) * self.time_zero

        # Demodulate data and low-pass filter
        data = self.idata + 1j * self.qdata
        phase = np.reshape(np.arange(self.idata.shape[2], dtype="float"), (1, 1, -1))
        phase *= self.fdemod / self.fs
        data *= np.exp(-2j * np.pi * phase)
        dsfactor = int(np.floor(self.fs / self.fc))
        kernel = np.ones((1, 1, dsfactor), dtype="float") / dsfactor
        data = convolve(data, kernel, "same")
        data = data[:, :, ::dsfactor]
        self.fs /= dsfactor

        self.idata = np.real(data)
        self.qdata = np.imag(data)

        # Validate that all information is properly included
        super().validate()


class EUTData(PlaneWaveData):
    """ Load data from TU/e. """

    def __init__(self, database_path, acq):

        # Make sure the selected dataset is valid
        moniker = "EUT{:03d}".format(acq) + "*.hdf5"
        fname = [
            y for x in os.walk(database_path) for y in glob(os.path.join(x[0], moniker))
        ]
        assert fname, "File not found."

        # Load dataset
        f = h5py.File(fname[0], "r")

        # Phantom-specific parameters
        if acq == 1:
            sound_speed = 1603
        elif acq == 2:
            sound_speed = 1618
        elif acq == 3:
            sound_speed = 1607
        elif acq == 4:
            sound_speed = 1614
        elif acq == 5:
            sound_speed = 1495
        else:
            sound_speed = 1479

        # Get data
        self.idata = np.array(f["channel_data"], dtype="float32")
        self.qdata = np.imag(hilbert(self.idata, axis=-1))
        self.angles = np.array(f["transmit_direction"])[:, 0]
        self.fc = np.array(f["modulation_frequency"]).item()
        self.fs = np.array(f["sampling_frequency"]).item()
        self.c = sound_speed
        self.time_zero = np.array(f["start_time"], dtype="float32")[0]
        self.fdemod = 0
        self.ele_pos = np.array(f["element_positions"], dtype="float32").T
        self.ele_pos[:, 0] -= np.mean(self.ele_pos[:, 0])

        # For this dataset, time zero is the center point
        for i, a in enumerate(self.angles):
            self.time_zero[i] = self.ele_pos[-1, 0] * np.abs(np.sin(a)) / self.c

        # Seems to be some offset
        self.time_zero += 10 / self.fc

        # Validate that all information is properly included
        super().validate()


class INSData(PlaneWaveData):
    """ Load data from INSERM. """

    def __init__(self, database_path, acq):

        # Make sure the selected dataset is valid
        moniker = "INS{:03d}".format(acq) + "*.hdf5"
        fname = [
            y for x in os.walk(database_path) for y in glob(os.path.join(x[0], moniker))
        ]
        assert fname, "File not found."

        # Load dataset
        f = h5py.File(fname[0], "r")

        # Phantom-specific parameters
        if acq == 1:
            sound_speed = 1521
        elif acq == 2:
            sound_speed = 1517
        elif acq == 3:
            sound_speed = 1506
        elif acq == 4:
            sound_speed = 1501
        elif acq == 5:
            sound_speed = 1506
        elif acq == 6:
            sound_speed = 1509
        elif acq == 7:
            sound_speed = 1490
        elif acq == 8:
            sound_speed = 1504
        elif acq == 9:
            sound_speed = 1473
        elif acq == 10:
            sound_speed = 1502
        elif acq == 11:
            sound_speed = 1511
        elif acq == 12:
            sound_speed = 1535
        elif acq == 13:
            sound_speed = 1453
        elif acq == 14:
            sound_speed = 1542
        elif acq == 15:
            sound_speed = 1539
        elif acq == 16:
            sound_speed = 1466
        elif acq == 17:
            sound_speed = 1462
        elif acq == 18:
            sound_speed = 1479
        elif acq == 19:
            sound_speed = 1469
        elif acq == 20:
            sound_speed = 1464
        elif acq == 21:
            sound_speed = 1508
        elif acq == 22:
            sound_speed = 1558
        elif acq == 23:
            sound_speed = 1463
        elif acq == 24:
            sound_speed = 1547
        elif acq == 25:
            sound_speed = 1477
        elif acq == 26:
            sound_speed = 1497
        else:
            sound_speed = 1540

        # Get data
        self.idata = np.array(f["channel_data"], dtype="float32")
        self.qdata = np.imag(hilbert(self.idata, axis=-1))
        self.angles = np.linspace(-16, 16, self.idata.shape[0]) * np.pi / 180
        self.fc = np.array(f["modulation_frequency"]).item()
        self.fs = np.array(f["sampling_frequency"]).item()
        self.c = sound_speed  # np.array(f["sound_speed"]).item()
        self.time_zero = -1 * np.array(f["start_time"], dtype="float32")[0]
        self.fdemod = 0
        self.ele_pos = np.array(f["element_positions"], dtype="float32").T
        self.ele_pos[:, 0] -= np.mean(self.ele_pos[:, 0])

        # For this dataset, time zero is the center point
        for i, a in enumerate(self.angles):
            self.time_zero[i] += self.ele_pos[-1, 0] * np.abs(np.sin(a)) / self.c

        # Validate that all information is properly included
        super().validate()


class OSLData(PlaneWaveData):
    """ Load data from University of Oslo. """

    def __init__(self, database_path, acq):

        # Make sure the selected dataset is valid
        moniker = "OSL{:03d}".format(acq) + ".hdf5"
        fname = [
            y for x in os.walk(database_path) for y in glob(os.path.join(x[0], moniker))
        ]
        assert fname, "File not found."
        assert acq in [2, 3, 4, 5, 6, 7, 10], "Focused Data. Use FTDataLoaders"

        # Load dataset
        f = h5py.File(fname[0], "r")

        # Phantom-specific parameters
        if acq == 2:
            sound_speed = 1536
        elif acq == 3:
            sound_speed = 1543
        elif acq == 4:
            sound_speed = 1538
        elif acq == 5:
            sound_speed = 1539
        elif acq == 6:
            sound_speed = 1541
        elif acq == 7:
            sound_speed = 1540
        else:
            sound_speed = 1540

        # Get data
        self.idata = np.array(f["channel_data"], dtype="float32")
        self.qdata = np.imag(hilbert(self.idata, axis=-1))
        self.angles = np.array(f["transmit_direction"][0], dtype="float32")
        self.fc = np.array(f["modulation_frequency"]).item()
        self.fs = np.array(f["sampling_frequency"]).item()
        self.c = sound_speed  # np.array(f["sound_speed"]).item()
        self.time_zero = -1 * np.array(f["start_time"], dtype="float32")[0]
        self.fdemod = 0
        self.ele_pos = np.array(f["element_positions"], dtype="float32").T
        self.ele_pos[:, 0] -= np.mean(self.ele_pos[:, 0])

        # Validate that all information is properly included
        super().validate()


class JHUData(PlaneWaveData):
    """ Load data from University of Oslo. """

    def __init__(self, database_path, acq):

        # Make sure the selected dataset is valid
        moniker = "JHU{:03d}".format(acq) + ".hdf5"
        fname = [
            y for x in os.walk(database_path) for y in glob(os.path.join(x[0], moniker))
        ]
        assert fname, "File not found."
        assert acq in list(range(24, 35)), "Focused Data. Use FTDataLoaders"

        # Load dataset
        f = h5py.File(fname[0], "r")

        # Get data
        self.idata = np.array(f["channel_data"], dtype="float32")
        self.qdata = np.imag(hilbert(self.idata, axis=-1))
        self.angles = np.array(f["angles"])
        self.fc = np.array(f["modulation_frequency"]).item()
        self.fs = np.array(f["sampling_frequency"]).item()
        self.c = np.array(f["sound_speed"]).item()
        self.time_zero = -1 * np.array(f["time_zero"], dtype="float32")
        self.fdemod = 0

        xpos = np.array(f["element_positions"], dtype="float32").T
        self.ele_pos = np.stack([xpos, 0 * xpos, 0 * xpos], axis=1)
        self.zlims = np.array([0e-3, self.idata.shape[2] * self.c / self.fs / 2])
        self.xlims = np.array([self.ele_pos[0, 0], self.ele_pos[-1, 0]])

        # For this dataset, time zero is the center point
        # self.time_zero = np.zeros((len(self.angles),), dtype="float32")
        # for i, a in enumerate(self.angles):
        #     self.time_zero[i] = self.ele_pos[-1, 0] * np.abs(np.sin(a)) / self.c

        # Seems to be some offset
        # self.time_zero -= 10 / self.fc

        # Validate that all information is properly included
        super().validate()


def get_filelist(data_type="task1"):
    # Get the filenames of the plane wave data
    if data_type == "all":
        print("Warning: TSH has 500 files. This will take a while.")
        filelist = {
            "OSL": [2, 3, 4, 5, 6, 7, 10],
            "MYO": [1, 2, 3, 4, 5, 6],
            "UFL": [1, 2, 3, 4, 5],
            "EUT": [1, 2, 3, 4, 5, 6],
            "INS": list(range(1, 27)),
            "JHU": list(range(24, 35)),
            "TSH": list(range(2, 502)),
        }
    elif data_type == "phantom":
        filelist = {
            "OSL": [2, 3, 4, 5, 6, 7],
            "MYO": [1, 2, 3, 4, 5, 6],
            "UFL": [1, 2, 3, 4, 5],
            "EUT": [1, 2, 3, 4, 5, 6],
            "INS": list(range(1, 27)),
        }
    elif data_type == "postcubdl":
        filelist = {"JHU": list(range(24, 35))}
    elif data_type == "invivo":
        print("Warning: TSH has 500 files. This will take a while.")
        filelist = {"JHU": list(range(24, 35)), "TSH": list(range(2, 502))}
    elif data_type == "simulation":
        filelist = {"OSL": [10]}
    elif data_type == "task1":
        filelist = {
            "OSL": [7, 10],
            "TSH": [2],
            "MYO": [1, 2, 3, 4, 5],
            "UFL": [1, 2, 4, 5],
            "EUT": [3, 6],
            "INS": [4, 6, 8, 15, 16, 19, 21],
        }
    else:
        filelist = {
            "OSL": [7, 10],
            "TSH": [2],
            "MYO": [1, 2, 3, 4, 5],
            "UFL": [1, 2, 4, 5],
            "EUT": [3, 6],
            "INS": [4, 6, 8, 15, 16, 19, 21],
        }

    return filelist
