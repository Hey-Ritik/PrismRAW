"""Guard MATLAB/HDF5 axis conventions and unsafe array deserialization."""

from pathlib import Path
import tempfile
import unittest
import numpy as np
from smartphone_isp.data.arrays import load_raw


class ArrayTests(unittest.TestCase):
    def test_npy_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.npy"
            array = np.arange(24).reshape(4, 6)
            np.save(path, array)
            np.testing.assert_array_equal(load_raw(path), array)

    def test_matlab_hdf5_axis_order(self):
        try:
            import h5py
        except ImportError:
            self.skipTest("h5py optional dependency not installed")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.mat"
            array = np.arange(24, dtype=np.float32).reshape(4, 6)
            # MATLAB 7.3 header triggers scipy's HDF5 detection.
            with h5py.File(path, "w", userblock_size=512) as archive:
                dataset = archive.create_dataset("x", data=array.T)
                dataset.attrs["MATLAB_class"] = np.bytes_("single")
            with path.open("r+b") as stream:
                stream.write(b"MATLAB 7.3 MAT-file".ljust(124, b" ") + b"\x00\x02IM")
            np.testing.assert_array_equal(load_raw(path, "x"), array)
