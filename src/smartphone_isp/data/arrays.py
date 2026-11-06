"""Array loading with explicit variable selection and no pickle deserialization."""

from pathlib import Path
import numpy as np


def load_raw(path, key=None):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    if path.suffix.lower() == ".npz":
        if not key:
            raise ValueError("NPZ input requires --key")
        with np.load(path, allow_pickle=False) as archive:
            return archive[key]
    if path.suffix.lower() == ".mat":
        if not key:
            raise ValueError("MAT input requires --key; inspect variables before choosing")
        from scipy.io import loadmat
        try:
            return loadmat(path, variable_names=[key])[key]
        except NotImplementedError:
            import h5py
            with h5py.File(path, "r") as archive:
                dataset = archive[key]
                if "MATLAB_class" not in dataset.attrs or dataset.ndim != 2:
                    raise ValueError("Expected a MATLAB v7.3 2D sensor array")
                # MATLAB HDF5 storage reverses dimension order.
                return dataset[:].T
    raise ValueError("Supported RAW containers: .npy, .npz, pre-v7.3 .mat; DNG pending")
