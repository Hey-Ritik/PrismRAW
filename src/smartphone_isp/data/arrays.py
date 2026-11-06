"""Array loading with explicit variable selection and no pickle deserialization."""

from pathlib import Path
import numpy as np


def load_raw(path, key=None):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        mapped = np.load(path, allow_pickle=False, mmap_mode="r")
        if not isinstance(mapped, np.memmap):
            if hasattr(mapped, "close"):
                mapped.close()
            raise ValueError("NPY input must contain one plain numeric sensor array")
        try:
            if mapped.ndim != 2 or mapped.size > 24_000_000:
                raise ValueError("NPY RAW must be 2D and no larger than 24 megapixels")
            return np.array(mapped, copy=True)
        finally:
            # Windows will not delete temporary uploads while a mapping is open.
            mapped._mmap.close()
    if path.suffix.lower() == ".npz":
        if not key:
            raise ValueError("NPZ input requires --key")
        import zipfile
        try:
            with zipfile.ZipFile(path) as archive:
                if sum(entry.file_size for entry in archive.infolist()) > 192 * 1024 * 1024:
                    raise ValueError("NPZ exceeds the 192 MB decoded-size limit")
        except zipfile.BadZipFile as exc:
            raise ValueError("NPZ input is not a valid array archive") from exc
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
