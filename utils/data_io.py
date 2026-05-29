import os

import numpy as np
import pandas as pd
import yaml


def get_data_config(config_path="data_config.yaml"):
    """Load data configuration from a YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_file_path(config, version = "version", file_prefix = "file_prefix"):
    """Construct the file path for the dataset based on the configuration."""
    root = config.get("root", "../data")
    version = config.get(version, "2026-05-15-15h")
    file_prefix = config.get(file_prefix, "df_all_neurons_")
    file_path = os.path.abspath(os.path.join(root, f"{file_prefix}{version}.parquet"))
    return file_path



# def get_file_path_ribbons(config):
#     """Construct the file path for the dataset based on the configuration."""
#     root = config.get("root", "../data")
#     version = config.get("version_rb", "2026-05-28-11h")
#     file_prefix = config.get("file_prefix_rb", "df_ribbons_")
#     file_path = os.path.abspath(os.path.join(root, f"{file_prefix}{version}.parquet"))
#     return file_path


def serialize_numpy_arrays(df):
    """Serialize multi-dimensional numpy arrays in dataframe columns for storage in parquet."""
    df_serialized = df.copy()

    for col in df_serialized.columns:
        if df_serialized[col].dtype == "object":
            # Check if this column contains numpy arrays
            sample_non_null = df_serialized[col].dropna()
            if len(sample_non_null) > 0:
                first_val = sample_non_null.iloc[0]
                if isinstance(first_val, np.ndarray):
                    print(f"Serializing numpy arrays in column: {col}")
                    # Convert numpy arrays to nested lists - needed to save N-dimensional arrays to parquet
                    df_serialized[col] = [x.tolist() if isinstance(x, np.ndarray) else x for x in df_serialized[col]]
                elif isinstance(first_val, list) and any(isinstance(v, np.ndarray) for v in first_val):
                    print(f"Serializing list-of-arrays in column: {col}")
                    # Lists of numpy arrays with mixed dtypes (e.g. float32 vs float64) cause ArrowInvalid.
                    # Convert each inner array to a plain Python list so PyArrow sees a uniform type.
                    df_serialized[col] = [
                        [x.tolist() if isinstance(x, np.ndarray) else x for x in val]
                        if isinstance(val, list) else val
                        for val in df_serialized[col]
                    ]

    return df_serialized


def restore_numpy_arrays(df):
    """Automatically detect and restore numpy arrays from nested structures."""
    df_restored = df.copy()

    def is_nested_array_structure(val):
        """Check if a value is a nested array structure that should be converted to numpy array."""

        # Check for numpy object arrays containing other arrays (from parquet loading)
        if isinstance(val, np.ndarray) and val.dtype == object:
            # Check if it contains arrays
            if val.size > 0:
                flat_val = val.flatten()
                for item in flat_val:
                    if isinstance(item, (np.ndarray, list)):
                        return True

        # Check for nested lists
        if isinstance(val, list) and len(val) > 0:
            # Check if it's a nested list (list of lists)
            if isinstance(val[0], (list, np.ndarray)):
                return True

        return False

    def convert_to_numpy_array(val):
        """Convert nested structure to proper numpy array."""

        # Handle numpy object arrays (from parquet)
        if isinstance(val, np.ndarray) and val.dtype == object:
            try:
                return np.array([convert_to_numpy_array(item) for item in val])
            except:
                try:
                    nested_list = [item.tolist() if isinstance(item, np.ndarray) else item for item in val]
                    return np.array(nested_list)
                except:
                    return val

        # Handle nested lists
        if isinstance(val, list):
            try:
                return np.array(val)
            except:
                return val

        if val is None or pd.isna(val):
            return val

        return val

    for col in df_restored.columns:
        if df_restored[col].dtype == "object":
            # Check if this column contains nested array structures
            sample_non_null = df_restored[col].dropna()
            if len(sample_non_null) > 0:
                first_val = sample_non_null.iloc[0]
                if is_nested_array_structure(first_val):
                    print(f"Restoring numpy arrays in column: {col}")
                    df_restored[col] = [convert_to_numpy_array(x) for x in df_restored[col]]

    return df_restored


NUMERIC_KINDS = {"b", "i", "u", "f", "c"}  # bool, int, uint, float, complex
STRING_KINDS = {"U", "S"}                   # unicode, byte string


def _is_null_scalar(x):
    """True if x is None, NaN, pd.NA, NaT, or any pandas-null sentinel."""
    if x is None:
        return True
    if isinstance(x, float) and np.isnan(x):
        return True
    if hasattr(x, "shape"):
        return False
    try:
        import pandas as pd
        res = pd.isna(x)
    except Exception:
        return False
    return res is True  # only trust a real Python bool


def _scalar_equal(x, y, rtol=1e-8, atol=1e-3):
    """Equality for two Python/numpy scalars or nested arrays."""
    if hasattr(x, "shape") and hasattr(y, "shape"):
        return safe_compare_arrays(x, y, rtol=rtol, atol=atol)
    if hasattr(x, "shape") or hasattr(y, "shape"):
        return False

    x_null = _is_null_scalar(x)
    y_null = _is_null_scalar(y)
    if x_null and y_null:
        return True
    if x_null or y_null:
        return False

    numeric_types = (int, float, complex, np.integer, np.floating,
                     np.complexfloating, np.bool_)
    if isinstance(x, numeric_types) and isinstance(y, numeric_types):
        return bool(np.isclose(x, y, rtol=rtol, atol=atol, equal_nan=True))

    if isinstance(x, bytes):
        x = x.decode("utf-8", errors="replace")
    if isinstance(y, bytes):
        y = y.decode("utf-8", errors="replace")

    try:
        return bool(x == y)
    except (TypeError, ValueError):
        return False


def safe_compare_arrays(val1, val2, rtol=1e-8, atol=1e-3):
    """Safely compare two numpy arrays in a dtype-aware way."""
    if not (hasattr(val1, "shape") and hasattr(val2, "shape")):
        raise TypeError("safe_compare_arrays expects numpy arrays on both sides")
    if val1.shape != val2.shape:
        return False
    if val1.size == 0:
        return True

    k1 = val1.dtype.kind
    k2 = val2.dtype.kind

    if k1 in NUMERIC_KINDS and k2 in NUMERIC_KINDS:
        return bool(np.allclose(val1, val2, equal_nan=True, rtol=rtol, atol=atol))

    if k1 in STRING_KINDS and k2 in STRING_KINDS:
        a1 = val1.astype(str) if k1 == "S" else val1
        a2 = val2.astype(str) if k2 == "S" else val2
        return bool(np.array_equal(a1, a2))

    if (k1 in NUMERIC_KINDS and k2 in STRING_KINDS) or \
       (k1 in STRING_KINDS and k2 in NUMERIC_KINDS):
        return False

    if k1 == "O" or k2 == "O":
        try:
            a1 = np.asarray(val1.tolist(), dtype=float)
            a2 = np.asarray(val2.tolist(), dtype=float)
            if a1.shape == a2.shape and a1.dtype.kind in NUMERIC_KINDS:
                return bool(np.allclose(a1, a2, equal_nan=True, rtol=rtol, atol=atol))
        except (ValueError, TypeError):
            pass

        for x, y in zip(val1.ravel(), val2.ravel()):
            if not _scalar_equal(x, y, rtol=rtol, atol=atol):
                return False
        return True

    return bool(np.array_equal(val1, val2))