import pandas as pd
import numpy as np
from functools import singledispatch
import os
from typing import Union,get_args

SUPPORTED_TYPES = Union[str, pd.DataFrame, pd.Series]

@singledispatch
def _load_data(data: SUPPORTED_TYPES):
    """Завантаження даних для підтримуваних типів"""
    supported = ", ".join(t.__name__ for t in get_args(SUPPORTED_TYPES))
    raise TypeError(f"Unsupported type: {type(data)}. Supported types: {', '.join(t.__name__ for t in supported)}")

@_load_data.register
def _(data: str):
    if not os.path.isfile(data):
        raise FileNotFoundError(f"File not found: {data}")
    
    file_extension = os.path.splitext(data)[-1].lower()
    if file_extension == ".csv":
        return pd.read_csv(data)
    elif file_extension == ".xlsx":
        return pd.read_excel(data)
    else:
        raise TypeError(f"Unsupported file format: {file_extension}. Supported formats: .csv, .xlsx")

@_load_data.register
def _(data: pd.DataFrame):
    return data

@_load_data.register
def _(data: pd.Series):
    return data.to_frame()
































    
