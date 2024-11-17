from functools import singledispatch
import numpy as np
import pandas as pd
import os
import copy
from typing import Union, get_args


SUPPORTED_TYPES = Union[str, pd.DataFrame, pd.Series, tuple]

@singledispatch
def _load_data(data: SUPPORTED_TYPES):
    """Завантаження даних для підтримуваних типів"""
    supported = get_args(SUPPORTED_TYPES)  # Отримуємо перелік підтримуваних типів
    raise TypeError(f"Unsupported type: {type(data)}. Supported types: {', '.join(t.__name__ for t in supported)}")

# Завантаження даних із файлу
@_load_data.register
def _(data: str):
    if not os.path.isfile(data):
        raise FileNotFoundError(f"File not found: {data}")
    
    # Перевірка розширення
    file_extension = os.path.splitext(data)[-1].lower()
    if file_extension == ".csv":
        return pd.read_csv(data)
    elif file_extension == ".xlsx":
        return pd.read_excel(data)
    else:
        raise TypeError(f"Unsupported file format: {file_extension}. Supported formats: .csv, .xlsx")

# Завантаження даних із DataFrame
@_load_data.register
def _(data: pd.DataFrame):
    return copy.deepcopy(data)

@_load_data.register
def _(data: pd.Series):
    return copy.deepcopy(data.to_frame())

@_load_data.register
def _(data: tuple):
    if len(data)!=2:
        raise ValueError(f"Tuple must contain exactly two elements: (absorbance, wavelength). Received: {data}")
    else:

        absorbance = _load_data(data[0])
        wavelength = _load_data(data[1])

    # Перевірка на сумісність розмірів
    if len(absorbance) != len(wavelength):
        raise ValueError(f"Absorbance and wavelength lengths do not match. "
                         f"Absorbance length: {len(absorbance)}, Wavelength length: {len(wavelength)}")
    
    return absorbance, wavelength

def _wl_split_dataset(data:pd.DataFrame):
    LEAST_WL = 399
    row = data.index[(data>=LEAST_WL).all(axis=1)].to_list()
    column = data.columns[(data>=LEAST_WL).all(axis=0)].to_list()
    if not row and 
    if row:
        wl = data.iloc[row,:]
        absorbance = data.drop(row,axis=0)
    elif column:
        wl = data.iloc[:,column]
        absorbance = data.drop(column,axis=1)
    return absorbance,wl
data = pd.read_csv('bogdan_set_raw_test.csv')
data1 = list(data.iloc[:,:-3].to_numpy())
wl = list(pd.read_csv('wl_test.csv').T.to_numpy())
data_wl = pd.DataFrame(data1+wl)

smth = _wl_split_dataset(data.iloc[:,:-3])
smth

