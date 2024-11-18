from functools import singledispatch
import numpy as np
import pandas as pd
from DataLoader import DataLoader

class BaseData:
    def __init__(self):
        self.wavelength = None
        self.absorbance = None
        self.targets = None
        self.dataset = None

    def load_data(self,dataset = None, wl = None):
        loader = DataLoader()
        self.wavelength, self.absorbance, self.targets, self.dataset = loader.load_data(dataset, wl)

mydata = BaseData()
dataset = pd.read_csv('bogdan_set_raw_test.csv')
wl = pd.read_csv('wl_test.csv')
mydata.load_data(wl)
mydata.targets


    
