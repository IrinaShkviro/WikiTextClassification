# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 13:26:46 2017

@author: Admin
"""

import pickle
import sys
import os

def calc_n_labels():
    os.chdir('saved')
    
    cat_codes_data = open('category_codes', 'rb')
    cat_dict = pickle.load(cat_codes_data)
    
    os.chdir('..')
    
    return len(cat_dict)
