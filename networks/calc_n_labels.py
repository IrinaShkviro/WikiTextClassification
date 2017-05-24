# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 13:26:46 2017

@author: Admin
"""

import pickle
import sys
import os

def calc_n_labels():
    this_folder = os.path.dirname(os.path.abspath(__file__))
    saved_folder = os.path.join(this_folder, '../saved')
    os.chdir(saved_folder)
    
    cat_codes_data = open('category_codes', 'rb')
    cat_dict = pickle.load(cat_codes_data)
    
    os.chdir('../networks')
    
    return len(cat_dict)
