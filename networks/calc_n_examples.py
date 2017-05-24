# -*- coding: utf-8 -*-
"""
Created on Thu May 11 12:10:32 2017

@author: Admin
"""

import os
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from doc_iterator import DocIterator

def calc_total_examples(dir_name = 'preproceed'):
    this_folder = os.path.dirname(os.path.abspath(__file__))
    dir_name = os.path.join(this_folder, '../' + dir_name)
    saved_dir = os.path.join(this_folder, '../saved')
    if not os.path.isdir(saved_dir):
        os.mkdir(saved_dir)
        
    train_docs = DocIterator(dir_name)
    
    # train the model
    train_samples = train_docs.calc_total_words()
    print('total examples: ', train_samples)
    
    #save the model    
    os.chdir(saved_dir)
    file = open('total_samples.txt', 'w')
    file.write(str(train_samples) + '\n')
    file.close()
    os.chdir('../networks')
    

if __name__ == '__main__':
    calc_total_examples()