# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 14:37:29 2016

@author: Admin
"""

import os
import time
import sys
import logging
from pprint import pprint
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

'''
import pkgutil
modules = pkgutil.iter_modules(gensim.__path__)
for module in modules:
    print(module[1])
'''

class MyLines(object):
    def __init__(self, dirname):
        self.dirname = dirname
        self.file_num = 0
        self.cur_name = ""
        self.dir_size =  (len(os.listdir(self.dirname)))
 
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            self.file_num = self.file_num + 1
            self.cur_name = fname
            sys.stdout.write("%i / %i \r" % (self.file_num, self.dir_size))
            for line in open(os.path.join(self.dirname, fname)):               
                yield line
            
                
dir_name = "preproceed"
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)
    
lines = MyLines('plains') # a memory-friendly iterator
write_lines = str("")
cur_file = 1
cur_name = ""
for line in lines:
    if lines.file_num != cur_file:
        os.chdir(dir_name)
        f = open(cur_name, "w")
        f.write(write_lines)
        f.close()
        os.chdir("..")
        write_lines = str("")
        cur_file = cur_file + 1
        
    cur_name = lines.cur_name
    line.strip()
    if line.isspace():
        continue
    
    line = line.replace("Category:", "")
    
    preproc_line = line
    start_search_id = 0
    
    while True:
        open_bracket = preproc_line.find('[', start_search_id)
        if (open_bracket == -1):
            break
        close_bracket = preproc_line.find(']', open_bracket)
        if (close_bracket == -1):
            break        
        digit_substr = preproc_line[open_bracket + 1 : close_bracket]
        if digit_substr.isdigit():
            start_search_id = open_bracket
            preproc_line = preproc_line[ : open_bracket] + preproc_line[close_bracket + 1 : ]
        else:
            start_search_id = close_bracket + 1
    
    write_lines = write_lines + preproc_line

os.chdir(dir_name)
f = open(cur_name, "w")
f.write(write_lines)
f.close()
os.chdir("..")

