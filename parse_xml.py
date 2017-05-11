# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 22:43:13 2016

@author: Admin
"""

try:
 import xml.etree.cElementTree as ET
except ImportError:
 import xml.etree.ElementTree as ET
 
import os
import sys

data_file = open("ids.txt", "r")
ids = []
for line in data_file:
    ids.append(int(line))
data_file.close()
ids = set(ids)

category = "fields_of_mathematics"
dir_name = category
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)

# get an iterable
context = ET.iterparse("enwiki-20170320-pages-articles.xml", events=("start", "end"))
#context = ET.iterparse("1.xml", events=("start", "end"))

start_page = False
suit_id = False
first_id = False
title = "title"
cur_id = "0"
finded = 0
set_size = len(ids)
for event, elem in context:
    if elem.tag == "{http://www.mediawiki.org/xml/export-0.10/}page":
        if (event == "start"):
            start_page = True
            first_id = True
        else:
            start_page = False
            first_id = False
            elem.clear()
    if start_page and first_id and elem.tag == "{http://www.mediawiki.org/xml/export-0.10/}id"\
    and event == "end": 
        first_id = False
        if elem.text == None:
            continue
        if int(elem.text) in ids:
            suit_id = True
            cur_id = elem.text
        else:
            suit_id = False
            elem.clear()
    if start_page and elem.tag == "{http://www.mediawiki.org/xml/export-0.10/}title"\
    and event == "end":
        title = elem.text
    if start_page and suit_id and elem.tag == "{http://www.mediawiki.org/xml/export-0.10/}text"\
    and event == "end":
        if elem.text == None or title == None:
            continue
        file_name = "id_%s.txt" % (cur_id)
        os.chdir(dir_name)
        f = open(file_name, 'w')
        f.write(title + '\n')
        f.write(elem.text)
        f.close()
        os.chdir('..')
        finded = finded + 1
        sys.stdout.write("%i / %i \r" % (finded, set_size))
        elem.clear()