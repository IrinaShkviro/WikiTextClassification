# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 11:10:21 2016

@author: Admin
"""

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

ids = set([10, 276])

category = "Mathematics"
dir_name = "wiki_texts_for_%s" % (category)
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)


try:
    tree = ET.ElementTree(file='1.xml')
except IOError as e:
    print('nERROR - cant find file: %sn' % e)
root = tree.getroot()
for child_of_root in root.iter('{http://www.mediawiki.org/xml/export-0.10/}page'):
    cur_id = child_of_root.find('{http://www.mediawiki.org/xml/export-0.10/}id')
    if int(cur_id.text) in ids:
        cur_title = child_of_root.find('{http://www.mediawiki.org/xml/export-0.10/}title')
        cur_revision = child_of_root.find('{http://www.mediawiki.org/xml/export-0.10/}revision')
        cur_text = cur_revision.find('{http://www.mediawiki.org/xml/export-0.10/}text')
        file_name = "id_%s.txt" % (cur_id.text)
        os.chdir(dir_name)
        f = open(file_name, 'w')
        f.write(cur_title.text + '\n')
        f.write(cur_text.text)
        f.close()
        os.chdir('..')                           