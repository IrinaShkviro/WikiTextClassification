# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 22:48:10 2016

@author: Admin
"""
import pymysql as mysql
import numpy as np

#import pymysql.connector
from pprint import pprint

cnx = mysql.connect(user='root', password='DbMysqlP@ss2016',
                              host='localhost', port=3306,
                              database='wiki')

curs = cnx.cursor()
  
def get_titles_by_id(subcats):
    category_titles = []
    for category in subcats:
        curs.execute('select page_title from page\
            where page_id = "cur_cat" limit 1'.replace('cur_cat', str(category[0])))
        category_titles.append(curs.fetchone())
    return category_titles
                       
def select_pages_by_category(category):
    curs.execute('select cl_from from categorylinks \
        where cl_to = "cur_cat" and cl_type = "page"'.replace('cur_cat', category))
    pages_in_category = curs.fetchall()
    curs.execute('select cl_from from categorylinks \
        where cl_to = "cur_cat" and cl_type = "subcat"'.replace('cur_cat', category))
    subcats = curs.fetchall()
    #print(subcats)
    subcat_titles = get_titles_by_id(subcats)
    return (pages_in_category, subcat_titles)
    
def dfs(category_list, cur_lvl, max_lvl, page_list = None):
    subcats = []
    for (category, ) in category_list:
        #print(category)
        (pages_in_category, subcat_titles) = select_pages_by_category(category.decode("ascii", "ignore") )
        if cur_lvl == 0:
            page_list = set(pages_in_category)
        else:
            page_list = page_list.union(set(pages_in_category))
        #page_list.extend(pages_in_category)
        subcats.extend(subcat_titles)
    if (cur_lvl < max_lvl and len(subcats) > 0):
        page_list = dfs(subcats, cur_lvl + 1, max_lvl, page_list)
    return page_list

page_list = dfs([(b"Fields_of_mathematics",)], 0, 3)
file_name = "ids.txt"
f = open(file_name, 'w')
for (page, ) in page_list:
    f.write(str(page) + '\n')
f.close()
print(len(page_list), "count")
cnx.close()