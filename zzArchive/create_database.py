import sys
import numpy as np
from time import perf_counter
sys.path.append('/home/gerard/Desktop/capstone_project')

from snapshot_2d_pattern_v1 import patoms2d

# create a database to store 2d patterns
import sqlite3

con = sqlite3.connect("db2d.db")
cur = con.cursor()

## create tables and insert data from here
# cur.execute(f"CREATE TABLE pat_0(frame, colour, x_pos, y_pos, norm_x_pos, norm_y_pos, norm_dist, patom_ind)")
# con.commit()
# con.close()

# np.random.seed(5555)
# rand_array = np.random.random((30, 720, 1280))
# z_len = rand_array.shape[0]
# y_len = rand_array.shape[1]
# x_len = rand_array.shape[2]

# def create_2d_tables():
#     con = sqlite3.connect("db2d.db")
#     cur = con.cursor()
#     cur.execute("SELECT name FROM sqlite_master where type='table';")
#     tables = cur.fetchall()
#     table_nums = []
#     for ind, i in enumerate(tables):
#         table_no = int(tables[ind][0].split('_')[1])
#         table_nums.append(table_no)
#     max_table = max(table_nums)
#     ind = max_table + 1
#     cur.execute(f"CREATE TABLE pat_{ind}(frame, colour, x_pos, y_pos, norm_x_pos, norm_y_pos, norm_dist, patom_ind)")
#     con.commit()
#     con.close()

# def insert_2d_tables(td, ind):
#     con = sqlite3.connect("db2d.db")
#     cur = con.cursor()
#     cur.executemany(f"INSERT INTO pat_{ind}(frame, colour, x_pos, y_pos, norm_x_pos, norm_y_pos, norm_dist, patom_ind) VALUES (?,?,?,?,?,?,?,?)", td)
#     con.commit()
#     con.close()

# def flatten_tuple(nested_tuple):
#     flat_list = []
#     for item in nested_tuple:
#         if isinstance(item, (tuple, list)):  # Check if item is a tuple or list
#             flat_list.extend(flatten_tuple(item))  # Recurse into nested structure
#         else:
#             flat_list.append(item)
#     return flat_list

# ## get frame from data to determine patterns in 2d slice of data
# for frame in range(rand_array.shape[0]):
#     # for the moment create a new table for each frame and add patterns
#     strt = perf_counter()
#     # create new table
#     create_2d_tables()
#     new_2d_patoms = patoms2d(x_len, y_len, rand_array[frame,:,:])
#     for patom in new_2d_patoms:
#         updated_patoms = []
#         frame_ind = [frame] * len(patom)
#         zipped = list(zip(frame_ind, patom))
#         for nested_tuple in zipped:
#             flattened = flatten_tuple(nested_tuple)
#             updated_patoms.append(flattened)
#         # insert data
#         insert_2d_tables(updated_patoms, frame)
#     print("Insert into table (secs):", (perf_counter()-strt))
    
    # compare new 2d patoms with existing 2d patoms

## select from tables here to view data
table_select = cur.execute("select * from pat_29 where patom_ind = 0")
print(table_select.fetchall())

# each 2d pattern will be stored in a sequential order so needs to be linked to previous 
# next 2d patterns
# will a single table consist of 2d patterns that ocurr in a single frame?
# or will a single table consist of 2d patterns that are connected sequentially? I think this is the preferred option
# in the 2nd optionI can get the average of the 2d pattern
# should the patterns be stored in row or column order?
## fields in 2d ouput: each pattern is a table with rows, where the last row is the center of the pattern. 
## each row consists of attritbutes, color value, x pos, y pos, normalised x pos from pattern centroid,
## normalised y position from pattern centroid, normalised distance from pattern centroid (centroid is mean of x,y vals)

## create a table in the database that can support a table structure as above
## how do I know which database to store each pattern in?

## looks like I can only have one database: create db once
## each table will contain tables that are similar (set similarity measure)

## a single table will contain all patterns that are similar



 # check to see if pattern is similar for patterns in each table
    # last set of rows in each table will be the average of all other tables inserted into table
    # need to add a field to specify a collection of rows is a pattern
    # if pattern is not similar to existing patterns in any existing tables create new table and add pattern
# res = cur.execute("SELECT name FROM sqlite_master")
# out = res.fetchone()
# print(out)







# res = 

# con.close()