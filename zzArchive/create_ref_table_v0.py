## add ignore warnings for now, will remove and debug once full algorithm is complete
import warnings
warnings.filterwarnings("ignore")

## import packages/libraries
from time import perf_counter, clock_gettime_ns, CLOCK_REALTIME
import numpy as np
from operator import itemgetter
from multiprocessing import Pool, cpu_count
from itertools import product
import sys
import sqlite3

## append filepath to allow files to be called from within project folder
sys.path.append('/home/gerard/Desktop/capstone_project/patoms')
sys.path.append('/home/gerard/Desktop/capstone_project')

con2d = sqlite3.connect("database_2d.db")
cur2d = con2d.cursor()

## build reference data table
# get non empty pattern tables
tables_nonref_test = cur2d.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE '%ref%';")
tables_nonref_test = tables_nonref_test.fetchall()  # List of tuples with table names
nonempty_nonref_tables_test = []
# Loop through each table and create reference table
for (table,) in tables_nonref_test: 
    cur2d.execute(f"SELECT * FROM {table} LIMIT 1;") 
    rows = cur2d.fetchone()
    if rows is None:
        pass
    else:
        nonempty_nonref_tables_test.append(table)
    # ref table requires avg pat cent x, y, the avg num of rows, which consists of the most frequent x, y positions
    # think of 4 volume areas on the unit circle: 
    # area 1: pos x pos y
    # area 2: pos x neg y
    # area 3: neg x pos y
    # area 4: neg x neg y
for ix, table in enumerate(nonempty_nonref_tables_test):
    ref_table = cur2d.execute(f"""
                    select
                        *
                        from
                        (
                        select
                            quad,
                            avg(x_pos_dist) over(partition by quad) as axv,
                            avg(y_pos_dist) over(partition by quad) as ayv,
                            count(x_pos_dist) over(partition by quad) as ac,
                            avg(x_cent) over() as x_cent_a,
                            avg(y_cent) over() as y_cent_a
                            
                            from
                            (
                            select 
                                patom_time, count(*) over() as num_rows, x_pos_dist, y_pos_dist, x_cent, y_cent, patom_ind, frame_ind,
                                case 
                                    when x_pos_dist >= 0 and y_pos_dist >= 0 then 1
                                    when x_pos_dist >= 0 and y_pos_dist < 0 then 2
                                    when x_pos_dist < 0 and y_pos_dist >= 0 then 3
                                    when x_pos_dist < 0 and y_pos_dist < 0 then 4
                                end as quad
                                    
                                from {table}
                                -- fileds: x_pos_dist, y_pos_dist, x_cent, y_cent, patom_ind, frame_ind, patom_time * rows
                            ) as bse1

                        ) as bse2

                        group by quad, axv, ayv, ac, x_cent_a, y_cent_a
                        
                    ;""")
    ref_table_out = ref_table.fetchall()
    ## next step is to turn output into table that has the same format as the input tables
    x_dists = []; y_dists = []; x_cents = []; y_cents = []
    for quad in ref_table_out:
        x_d = [quad[1]] * quad[3]; x_dists.append(x_d)
        y_d = [quad[2]] * quad[3]; y_dists.append(y_d)
        x_c = [quad[4]] * quad[3]; x_cents.append(x_c)
        y_c = [quad[5]] * quad[3]; y_cents.append(y_c)

    x_dist = [x for xs in x_dists for x in xs]
    y_dist = [x for xs in y_dists for x in xs]
    x_cent = [x for xs in x_cents for x in xs]
    y_cent = [x for xs in y_cents for x in xs]

    patom_ind = [0] * len(x_dist)
    frame_ind = [0] * len(x_dist)
    patom_time = [clock_gettime_ns(CLOCK_REALTIME)] * len(x_dist)
    ref_table_complete = list(zip(x_dist, y_dist, x_cent, y_cent, patom_ind, frame_ind, patom_time))
    ref_patom_table_num = nonempty_nonref_tables_test[ix][-3:]
    cur2d.executemany(f"INSERT INTO pat_2d_ref_{ref_patom_table_num}(x_pos_dist, y_pos_dist, x_cent, y_cent, patom_ind, frame_ind, patom_time) VALUES (?,?,?,?,?,?,?)", ref_table_complete)
    
con2d.commit()
con2d.close()