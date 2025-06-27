## add ignore warnings for now, will remove and debug once full algorithm is complete
import warnings
warnings.filterwarnings("ignore")

## import packages/libraries
from time import clock_gettime_ns, CLOCK_REALTIME
import sys
import sqlite3

## append filepath to allow files to be called from within project folder
sys.path.append('/home/gerard/Desktop/capstone_project/patoms')
sys.path.append('/home/gerard/Desktop/capstone_project')

con2d = sqlite3.connect("db2d.db")
cur2d = con2d.cursor()

## update reference data table
def update_ref_table(patom_table):
    table_num = patom_table[-6:]
    cur2d.executemany(f"DROP TABLE IF EXISTS pat_2d_{table_num}_ref;")
    cur2d.execute(f"CREATE TABLE pat_2d_ref_{table_num}(x_pos_dist, y_pos_dist, x_cent, y_cent, patom_ind, frame_ind, patom_time);")
    # think of 4 volume areas on the unit circle: 
    # area 1: pos x pos y
    # area 2: pos x neg y
    # area 3: neg x pos y
    # area 4: neg x neg y
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
                                    
                                from {patom_table}
                                -- fileds: x_pos_dist, y_pos_dist, x_cent, y_cent, patom_ind, frame_ind, patom_time * rows
                            ) as bse1

                        ) as bse2

                        group by quad, axv, ayv, ac, x_cent_a, y_cent_a
                        
                    ;""").fetchall()
    ## next step is to turn output into table that has the same format as the input tables
    x_dists = []; y_dists = []; x_cents = []; y_cents = []
    for quad in ref_table:
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
    cur2d.executemany(f"INSERT INTO pat_2d_ref_{table_num}(x_pos_dist, y_pos_dist, x_cent, y_cent, patom_ind, frame_ind, patom_time) VALUES (?,?,?,?,?,?,?)", ref_table_complete)
    