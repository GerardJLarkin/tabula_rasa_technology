import sys
import sqlite3
sys.path.append('/home/gerard/Desktop/capstone_project')

# create a database to store 2d patterns
con1 = sqlite3.connect("database_2d.db")
cur1 = con1.cursor()

# create empty tables in 2d database
for i in range(500):
  table_num = str(i).zfill(3)
  cur1.execute(f"DROP TABLE IF EXISTS pat_2d_{table_num};")

# create reference tables for each of the above tables
for i in range(500):
  table_num = str(i).zfill(3)
  cur1.execute(f"DROP TABLE IF EXISTS pat_2d_{table_num}_ref;")

con1.commit()
con1.close()


# create a database to store 3d patterns
# con2 = sqlite3.connect("database_3d.db")
# cur2 = con2.cursor()

# # create empty tables in 3d database
# for i in range(10000):
#   cur2.execute(f"CREATE TABLE pat_3d_{i}(colour, x_pos, y_pos, z_pos, norm_x_pos, norm_y_pos, norm_z_pos, norm_dist, patom_ind)")

# con2.commit()
# con2.close()