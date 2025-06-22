## get table names from 2d database
import warnings
warnings.filterwarnings("ignore")

## import packages/libraries
import sys
import sqlite3

## append filepath to allow files to be called from within project folder
sys.path.append('/home/gerard/Desktop/capstone_project/patoms')
sys.path.append('/home/gerard/Desktop/capstone_project')

def table_names_2d(cursor):
    cur2d = cursor
    ## get pattern tables and find empty and non-empty
    nonref = cur2d.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE '%ref%';").fetchall()
    empty_nonref = []
    nonempty_nonref = []
    for (table,) in nonref:
        row = cur2d.execute(f"SELECT * FROM {table} LIMIT 1").fetchone()
        if row is None:
            empty_nonref.append(table)
        else:
            nonempty_nonref.append(table)

    ## get reference tables and find empty and non-empty
    ref = cur2d.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%ref%';").fetchall()
    empty_ref = []
    nonempty_ref = []
    for (table,) in ref:
        row = cur2d.execute(f"SELECT * FROM {table} LIMIT 1").fetchone()
        if row is None:
            empty_ref.append(table)
        else:
            nonempty_ref.append(table)
    
    return [empty_nonref, nonempty_nonref, empty_ref, nonempty_ref]