# convert patom format to table format
import warnings
warnings.filterwarnings("ignore")

## import packages/libraries
import sys

## append filepath to allow files to be called from within project folder
sys.path.append('/home/gerard/Desktop/capstone_project/patoms')
sys.path.append('/home/gerard/Desktop/capstone_project')

def patom_to_table_func(new_patom):
    pat_len = len(new_patom[0])
    cent_x = [new_patom[2]] * pat_len
    cent_y = [new_patom[3]] * pat_len
    patom_ind = [new_patom[4]] * pat_len
    frame_ind = [new_patom[5]] * pat_len
    patom_time = [new_patom[6]] * pat_len
    patom_to_table = list(zip(new_patom[0], new_patom[1], cent_x, cent_y, patom_ind, frame_ind, patom_time))

    return patom_to_table