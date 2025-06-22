# function to compare patterns for similarity
import numpy as np
from os import listdir
import csv
from itertools import product

# similarity will be a measure of
# the length of the patterns - maybe?
# the percentage of pixels that fall within the distance threshold for the smaller pattern
# the percentage of pixels in the smaller pattern that fall within the distance threshold measured against the total number of pixels in the larger pattern
# the percentage difference in the pattern centroids
# the colour values of each pixel that are identified as falling within the threshold distance
# the orientation of the pattern - maybe?
# the positioning of the pattern in relation to other patterns

centroid_diff = 0.3
distance_threshold = 0.05
sim_threshold = 0.95

files = [f for f in listdir("/home/gerard/Desktop/capstone_project/patoms") if f.endswith(".csv")]

compare_list = list(product(files, files))
filtered_lists = [sublist for sublist in compare_list if len(set(sublist)) == 2]

for ind, i in enumerate(filtered_lists):
    with open(f'/home/gerard/Desktop/capstone_project/patoms/{i[0]}', 'r') as f1:
        reader = csv.reader(f1)
        data = list(reader)
        data_array1 = np.array(data, dtype=float)
        shape1 = data_array1.shape[0] - 1
    with open(f'/home/gerard/Desktop/capstone_project/patoms/{i[1]}', 'r') as f2:
        reader = csv.reader(f2)
        data = list(reader)
        data_array2 = np.array(data, dtype=float)
        shape2 = data_array2.shape[0] - 1
        # check similarity between nearest pixel(s) distiance from centroid is within tolerance (add weighting)
        # check similarity between nearest pixel(s) colour is within tolerance (add weighting)
        # determine the smaller pattern (array shape on axis 1 is smaller)
        # set the distance threshold limit (i.e. the point regardless of location must have a similar distance to its centroid point)
        # set the threshold for the number of points between objects (i.e. for the smaller pattern at least 80% of points must be within distance threshold limit)
        # for each point in the smaller pattern calculate the distance between the point and its centroid and all points in the larger pattern
        #print(shape1, shape2)
        # centroid of patterns has to be within tolerance also
        nxc1 = data_array1[-1:][0,4]; nxc2 = data_array2[-1:][0,4]
        if nxc1 == nxc2:
            nxc_diff = 0.0
        else:
            nxc_diff = abs(nxc1 - nxc2)/((nxc1 + nxc2)/2)
        nyc1 = data_array1[-1:][0,5]; nyc2 = data_array2[-1:][0,5] 
        if nyc1 == nyc2:
            nyc_diff = 0.0
        else:
            nyc_diff = abs(nyc1 - nyc2)/((nyc1 + nyc2)/2)
        nzc1 = data_array1[-1:][0,6]; nzc2 = data_array2[-1:][0,6] 
        if nzc1 == nzc2:
            nzc_diff = 0.0
        else:
            nzc_diff = abs(nzc1 - nzc2)/((nzc1 + nzc2)/2)
        if (nxc_diff + nyc_diff + nzc_diff) <= centroid_diff:
            if shape1 <= shape2:
                # create a dict to hold the key:value pair of norm_pixel_loc:num of corresponding norm_pixel_locations within its threshold limit
                dic1 = dict.fromkeys(list(range(shape1)), 0)
                for ind1, i in enumerate(data_array1):
                    i_norm_dist = i[-1]
                    for ind2, j in enumerate(data_array2):
                        j_norm_dist = j[-1]
                        perc_diff = abs(i_norm_dist - j_norm_dist)/((i_norm_dist + j_norm_dist)/2)
                        # check if pixel falls within the threshold of the distance difference
                        if perc_diff <= distance_threshold:
                            # if within threshold increase the count of that normalised pixel centroid distance
                            dic1[ind1] +=1
                # count the the number of normalised pixel locations who have values greater than 1
                sim_sum = sum(int(i) > 0 for i in dic1.values())
                # calculated value to determine how similar (based on normalised pixel distiance from centroid) pattern A (smaller pattern) is to be pattern B (larger pattern)
                # do I need to conisder what the actual location is????
                # do I need to consider if a single pixel location in pattern A is within the distance threshold for a large portion of pattern Bs norm pixel locations?
                sim_perc1 = sim_sum/shape1
                sim_perc2 = sim_sum/shape2
                if sim_perc >= sim_threshold:
                    print(filtered_lists[ind], ind, sim_perc)
            else:
                dic2 = dict.fromkeys(list(range(shape2)), 0)
                for ind1, i in enumerate(data_array2):
                    i_norm_dist = i[-1]
                    for ind2, j in enumerate(data_array1):
                        j_norm_dist = j[-1]
                        perc_diff = abs(i_norm_dist - j_norm_dist)/((i_norm_dist + j_norm_dist)/2)
                        if perc_diff <= distance_threshold:
                            dic2[ind1] +=1
                
                sim_sum = sum(int(i) > 0 for i in dic2.values())
                sim_perc = sim_sum/shape1
                #if sim_perc >= sim_threshold:
                    #print(filtered_lists[ind], ind, sim_perc)
