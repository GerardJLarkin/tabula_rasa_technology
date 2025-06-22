import warnings
warnings.filterwarnings("ignore")

# function to compare patterns for similarity
import numpy as np

## set similarity threshold limits
distance_threshold = 0.50
xc_perc = 0.50 
yc_perc = 0.50
xp_perc = 0.50
xn_perc = 0.50
yp_perc = 0.50
yn_perc = 0.50

# only want final row of pattern so consider extracting that before passing as arguments
def pattern_centroid_compare(pat_cent_x1, pat_cent_y1, pat_cent_x2, pat_cent_y2):
    if pat_cent_x1 == pat_cent_x2:
        xc_sim = 1.0
    else:
        xc_sim = 1.0 - (abs(pat_cent_x1 - pat_cent_x2)/((pat_cent_x1 + pat_cent_x2)/2))
    
    if pat_cent_y1 == pat_cent_y2:
        yc_sim = 1.0
    else:
        yc_sim = 1.0 - (abs(pat_cent_y1 - pat_cent_y2)/((pat_cent_y1 + pat_cent_y2)/2))
    
    if (xc_sim >= xc_perc) and (yc_sim >= yc_perc):
        return xc_sim, yc_sim
    else:
        return None

# only intersted is final column of pattern so extract that before passing to argument
def distance_compare(dist1_x, dist1_y, dist2_x, dist2_y):
    negx1 = [i for i in dist1_x if i < 0]; posx1 = [i for i in dist1_x if i >= 0]; negy1 = [i for i in dist1_y if i < 0]; posy1 = [i for i in dist1_y if i >= 0]
    negx2 = [i for i in dist2_x if i < 0]; posx2 = [i for i in dist2_x if i >= 0]; negy2 = [i for i in dist2_y if i < 0]; posy2 = [i for i in dist2_y if i >= 0]
    arr1_posx = np.array(posx1); arr1_negx = np.array(negx1)
    arr1_posy = np.array(posy1); arr1_negy = np.array(negy1)
    arr2_posx = np.array(posx2); arr2_negx = np.array(negx2)
    arr2_posy = np.array(posy2); arr2_negy = np.array(negy2)

    diffxp = arr1_posx[:, None] - arr2_posx
    avgxp = (arr1_posx[:, None]) + np.abs(arr2_posx) / 2
    sim_perc_xp = (diffxp / avgxp).flatten()
    sim_perc_xp = len(sim_perc_xp[sim_perc_xp <= distance_threshold].tolist())

    diffxn = arr1_negx[:, None] - arr2_negx
    avgxn = (arr1_negx[:, None]) + np.abs(arr2_negx) / 2
    sim_perc_xn = (diffxn / avgxn).flatten()
    sim_perc_xn = len(sim_perc_xn[sim_perc_xn <= distance_threshold].tolist())

    diffyp = arr1_posy[:, None] - arr2_posy
    avgyp = (arr1_posy[:, None]) + np.abs(arr2_posy) / 2
    sim_perc_yp = (diffyp / avgyp).flatten()
    sim_perc_yp = len(sim_perc_yp[sim_perc_yp <= distance_threshold].tolist())

    diffyn = arr1_negy[:, None] - arr2_negy
    avgyn = (arr1_negy[:, None]) + np.abs(arr2_negy) / 2
    sim_perc_yn = (diffyn / avgyn).flatten()
    sim_perc_yn = len(sim_perc_yn[sim_perc_yn <= distance_threshold].tolist())

    len_negx = len(negx1) * len(negx2); len_posx = len(posx1) * len(posx2); len_negy = len(negy1) * len(negy2); len_posy = len(posy1) * len(posy2)
    sim_perc_xp = sim_perc_xp/len_posx
    sim_perc_yp = sim_perc_yp/len_posy
    sim_perc_xn = sim_perc_xn/len_negx
    sim_perc_yn = sim_perc_yn/len_negy
    
    if (sim_perc_xp >= xp_perc) and (sim_perc_xn >= xn_perc) and (sim_perc_yp >= yp_perc) and (sim_perc_yn >= yn_perc):
        return sim_perc_xp, sim_perc_xn, sim_perc_yp, sim_perc_yn
    else:
        return None

# adding in comment line
def pattern_compare_2d(pat1_dist_x, pat1_dist_y, pat1_cent_x, pat1_cent_y, pat2_dist_x, pat2_dist_y, pat2_cent_x, pat2_cent_y, i):
    centroid_sim = pattern_centroid_compare(pat1_cent_x, pat1_cent_y, pat2_cent_x, pat2_cent_y)
    dist_sim = distance_compare(pat1_dist_x, pat1_dist_y, pat2_dist_x, pat2_dist_y)
    
    if centroid_sim and dist_sim:
        return i