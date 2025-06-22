import warnings
warnings.filterwarnings("ignore")
import numpy as np

def pattern_compare_2d(new_patom, ref_array):
    # new patom: [colour, norm_x, norm_y, pattern_centroid_x, pattern_centroid_y, segment, segment_cnt, frame_ind_arr] + [col_d, xc_d, yc_d, x_d, y_d] + [segment_similar]
    # ref patoms: [colour, norm_x, norm_y, pattern_centroid_x, pattern_centroid_y, segment, segment_cnt, frame_ind_arr]
    choice = [0.25]
    nsegment1 = new_patom[new_patom[:,5] == 1]; rsegment1 = ref_array[ref_array[:,5] == 1]
    ncol1 = nsegment1[:,0].reshape(nsegment1.shape[0],1); rcol1 = rsegment1[:,0].reshape(1, rsegment1.shape[0]); mse_col1 = 1 - np.nan_to_num(np.mean((ncol1 - rcol1)) ** 2)
    nx1 = nsegment1[:,1].reshape(nsegment1.shape[0],1); rx1 = rsegment1[:,1].reshape(1, rsegment1.shape[0]); mse_x1 = 1 - np.mean((nx1 - rx1) ** 2)
    ny1 = nsegment1[:,2].reshape(nsegment1.shape[0],1); ry1 = rsegment1[:,2].reshape(1, rsegment1.shape[0]); mse_y1 = 1 - np.mean((ny1 - ry1) ** 2)
    nxc1 = nsegment1[:,3].reshape(nsegment1.shape[0],1); rxc1 = rsegment1[:,3].reshape(1, rsegment1.shape[0]); mse_xc1 = 1 - np.mean((nxc1 - rxc1) ** 2)
    nyc1 = nsegment1[:,4].reshape(nsegment1.shape[0],1); ryc1 = rsegment1[:,4].reshape(1, rsegment1.shape[0]); mse_yc1 = 1 - np.mean((nyc1 - ryc1) ** 2)
    cond1 = [(mse_col1 >= 0.4) & (mse_x1 >= 0.8) & (mse_y1 >= 0.8) & (mse_xc1 >= 0.7) & (mse_yc1 >= 0.7)]
    segment_similar1 = np.select(cond1, choice, 0)
    segment1_res = np.resize(np.array([mse_col1, mse_x1, mse_y1, mse_xc1, mse_yc1]), (nsegment1.shape[0],5))
    segment1 = np.hstack((nsegment1, segment1_res))

    nsegment2 = new_patom[new_patom[:,5] == 2]; rsegment2 = ref_array[ref_array[:,5] == 2]
    ncol2 = nsegment2[:,0].reshape(nsegment2.shape[0],1); rcol2 = rsegment2[:,0].reshape(1, rsegment2.shape[0]); mse_col2 = 1 - np.mean((ncol2 - rcol2) ** 2)
    nx2 = nsegment2[:,1].reshape(nsegment2.shape[0],1); rx2 = rsegment2[:,1].reshape(1, rsegment2.shape[0]); mse_x2 = 1 - np.mean((nx2 - rx2) ** 2)
    ny2 = nsegment2[:,2].reshape(nsegment2.shape[0],1); ry2 = rsegment2[:,2].reshape(1, rsegment2.shape[0]); mse_y2 = 1 - np.mean((ny2 - ry2) ** 2)
    nxc2 = nsegment2[:,3].reshape(nsegment2.shape[0],1); rxc2 = rsegment2[:,3].reshape(1, rsegment2.shape[0]); mse_xc2 = 1 - np.mean((nxc2 - rxc2) ** 2)
    nyc2 = nsegment2[:,4].reshape(nsegment2.shape[0],1); ryc2 = rsegment2[:,4].reshape(1, rsegment2.shape[0]); mse_yc2 = 1 - np.mean((nyc2 - ryc2) ** 2)
    cond2 = [(mse_col2 >= 0.4) & (mse_x2 >= 0.8) & (mse_y2 >= 0.8) & (mse_xc2 >= 0.7) & (mse_yc2 >= 0.7)]
    segment_similar2 = np.select(cond2, choice, 0)
    segment2_res = np.resize(np.array([mse_col2, mse_x2, mse_y2, mse_xc2, mse_yc2]), (nsegment2.shape[0],5))
    segment2 = np.hstack((nsegment2, segment2_res))

    nsegment3 = new_patom[new_patom[:,5] == 3]; rsegment3 = ref_array[ref_array[:,5] == 3]
    ncol3 = nsegment3[:,0].reshape(nsegment3.shape[0],1); rcol3 = rsegment3[:,0].reshape(1, rsegment3.shape[0]); mse_col3 = 1 - np.mean((ncol3 - rcol3) ** 2)
    nx3 = nsegment3[:,1].reshape(nsegment3.shape[0],1); rx3 = rsegment3[:,1].reshape(1, rsegment3.shape[0]); mse_x3 = 1 - np.mean((nx3 - rx3) ** 2)
    ny3 = nsegment3[:,2].reshape(nsegment3.shape[0],1); ry3 = rsegment3[:,2].reshape(1, rsegment3.shape[0]); mse_y3 = 1 - np.mean((ny3 - ry3) ** 2)
    nxc3 = nsegment3[:,3].reshape(nsegment3.shape[0],1); rxc3 = rsegment3[:,3].reshape(1, rsegment3.shape[0]); mse_xc3 = 1 - np.mean((nxc3 - rxc3) ** 2)
    nyc3 = nsegment3[:,4].reshape(nsegment3.shape[0],1); ryc3 = rsegment3[:,4].reshape(1, rsegment3.shape[0]); mse_yc3 = 1 - np.mean((nyc3 - ryc3) ** 2)
    cond3 = [(mse_col3 >= 0.4) & (mse_x3 >= 0.8) & (mse_y3 >= 0.8) & (mse_xc3 >= 0.7) & (mse_yc3 >= 0.7)]
    segment_similar3 = np.select(cond3, choice, 0)
    segment3_res = np.resize(np.array([mse_col3, mse_x3, mse_y3, mse_xc3, mse_yc3]), (nsegment3.shape[0],5))
    segment3 = np.hstack((nsegment3, segment3_res))

    nsegment4 = new_patom[new_patom[:,5] == 4]; rsegment4 = ref_array[ref_array[:,5] == 4]
    ncol4 = nsegment4[:,0].reshape(nsegment4.shape[0],1); rcol4 = rsegment4[:,0].reshape(1, rsegment4.shape[0]); mse_col4 = 1 - np.mean((ncol4 - rcol4) ** 2)
    nx4 = nsegment4[:,1].reshape(nsegment4.shape[0],1); rx4 = rsegment4[:,1].reshape(1, rsegment4.shape[0]); mse_x4 = 1 - np.mean((nx4 - rx4) ** 2)
    ny4 = nsegment4[:,2].reshape(nsegment4.shape[0],1); ry4 = rsegment4[:,2].reshape(1, rsegment4.shape[0]); mse_y4 = 1 - np.mean((ny4 - ry4) ** 2)
    nxc4 = nsegment4[:,3].reshape(nsegment4.shape[0],1); rxc4 = rsegment4[:,3].reshape(1, rsegment4.shape[0]); mse_xc4 = 1 - np.mean((nxc4 - rxc4) ** 2)
    nyc4 = nsegment4[:,4].reshape(nsegment4.shape[0],1); ryc4 = rsegment4[:,4].reshape(1, rsegment4.shape[0]); mse_yc4 = 1 - np.mean((nyc4 - ryc4) ** 2)
    cond4 = [(mse_col4 >= 0.4) & (mse_x4 >= 0.8) & (mse_y4 >= 0.8) & (mse_xc4 >= 0.7) & (mse_yc4 >= 0.7)]
    segment_similar4 = np.select(cond4, choice, 0)
    segment4_res = np.resize(np.array([mse_col4, mse_x4, mse_y4, mse_xc4, mse_yc4]), (nsegment4.shape[0],5))
    segment4 = np.hstack((nsegment4, segment4_res))

    nsegment5 = new_patom[new_patom[:,5] == 5]; rsegment5 = ref_array[ref_array[:,5] == 5]
    ncol5 = nsegment5[:,0].reshape(nsegment5.shape[0],1); rcol5 = rsegment5[:,0].reshape(1, rsegment5.shape[0]); mse_col5 = 1 - np.mean((ncol5 - rcol5) ** 2)
    nx5 = nsegment5[:,1].reshape(nsegment5.shape[0],1); rx5 = rsegment5[:,1].reshape(1, rsegment5.shape[0]); mse_x5 = 1 - np.mean((nx5 - rx5) ** 2)
    ny5 = nsegment5[:,2].reshape(nsegment5.shape[0],1); ry5 = rsegment5[:,2].reshape(1, rsegment5.shape[0]); mse_y5 = 1 - np.mean((ny5 - ry5) ** 2)
    nxc5 = nsegment5[:,3].reshape(nsegment5.shape[0],1); rxc5 = rsegment5[:,3].reshape(1, rsegment5.shape[0]); mse_xc5 = 1 - np.mean((nxc5 - rxc5) ** 2)
    nyc5 = nsegment5[:,4].reshape(nsegment5.shape[0],1); ryc5 = rsegment5[:,4].reshape(1, rsegment5.shape[0]); mse_yc5 = 1 - np.mean((nyc5 - ryc5) ** 2)
    cond5 = [(mse_col5 >= 0.4) & (mse_x5 >= 0.8) & (mse_y5 >= 0.8) & (mse_xc5 >= 0.7) & (mse_yc5 >= 0.7)]
    segment_similar5 = np.select(cond5, choice, 0)
    segment5_res = np.resize(np.array([mse_col5, mse_x5, mse_y5, mse_xc5, mse_yc5]), (nsegment5.shape[0],5))
    segment5 = np.hstack((nsegment5, segment5_res))

    nsegment6 = new_patom[new_patom[:,5] == 6]; rsegment6 = ref_array[ref_array[:,5] == 6]
    ncol6 = nsegment6[:,0].reshape(nsegment6.shape[0],1); rcol6 = rsegment6[:,0].reshape(1, rsegment6.shape[0]); mse_col6 = 1 - np.mean((ncol6 - rcol6) ** 2)
    nx6 = nsegment6[:,1].reshape(nsegment6.shape[0],1); rx6 = rsegment6[:,1].reshape(1, rsegment6.shape[0]); mse_x6 = 1 - np.mean((nx6 - rx6) ** 2)
    ny6 = nsegment6[:,2].reshape(nsegment6.shape[0],1); ry6 = rsegment6[:,2].reshape(1, rsegment6.shape[0]); mse_y6 = 1 - np.mean((ny6 - ry6) ** 2)
    nxc6 = nsegment6[:,3].reshape(nsegment6.shape[0],1); rxc6 = rsegment6[:,3].reshape(1, rsegment6.shape[0]); mse_xc6 = 1 - np.mean((nxc6 - rxc6) ** 2)
    nyc6 = nsegment6[:,4].reshape(nsegment6.shape[0],1); ryc6 = rsegment6[:,4].reshape(1, rsegment6.shape[0]); mse_yc6 = 1 - np.mean((nyc6 - ryc6) ** 2)
    cond6 = [(mse_col6 >= 0.4) & (mse_x6 >= 0.8) & (mse_y6 >= 0.8) & (mse_xc6 >= 0.7) & (mse_yc6 >= 0.7)]
    segment_similar6 = np.select(cond6, choice, 0)
    segment6_res = np.resize(np.array([mse_col6, mse_x6, mse_y6, mse_xc6, mse_yc6]), (nsegment6.shape[0],5))
    segment6 = np.hstack((nsegment6, segment6_res))

    nsegment7 = new_patom[new_patom[:,5] == 7]; rsegment7 = ref_array[ref_array[:,5] == 7]
    ncol7 = nsegment7[:,0].reshape(nsegment7.shape[0],1); rcol7 = rsegment7[:,0].reshape(1, rsegment7.shape[0]); mse_col7 = 1 - np.mean((ncol7 - rcol7) ** 2)
    nx7 = nsegment7[:,1].reshape(nsegment7.shape[0],1); rx7 = rsegment7[:,1].reshape(1, rsegment7.shape[0]); mse_x7 = 1 - np.mean((nx7 - rx7) ** 2)
    ny7 = nsegment7[:,2].reshape(nsegment7.shape[0],1); ry7 = rsegment7[:,2].reshape(1, rsegment7.shape[0]); mse_y7 = 1 - np.mean((ny7 - ry7) ** 2)
    nxc7 = nsegment7[:,3].reshape(nsegment7.shape[0],1); rxc7 = rsegment7[:,3].reshape(1, rsegment7.shape[0]); mse_xc7 = 1 - np.mean((nxc7 - rxc7) ** 2)
    nyc7 = nsegment7[:,4].reshape(nsegment7.shape[0],1); ryc7 = rsegment7[:,4].reshape(1, rsegment7.shape[0]); mse_yc7 = 1 - np.mean((nyc7 - ryc7) ** 2)
    cond7 = [(mse_col7 >= 0.4) & (mse_x7 >= 0.8) & (mse_y7 >= 0.8) & (mse_xc7 >= 0.7) & (mse_yc7 >= 0.7)]
    segment_similar7 = np.select(cond7, choice, 0)
    segment7_res = np.resize(np.array([mse_col7, mse_x7, mse_y7, mse_xc7, mse_yc7]), (nsegment7.shape[0],5))
    segment7 = np.hstack((nsegment7, segment7_res))

    nsegment8 = new_patom[new_patom[:,5] == 8]; rsegment8 = ref_array[ref_array[:,5] == 8]
    ncol8 = nsegment8[:,0].reshape(nsegment8.shape[0],1); rcol8 = rsegment8[:,0].reshape(1, rsegment8.shape[0]); mse_col8 = 1 - np.mean((ncol8 - rcol8) ** 2)
    nx8 = nsegment8[:,1].reshape(nsegment8.shape[0],1); rx8 = rsegment8[:,1].reshape(1, rsegment8.shape[0]); mse_x8 = 1 - np.mean((nx8 - rx8) ** 2)
    ny8 = nsegment8[:,2].reshape(nsegment8.shape[0],1); ry8 = rsegment8[:,2].reshape(1, rsegment8.shape[0]); mse_y8 = 1 - np.mean((ny8 - ry8) ** 2)
    nxc8 = nsegment8[:,3].reshape(nsegment8.shape[0],1); rxc8 = rsegment8[:,3].reshape(1, rsegment8.shape[0]); mse_xc8 = 1 - np.mean((nxc8 - rxc8) ** 2)
    nyc8 = nsegment8[:,4].reshape(nsegment8.shape[0],1); ryc8 = rsegment8[:,4].reshape(1, rsegment8.shape[0]); mse_yc8 = 1 - np.mean((nyc8 - ryc8) ** 2)
    cond8 = [(mse_col8 >= 0.4) & (mse_x8 >= 0.8) & (mse_y8 >= 0.8) & (mse_xc8 >= 0.7) & (mse_yc8 >= 0.7)]
    segment_similar8 = np.select(cond8, choice, 0)
    segment8_res = np.resize(np.array([mse_col8, mse_x8, mse_y8, mse_xc8, mse_yc8]), (nsegment8.shape[0],5))
    segment8 = np.hstack((nsegment8, segment8_res))

    nsegment9 = new_patom[new_patom[:,5] == 9]; rsegment9 = ref_array[ref_array[:,5] == 9]
    ncol9 = nsegment9[:,0].reshape(nsegment9.shape[0],1); rcol9 = rsegment9[:,0].reshape(1, rsegment9.shape[0]); mse_col9 = 1 - np.mean((ncol9 - rcol9) ** 2)
    nx9 = nsegment9[:,1].reshape(nsegment9.shape[0],1); rx9 = rsegment9[:,1].reshape(1, rsegment9.shape[0]); mse_x9 = 1 - np.mean((nx9 - rx9) ** 2)
    ny9 = nsegment9[:,2].reshape(nsegment9.shape[0],1); ry9 = rsegment9[:,2].reshape(1, rsegment9.shape[0]); mse_y9 = 1 - np.mean((ny9 - ry9) ** 2)
    nxc9 = nsegment9[:,3].reshape(nsegment9.shape[0],1); rxc9 = rsegment9[:,3].reshape(1, rsegment9.shape[0]); mse_xc9 = 1 - np.mean((nxc9 - rxc9) ** 2)
    nyc9 = nsegment9[:,4].reshape(nsegment9.shape[0],1); ryc9 = rsegment9[:,4].reshape(1, rsegment9.shape[0]); mse_yc9 = 1 - np.mean((nyc9 - ryc9) ** 2)
    cond9 = [(mse_col9 >= 0.4) & (mse_x9 >= 0.8) & (mse_y9 >= 0.8) & (mse_xc9 >= 0.7) & (mse_yc9 >= 0.7)]
    segment_similar9 = np.select(cond9, choice, 0)
    segment9_res = np.resize(np.array([mse_col9, mse_x9, mse_y9, mse_xc9, mse_yc9]), (nsegment9.shape[0],5))
    segment9 = np.hstack((nsegment9, segment9_res))

    nsegment10 = new_patom[new_patom[:,5] == 10]; rsegment10 = ref_array[ref_array[:,5] == 10]
    ncol10 = nsegment10[:,0].reshape(nsegment10.shape[0],1); rcol10 = rsegment10[:,0].reshape(1, rsegment10.shape[0]); mse_col10 = 1 - np.mean((ncol10 - rcol10) ** 2)
    nx10 = nsegment10[:,1].reshape(nsegment10.shape[0],1); rx10 = rsegment10[:,1].reshape(1, rsegment10.shape[0]); mse_x10 = 1 - np.mean((nx10 - rx10) ** 2)
    ny10 = nsegment10[:,2].reshape(nsegment10.shape[0],1); ry10 = rsegment10[:,2].reshape(1, rsegment10.shape[0]); mse_y10 = 1 - np.mean((ny10 - ry10) ** 2)
    nxc10 = nsegment10[:,3].reshape(nsegment10.shape[0],1); rxc10 = rsegment10[:,3].reshape(1, rsegment10.shape[0]); mse_xc10 = 1 - np.mean((nxc10 - rxc10) ** 2)
    nyc10 = nsegment10[:,4].reshape(nsegment10.shape[0],1); ryc10 = rsegment10[:,4].reshape(1, rsegment10.shape[0]); mse_yc10 = 1 - np.mean((nyc10 - ryc10) ** 2)
    cond10 = [(mse_col10 >= 0.4) & (mse_x10 >= 0.8) & (mse_y10 >= 0.8) & (mse_xc10 >= 0.7) & (mse_yc10 >= 0.7)]
    segment_similar10 = np.select(cond10, choice, 0)
    segment10_res = np.resize(np.array([mse_col10, mse_x10, mse_y10, mse_xc10, mse_yc10]), (nsegment10.shape[0],5))
    segment10 = np.hstack((nsegment10, segment10_res))

    nsegment11 = new_patom[new_patom[:,5] == 11]; rsegment11 = ref_array[ref_array[:,5] == 11]
    ncol11 = nsegment11[:,0].reshape(nsegment11.shape[0],1); rcol11 = rsegment11[:,0].reshape(1, rsegment11.shape[0]); mse_col11 = 1 - np.mean((ncol11 - rcol11) ** 2)
    nx11 = nsegment11[:,1].reshape(nsegment11.shape[0],1); rx11 = rsegment11[:,1].reshape(1, rsegment11.shape[0]); mse_x11 = 1 - np.mean((nx11 - rx11) ** 2)
    ny11 = nsegment11[:,2].reshape(nsegment11.shape[0],1); ry11 = rsegment11[:,2].reshape(1, rsegment11.shape[0]); mse_y11 = 1 - np.mean((ny11 - ry11) ** 2)
    nxc11 = nsegment11[:,3].reshape(nsegment11.shape[0],1); rxc11 = rsegment11[:,3].reshape(1, rsegment11.shape[0]); mse_xc11 = 1 - np.mean((nxc11 - rxc11) ** 2)
    nyc11 = nsegment11[:,4].reshape(nsegment11.shape[0],1); ryc11 = rsegment11[:,4].reshape(1, rsegment11.shape[0]); mse_yc11 = 1 - np.mean((nyc11 - ryc11) ** 2)
    cond11 = [(mse_col11 >= 0.4) & (mse_x11 >= 0.8) & (mse_y11 >= 0.8) & (mse_xc11 >= 0.7) & (mse_yc11 >= 0.7)]
    segment_similar11 = np.select(cond11, choice, 0)
    segment11_res = np.resize(np.array([mse_col11, mse_x11, mse_y11, mse_xc11, mse_yc11]), (nsegment11.shape[0],5))
    segment11 = np.hstack((nsegment11, segment11_res))

    nsegment12 = new_patom[new_patom[:,5] == 12]; rsegment12 = ref_array[ref_array[:,5] == 12]
    ncol12 = nsegment12[:,0].reshape(nsegment12.shape[0],1); rcol12 = rsegment12[:,0].reshape(1, rsegment12.shape[0]); mse_col12 = 1 - np.mean((ncol12 - rcol12) ** 2)
    nx12 = nsegment12[:,1].reshape(nsegment12.shape[0],1); rx12 = rsegment12[:,1].reshape(1, rsegment12.shape[0]); mse_x12 = 1 - np.mean((nx12 - rx12) ** 2)
    ny12 = nsegment12[:,2].reshape(nsegment12.shape[0],1); ry12 = rsegment12[:,2].reshape(1, rsegment12.shape[0]); mse_y12 = 1 - np.mean((ny12 - ry12) ** 2)
    nxc12 = nsegment12[:,3].reshape(nsegment12.shape[0],1); rxc12 = rsegment12[:,3].reshape(1, rsegment12.shape[0]); mse_xc12 = 1 - np.mean((nxc12 - rxc12) ** 2)
    nyc12 = nsegment12[:,4].reshape(nsegment12.shape[0],1); ryc12 = rsegment12[:,4].reshape(1, rsegment12.shape[0]); mse_yc12 = 1 - np.mean((nyc12 - ryc12) ** 2)
    cond12 = [(mse_col12 >= 0.4) & (mse_x12 >= 0.8) & (mse_y12 >= 0.8) & (mse_xc12 >= 0.7) & (mse_yc12 >= 0.7)]
    segment_similar12 = np.select(cond12, choice, 0)
    segment12_res = np.resize(np.array([mse_col12, mse_x12, mse_y12, mse_xc12, mse_yc12]), (nsegment12.shape[0],5))
    segment12 = np.hstack((nsegment12, segment12_res))

    nsegment13 = new_patom[new_patom[:,5] == 13]; rsegment13 = ref_array[ref_array[:,5] == 13]
    ncol13 = nsegment13[:,0].reshape(nsegment13.shape[0],1); rcol13 = rsegment13[:,0].reshape(1, rsegment13.shape[0]); mse_col13 = 1 - np.mean((ncol13 - rcol13) ** 2)
    nx13 = nsegment13[:,1].reshape(nsegment13.shape[0],1); rx13 = rsegment13[:,1].reshape(1, rsegment13.shape[0]); mse_x13 = 1 - np.mean((nx13 - rx13) ** 2)
    ny13 = nsegment13[:,2].reshape(nsegment13.shape[0],1); ry13 = rsegment13[:,2].reshape(1, rsegment13.shape[0]); mse_y13 = 1 - np.mean((ny13 - ry13) ** 2)
    nxc13 = nsegment13[:,3].reshape(nsegment13.shape[0],1); rxc13 = rsegment13[:,3].reshape(1, rsegment13.shape[0]); mse_xc13 = 1 - np.mean((nxc13 - rxc13) ** 2)
    nyc13 = nsegment13[:,4].reshape(nsegment13.shape[0],1); ryc13 = rsegment13[:,4].reshape(1, rsegment13.shape[0]); mse_yc13 = 1 - np.mean((nyc13 - ryc13) ** 2)
    cond13 = [(mse_col13 >= 0.4) & (mse_x13 >= 0.8) & (mse_y13 >= 0.8) & (mse_xc13 >= 0.7) & (mse_yc13 >= 0.7)]
    segment_similar13 = np.select(cond13, choice, 0)
    segment13_res = np.resize(np.array([mse_col13, mse_x13, mse_y13, mse_xc13, mse_yc13]), (nsegment13.shape[0],5))
    segment13 = np.hstack((nsegment13, segment13_res))

    nsegment14 = new_patom[new_patom[:,5] == 14]; rsegment14 = ref_array[ref_array[:,5] == 14]
    ncol14 = nsegment14[:,0].reshape(nsegment14.shape[0],1); rcol14 = rsegment14[:,0].reshape(1, rsegment14.shape[0]); mse_col14 = 1 - np.mean((ncol14 - rcol14) ** 14)
    nx14 = nsegment14[:,1].reshape(nsegment14.shape[0],1); rx14 = rsegment14[:,1].reshape(1, rsegment14.shape[0]); mse_x14 = 1 - np.mean((nx14 - rx14) ** 14)
    ny14 = nsegment14[:,2].reshape(nsegment14.shape[0],1); ry14 = rsegment14[:,2].reshape(1, rsegment14.shape[0]); mse_y14 = 1 - np.mean((ny14 - ry14) ** 14)
    nxc14 = nsegment14[:,3].reshape(nsegment14.shape[0],1); rxc14 = rsegment14[:,3].reshape(1, rsegment14.shape[0]); mse_xc14 = 1 - np.mean((nxc14 - rxc14) ** 14)
    nyc14 = nsegment14[:,4].reshape(nsegment14.shape[0],1); ryc14 = rsegment14[:,4].reshape(1, rsegment14.shape[0]); mse_yc14 = 1 - np.mean((nyc14 - ryc14) ** 14)
    cond14 = [(mse_col14 >= 0.4) & (mse_x14 >= 0.8) & (mse_y14 >= 0.8) & (mse_xc14 >= 0.7) & (mse_yc14 >= 0.7)]
    segment_similar14 = np.select(cond14, choice, 0)
    segment14_res = np.resize(np.array([mse_col14, mse_x14, mse_y14, mse_xc14, mse_yc14]), (nsegment14.shape[0],5))
    segment14 = np.hstack((nsegment14, segment14_res))

    nsegment15 = new_patom[new_patom[:,5] == 15]; rsegment15 = ref_array[ref_array[:,5] == 15]
    ncol15 = nsegment15[:,0].reshape(nsegment15.shape[0],1); rcol15 = rsegment15[:,0].reshape(1, rsegment15.shape[0]); mse_col15 = 1 - np.mean((ncol15 - rcol15) ** 2)
    nx15 = nsegment15[:,1].reshape(nsegment15.shape[0],1); rx15 = rsegment15[:,1].reshape(1, rsegment15.shape[0]); mse_x15 = 1 - np.mean((nx15 - rx15) ** 2)
    ny15 = nsegment15[:,2].reshape(nsegment15.shape[0],1); ry15 = rsegment15[:,2].reshape(1, rsegment15.shape[0]); mse_y15 = 1 - np.mean((ny15 - ry15) ** 2)
    nxc15 = nsegment15[:,3].reshape(nsegment15.shape[0],1); rxc15 = rsegment15[:,3].reshape(1, rsegment15.shape[0]); mse_xc15 = 1 - np.mean((nxc15 - rxc15) ** 2)
    nyc15 = nsegment15[:,4].reshape(nsegment15.shape[0],1); ryc15 = rsegment15[:,4].reshape(1, rsegment15.shape[0]); mse_yc15 = 1 - np.mean((nyc15 - ryc15) ** 2)
    cond15 = [(mse_col15 >= 0.4) & (mse_x15 >= 0.8) & (mse_y15 >= 0.8) & (mse_xc15 >= 0.7) & (mse_yc15 >= 0.7)]
    segment_similar15 = np.select(cond15, choice, 0)
    segment15_res = np.resize(np.array([mse_col15, mse_x15, mse_y15, mse_xc15, mse_yc15]), (nsegment15.shape[0],5))
    segment15 = np.hstack((nsegment15, segment15_res))

    nsegment16 = new_patom[new_patom[:,5] == 16]; rsegment16 = ref_array[ref_array[:,5] == 16]
    ncol16 = nsegment16[:,0].reshape(nsegment16.shape[0],1); rcol16 = rsegment16[:,0].reshape(1,rsegment16.shape[0]); mse_col16 = 1 - np.mean((ncol16 - rcol16) ** 2)
    nx16 = nsegment16[:,1].reshape(nsegment16.shape[0],1); rx16 = rsegment16[:,1].reshape(1,rsegment16.shape[0]); mse_x16 = 1 - np.mean((nx16 - rx16) ** 2)
    ny16 = nsegment16[:,2].reshape(nsegment16.shape[0],1); ry16 = rsegment16[:,2].reshape(1,rsegment16.shape[0]); mse_y16 = 1 - np.mean((ny16 - ry16) ** 2)
    nxc16 = nsegment16[:,3].reshape(nsegment16.shape[0],1); rxc16 = rsegment16[:,3].reshape(1,rsegment16.shape[0]); mse_xc16 = 1 - np.mean((nxc16 - rxc16) ** 2)
    nyc16 = nsegment16[:,4].reshape(nsegment16.shape[0],1); ryc16 = rsegment16[:,4].reshape(1,rsegment16.shape[0]); mse_yc16 = 1 - np.mean((nyc16 - ryc16) ** 2)
    cond16 = [(mse_col16 >= 0.4) & (mse_x16 >= 0.8) & (mse_y16 >= 0.8) & (mse_xc16 >= 0.7) & (mse_yc16 >= 0.7)]
    segment_similar16 = np.select(cond16, choice, 0)
    segment16_res = np.resize(np.array([mse_col16, mse_x16, mse_y16, mse_xc16, mse_yc16]), (nsegment16.shape[0],5))
    segment16 = np.hstack((nsegment16, segment16_res))
    
    new_patom_sim = np.vstack((segment1, segment2, segment3, segment4, segment5, segment6, segment7, segment8, segment9, segment10,\
                               segment11, segment12, segment13, segment14, segment15, segment16))
    patom_sim = np.resize(np.array([segment_similar1 + segment_similar2 + segment_similar3 + segment_similar4 + segment_similar5 + \
                                    segment_similar6 + segment_similar7 + segment_similar8 + segment_similar9 + segment_similar10 + \
                                    segment_similar11 + segment_similar12 + segment_similar13 + segment_similar14 + segment_similar15 + \
                                    segment_similar16]), (new_patom_sim.shape[0],1))
    new_patom_sim = np.unique(np.column_stack((new_patom_sim, patom_sim)), axis=0)

    return new_patom_sim