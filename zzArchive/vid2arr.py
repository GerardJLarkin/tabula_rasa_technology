import cv2
import numpy as np
from time import perf_counter
import datetime 

path = "/home/gerard/Desktop/capstone_project/low_res_vid.mp4"
cap = cv2.VideoCapture(path)
ret = True

frames1 = []
while ret:
    ret, img = cap.read() # read one frame from the 'capture' object; img is (H, W, C)
    if ret:
        resized_img = img[:720,:1280,:]
        frames1.append(resized_img)

frames2 = frames1
frames3 = frames1
frames4 = frames1
frames5 = frames1
frames = frames1+frames2+frames3+frames4+frames5
video = np.stack(frames, axis=0) # dimensions (T, H, W, C)
data = video[:900,:,:,:]

s = perf_counter()
# colour_arrays = []
# for i in range(10):
#     strt_ind = ((i*30)-30)+30
#     end_ind = (i*30)+30
#     array1 = data[strt_ind:end_ind,:,:,0].reshape(1,-1)[0].tolist()
#     arr1 = [str(x) for x in array1]
#     array2 = data[strt_ind:end_ind,:,:,1].reshape(1,-1)[0].tolist()
#     arr2 = [str(x) for x in array2]
#     array3 = data[strt_ind:end_ind,:,:,2].reshape(1,-1)[0].tolist()
#     arr3 = [str(x) for x in array3]

#     colour_array = []
#     for i in range(len(arr1)):
#         new_element = arr1[i] + arr2[i].zfill(3) + arr3[i].zfill(3)
#         colour_array.append(new_element)

#     array = np.array([float(i)/(16777216.00+float(i)) for i in colour_array]).reshape(30, 720, 1280)
#     arr_norm = (array-np.min(array))/(np.max(array)-np.min(array))

#     colour_arrays.append(arr_norm)

colour_arrays = []
for i in range(30):
    array1 = data[i,:,:,0].reshape(1,-1)[0].tolist()
    arr1 = [str(x) for x in array1]
    array2 = data[i,:,:,1].reshape(1,-1)[0].tolist()
    arr2 = [str(x).zfill(3) for x in array2]
    array3 = data[i,:,:,2].reshape(1,-1)[0].tolist()
    arr3 = [str(x).zfill(3) for x in array3]
    
    colour_array = []
    for i in range(len(arr1)):
        new_element = arr1[i] + arr2[i] + arr3[i]
        colour_array.append(new_element)
    
    array = np.array([float(i)/(16777216.00+float(i)) for i in colour_array]).reshape(720, 1280)
    arr_norm = (array-np.min(array))/(np.max(array)-np.min(array))
    colour_arrays.append(arr_norm)


final_array = np.stack(colour_arrays, axis=0)
print(final_array.shape)
e = perf_counter()
np.save('/home/gerard/Desktop/capstone_project/norm_file.npy', final_array)
print('Time taken in seconds:', e-s)
