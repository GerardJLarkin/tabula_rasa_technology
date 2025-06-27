from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

image_path = 'red_circle.png'

image = Image.open(image_path)

image_array = np.array(image)[:,:,3]

print('Image array shape:', image_array.shape)

top = 900
bottom = 10
left = 10
right = 900

padded_arrays = []
for i in range(0, 900, 6):
    pad_array = np.pad(image_array, ((top-i, bottom+i), (left+i, right-i)), 'constant', constant_values=(0)) # top, bottom, left, right
    padded_arrays.append(pad_array)

print(len(padded_arrays))