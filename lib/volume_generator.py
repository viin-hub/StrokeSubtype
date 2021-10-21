import nibabel as nib
import warnings
warnings.filterwarnings('ignore')
import sys
import math
import random
import numpy as np
from skimage.transform import resize
import os
from os import listdir
from os.path import isfile, join


len_border=20
len_border_z=15
len_depth=3
input_rows = 64
input_cols = 64
input_deps = 32
crop_rows = 64
crop_cols = 64
scale = 32

def infinite_generator_from_one_volume(img_array):
    size_x, size_y, size_z = img_array.shape
    if size_z-input_deps-len_depth-1-len_border_z < len_border_z:
        return None
    
    slice_set = np.zeros((scale, input_rows, input_cols, input_deps), dtype=float)
    
    num_pair = 0
    cnt = 0
    while True:
        cnt += 1
        if cnt > 50 * scale and num_pair == 0:
            return None
        elif cnt > 50 * scale and num_pair > 0:
            return np.array(slice_set[:num_pair])

        start_x = random.randint(0+len_border, size_x-crop_rows-1-len_border)
        start_y = random.randint(0+len_border, size_y-crop_cols-1-len_border)
        start_z = random.randint(0+len_border_z, size_z-input_deps-len_depth-1-len_border_z)
        
        crop_window = img_array[start_x : start_x+crop_rows,
                                start_y : start_y+crop_cols,
                                start_z : start_z+input_deps+len_depth,
                               ]
        if crop_rows != input_rows or crop_cols != input_cols:
            crop_window = resize(crop_window, 
                                 (input_rows, input_cols, input_deps+len_depth), 
                                 preserve_range=True,
                                )
        
        t_img = np.zeros((input_rows, input_cols, input_deps), dtype=float)
        d_img = np.zeros((input_rows, input_cols, input_deps), dtype=float)
        
        for d in range(input_deps):
            for i in range(input_rows):
                for j in range(input_cols):
                    for k in range(len_depth):
                        t_img[i, j, d] = crop_window[i, j, d+k]
                        d_img[i, j, d] = k

                            
        d_img = d_img.astype('float32')
        d_img /= (len_depth - 1)
        d_img = 1.0 - d_img
        
        # if np.sum(d_img) > lung_max * input_rows * input_cols * input_deps:
        #     continue
        
        slice_set[num_pair] = crop_window[:,:,:input_deps]
        
        num_pair += 1
        if num_pair == scale:
            break

        tar_folder = '/home/miranda/Documents/code/3DCNN/generated_cubes'
        slice_set = np.zeros((scale, input_rows, input_cols, input_deps), dtype=float)
        for i in range(scale):
            temp = slice_set[i, :, :, :]
            np.save(os.path.join(tar_folder, iid+"_%s.npy"%i), temp)
            
    return np.array(slice_set)

data_path = '/home/miranda/Documents/code/3DCNN/data_raw'
niifiles = [join(data_path, f) for f in listdir(data_path) if isfile(join(data_path, f))]

for f in niifiles:
    print(f)
    scan = nib.load(f)
    base = os.path.basename(f)
    iid = os.path.splitext(os.path.splitext(base)[0])[0]
    data = scan.get_fdata()
    arr = infinite_generator_from_one_volume(data)
