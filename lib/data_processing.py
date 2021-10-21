import os
import pandas as pd
from pathlib import Path
from shutil import copyfile
import nibabel as nib
from nilearn.image import resample_img
import pydicom
import numpy as np
from scipy import ndimage
import glob
import random
import matplotlib.pyplot as plt
from nilearn.masking import apply_mask
import operator
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from scipy import ndimage
from skimage import morphology
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn import decomposition
from time import time
from numpy.random import RandomState
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV,RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import multilabel_confusion_matrix
import collections
import shutil


def sub_groups():

	dic = pd.read_excel('/home/miranda/Documents/data/INSPIRE/stats/Miranda project INSPIRE pts.xlsx', header=[3], sheet_name=None)
	df_info = pd.concat(dic.values(), axis=0)

	df = pd.read_csv('/home/miranda/Documents/data/INSPIRE/subtype/sub_ct_bl/ncct_brain.csv')
	y = []
	x = []
	for index, row in df.iterrows():
		f = row.tolist()[0]
		elements = f.split("/")
		pid = elements[-2]
		e2 = pid.split("_")
		iid = "_".join(e2[0:2])
		s = df_info.loc[df_info['INSPIRE ID'] == iid]
		mech = s['Stroke Mechanism'].tolist()[0]
		if mech == 'Cardioembolic':
			y.append(0)
			x.append(f)
		elif mech == 'Large Artery Atherosclerosis':
			y.append(1)
			x.append(f)
		elif mech == 'Small vessel (lacunar)':
			y.append(2)
			x.append(f)
		# fid = elements[8]
		# e2 = fid.split("_")
		# iid = "_".join(e2[0:2])
		# s = df_info.loc[df_info['INSPIRE ID'] == iid]
		# label = s['Stroke Mechanism'].tolist()[0]
		# f = Path(f)
		# if f.is_file():
		# 	if label == 'Cardioembolic': # 67
		# 		copyfile(f,os.path.join('/home/miranda/Documents/code/3DCNN/SSubtype/CE_vessel/',iid+'_'+elements[-1]))
		# 	elif label == 'Large Artery Atherosclerosis': # 36
		# 		copyfile(f,os.path.join('/home/miranda/Documents/code/3DCNN/SSubtype/LAA_vessel/',iid+'_'+elements[-1]))
		# 	elif label == 'Small vessel (lacunar)': # 25
		# 		copyfile(f,os.path.join('/home/miranda/Documents/code/3DCNN/SSubtype/SV_vessel/',iid+'_'+elements[-1]))
	
	d = {'IMG': x,'Label':y} 
	df_img = pd.DataFrame(data=d)
	df_img.to_csv('/home/miranda/Documents/data/INSPIRE/subtype/sub_ct_bl/ncct_brain_y.csv',index=False)

def demographics():

	df_info = pd.read_csv('/home/miranda/Documents/data/INSPIRE/stats/IMG_info.csv') 
	dic = pd.read_excel('/home/miranda/Documents/data/INSPIRE/stats/Miranda project INSPIRE pts.xlsx', header=[3], sheet_name=None)
	df_ref = pd.concat(dic.values(), axis=0)

	df = pd.read_csv('/home/miranda/Documents/data/INSPIRE/subtype/CT_BL/brain_files.csv')

	
	list_id = []
	list_sex = []
	list_age = []
	list_label = []
	list_fpath = []
	for index, row in df.iterrows():
		f = row.tolist()[0]
		elements = f.split("/")
		fid = elements[8]
		e2 = fid.split("_")
		iid = "_".join(e2[0:2]) 
		sl = df_ref.loc[df_ref['INSPIRE ID'] == iid]
		label = sl['Stroke Mechanism'].tolist()[0]
		s = df_info.loc[df_info['ID'] == iid]
		s = s.drop_duplicates(subset ="ID", keep = 'first')
		sex = s['Sex'].tolist()[0]
		age = s['Age'].tolist()[0]
		res_age = age.replace('Y', '') 
		res_age = res_age.replace('0', '')
		res_age = int(res_age)	

		list_id.append(iid)
		list_sex.append(sex)
		list_age.append(res_age)
		list_label.append(label)
		list_fpath.append(f)

	d = {'ID': list_id,'Age':list_age, 'Sex':list_sex,'Mechanism':list_label,'Path':list_fpath} 
	df_img = pd.DataFrame(data=d)
	df_img.to_csv('/home/miranda/Documents/data/INSPIRE/subtype/CT_BL/brain_files_info.csv',index=False)

	print('Sex:', df_img['Sex'].value_counts())
	print('Mean Age:', df_img['Age'].mean())
	print('Std Age:', df_img['Age'].std())

	df_img_laa = df_img.loc[df_img['Mechanism'] == 'Large Artery Atherosclerosis']
	df_img_ce = df_img.loc[df_img['Mechanism'] == 'Cardioembolic']
	df_img_sv = df_img.loc[df_img['Mechanism'] == 'Small vessel (lacunar)']
	print('-------------------')
	print('---LAA-------------')
	print('-------------------')
	print('-------------------')
	print('Sex:', df_img_laa['Sex'].value_counts())
	print('Mean Age:', df_img_laa['Age'].mean())
	print('Std Age:', df_img_laa['Age'].std())

	print('-------------------')
	print('----CE-------------')
	print('-------------------')
	print('-------------------')
	print('Sex:', df_img_ce['Sex'].value_counts())
	print('Mean Age:', df_img_ce['Age'].mean())
	print('Std Age:', df_img_ce['Age'].std())

	print('-------------------')
	print('------SV-----------')
	print('-------------------')
	print('-------------------')
	print('Sex:', df_img_sv['Sex'].value_counts())
	print('Mean Age:', df_img_sv['Age'].mean())
	print('Std Age:', df_img_sv['Age'].std())

def normalization(ilist):

	df = pd.read_csv(ilist)

	for index, row in df.iterrows():
		img = row.tolist()[0]
		elements = img.split('/')
		name = elements[-1]
		# print(name)
		basename = name.replace('.nii.gz', '') 
		folder = '/'.join(elements[:-1])
		# print(folder)
		ni = nib.load(img)
		data = ni.get_fdata()
		norm_mean = np.mean(data)
		norm_std = np.std(data)
		normalized_img = (data - norm_mean) / (1.0 * norm_std)
		new_image = nib.Nifti1Image(normalized_img, affine=ni.affine)
		nib.save(new_image, os.path.join(folder,basename+'_'+'norm'+'.nii.gz'))

def viz_check():

	# list_ce = glob.glob('/home/miranda/Documents/code/3DCNN/SSubtype/CE_brain/*.nii.gz',recursive=True)
	# list_laa = glob.glob('/home/miranda/Documents/code/3DCNN/SSubtype/LAA_brain/*.nii.gz',recursive=True)
	# list_sv = glob.glob('/home/miranda/Documents/code/3DCNN/SSubtype/SV_brain/*.nii.gz',recursive=True)

	df = pd.read_csv('/home/miranda/Documents/code/3DCNN/data_norm/data_crop_info.csv')
	list_sv = df['path'].tolist()
	for f in list_sv:
		scan = nib.load(f)
		# Get raw data
		scan = scan.get_fdata()
		plt.imshow(scan[100, :, :], cmap="gray")
		# plt.show()
		elements = f.split('/')
		name = elements[-1]
		basename = name.replace('.nii.gz', '') 
		plt.savefig('/home/miranda/Documents/code/3DCNN/data_norm/pic/'+basename+'.png')

def attention_map():

	template = nib.load('/home/miranda/Documents/data/VesselAtlas/vesselProbabilities1mm.nii.gz')
	data = template.get_fdata()
	data[data > 0] = 1
	new_image = nib.Nifti1Image(data, affine=template.affine)
	nib.save(new_image, '/home/miranda/Documents/code/3DCNN/SSubtype/vessel_binary.nii.gz')

	# scan = nib.load('/home/miranda/Documents/code/3DCNN/SSubtype/CE_bs/INSP_AU010283_CT_23_Head_2715175211_Brain_Helical_Trauma_20170424163057_23.ni_NCCT_scaled_brain_norm.nii.gz')
	# arr = scan.get_fdata()

	# print(arr.shape)
	# print(data.shape)
	# C = np.matmul(arr, data)
	# image = nib.Nifti1Image(C, affine=scan.affine)
	# nib.save(image, '/home/miranda/Documents/code/3DCNN/SSubtype/example.nii.gz')

def rescale_affine(input_affine, voxel_dims=[1, 1, 1], target_center_coords= None):
	"""
	This function uses a generic approach to rescaling an affine to arbitrary
	voxel dimensions. It allows for affines with off-diagonal elements by
	decomposing the affine matrix into u,s,v (or rather the numpy equivalents)
	and applying the scaling to the scaling matrix (s).

	Parameters
	----------
	input_affine : np.array of shape 4,4
	    Result of nibabel.nifti1.Nifti1Image.affine
	voxel_dims : list
	    Length in mm for x,y, and z dimensions of each voxel.
	target_center_coords: list of float
	    3 numbers to specify the translation part of the affine if not using the same as the input_affine.

	Returns
	-------
	target_affine : 4x4matrix
	    The resampled image.
	"""
	# Initialize target_affine
	target_affine = input_affine.copy()
	# Decompose the image affine to allow scaling
	u,s,v = np.linalg.svd(target_affine[:3,:3],full_matrices=False)

	# Rescale the image to the appropriate voxel dimensions
	s = voxel_dims

	# Reconstruct the affine
	target_affine[:3,:3] = u @ np.diag(s) @ v

	# Set the translation component of the affine computed from the input
	# image affine if coordinates are specified by the user.
	if target_center_coords is not None:
		target_affine[:3,3] = target_center_coords
	return target_affine


def resample_template():

	# resample vessle atlas with the same size of MNI152 182*218*182
	orig = nib.load('/home/miranda/Documents/data/VesselAtlas/vesselProbabilities.nii.gz')
	affine = orig.affine
	trg_affine = rescale_affine(affine)

	img = resample_img(orig, target_affine=trg_affine, target_shape=[182,218,182])
	nib.save(img, '/home/miranda/Documents/data/VesselAtlas/vesselProbabilities1mm.nii.gz')  
	# img = orig.__class__(np.array(orig.dataobj),
	# 					rescale_affine(orig.affine, 0.5),
	# 					orig.header)
	# img.to_filename('/home/miranda/Documents/data/VesselAtlas/vesselProbabilities1mm.nii.gz')

def vessel_data():
	df = pd.read_csv('/home/miranda/Documents/data/INSPIRE/subtype/CT_BL/f_ncct_thr_sub_2.csv')

	mask_img = nib.load('/home/miranda/Documents/data/VesselAtlas/vesselProbabilities1mm.nii.gz')
	mask = mask_img.get_fdata()

	paths = []
	for index, row in df.iterrows():
		f = row.tolist()[0]
		scan = nib.load(f)
		affine =  scan.affine
		# Get raw data
		scan = scan.get_fdata()
		shape = scan.shape
		elements = f.split('/')
		name = elements[-1]
		basename = name.replace('.nii.gz', '') 
		folder = '/'.join(elements[:-1])
		if shape != (182,218,182):
			print(f)
		else:
			C = np.zeros((182,218,182))
			for i in range(182):
				C[i] = np.multiply(scan[i], mask[i])
			C = np.array(C)
			new_image = nib.Nifti1Image(C, affine=affine)
			nib.save(new_image, os.path.join(folder,basename+'_'+'vessel'+'.nii.gz'))
			paths.append(os.path.join(folder,basename+'_'+'vessel'+'.nii.gz'))

	df_img = pd.DataFrame(paths)
	df_img.to_csv('/home/miranda/Documents/data/INSPIRE/subtype/CT_BL/f_ncct_thr_sub_2_vessel.csv',index=False)

def read_xlsx(file):
	dic = pd.read_html(file)
	df_info = pd.concat(dic.values(), axis=0)
	print(df_info)

def remove_badscan():

	ce_scan_paths = [
	    os.path.join('/home/miranda/Documents/code/3DCNN/', "SSubtype/CE", x)
	    for x in os.listdir("SSubtype/CE")
	]

	laa_scan_paths = [
	    os.path.join('/home/miranda/Documents/code/3DCNN/', "SSubtype/LAA", x)
	    for x in os.listdir("SSubtype/LAA")
	]

	sv_scan_paths = [
	    os.path.join('/home/miranda/Documents/code/3DCNN/', "SSubtype/SV", x)
	    for x in os.listdir("SSubtype/SV")
	]

	ce_scan_paths_d = [
	    os.path.join('/home/miranda/Documents/code/3DCNN/', "SSubtype/vessel/CE", x)
	    for x in os.listdir("SSubtype/vessel/CE")
	]

	laa_scan_paths_d = [
	    os.path.join('/home/miranda/Documents/code/3DCNN/', "SSubtype/vessel/LAA", x)
	    for x in os.listdir("SSubtype/vessel/LAA")
	]

	sv_scan_paths_d = [
	    os.path.join('/home/miranda/Documents/code/3DCNN/', "SSubtype/vessel/SV", x)
	    for x in os.listdir("SSubtype/vessel/SV")
	]

	merged_list = ce_scan_paths + laa_scan_paths + sv_scan_paths + ce_scan_paths_d +\
	laa_scan_paths_d + sv_scan_paths_d

	flist = []
	for f in merged_list:
		scan = nib.load(f)
		# Get raw data
		scan = scan.get_fdata()
		if np.any(np.isnan(scan)) == True:
			flist.append(f)
	df = pd.DataFrame(flist)
	df.to_csv('/home/miranda/Documents/code/3DCNN/SSubtype/badScan.csv', index=False)

def read_nifti_file(filepath):
	"""Read and load volume"""
	# Read file
	scan = nib.load(filepath)
	affine=scan.affine
	# Get raw data
	scan = scan.get_fdata()
	
		
	return scan, affine


def normalize(volume):
	"""Normalize the volume"""


	min = 0
	max = 100
	volume[volume < min] = min
	volume[volume > max] = max
	volume = (volume - min) / (max - min)

	# volume[volume <= min] = 0
	# volume = (volume - min) / (max - min)
	# volume[volume < 0] = 0
	# volume[volume > 1] = 1
	
	volume = volume.astype("float32")
	return volume


def resize_volume(img, desired_depth, desired_width, desired_height):
	"""Resize across z-axis"""
	# Set the desired depth
	# desired_depth = 128
	# desired_width = 128
	# desired_height = 128
	# Get current depth
	current_depth = img.shape[-1]
	current_width = img.shape[0]
	current_height = img.shape[1]
	# Compute depth factor
	depth = current_depth / desired_depth
	width = current_width / desired_width
	height = current_height / desired_height
	depth_factor = 1 / depth
	width_factor = 1 / width
	height_factor = 1 / height
	# Rotate
	img = ndimage.rotate(img, 90, reshape=False)
	# Resize across z-axis
	img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
	return img


def process_scan(path):
	"""Read and resize volume"""
	# Read scan
	volume, affine = read_nifti_file(path)
	# Normalize
	volume = normalize(volume)
	# Resize width, height and depth
	# volume = resize_volume(volume)
	return volume, affine

def crop_image(image, display=False):
	# Create a mask with the background pixels
	mask = image == 0

	# Find the brain area
	coords = np.array(np.nonzero(~mask))
	top_left = np.min(coords, axis=1) 
	# print('coords', coords)
	bottom_right = np.max(coords, axis=1) 
	# print('top_left', top_left)
	# print('bottom_right', bottom_right)

	# Remove the background
	croped_image = image[top_left[0]+17:bottom_right[0]-15, top_left[1]+17:bottom_right[1]-15, top_left[2]+35:bottom_right[2]-15]

	return croped_image


def prepare_data():
	# load data 
	ce_scan_paths = [
	    os.path.join('/home/miranda/Documents/code/3DCNN/', "SSubtype/CE", x)
	    for x in os.listdir("/home/miranda/Documents/code/3DCNN/SSubtype/CE")
	]

	laa_scan_paths = [
	    os.path.join('/home/miranda/Documents/code/3DCNN/', "SSubtype/LAA", x)
	    for x in os.listdir("/home/miranda/Documents/code/3DCNN/SSubtype/LAA")
	]

	sv_scan_paths = [
	    os.path.join('/home/miranda/Documents/code/3DCNN/', "SSubtype/SV", x)
	    for x in os.listdir("/home/miranda/Documents/code/3DCNN/SSubtype/SV")
	]

	print("CT scans with CE: " + str(len(ce_scan_paths)))
	print("CT scans with LAA: " + str(len(laa_scan_paths)))
	print("CT scans with SV: " + str(len(sv_scan_paths)))

	# Read and process the scans.
	# Each scan is resized across height, width, and depth and rescaled.


	f = []
	label = []
	f_crop = []
	for path in ce_scan_paths:
		volume, affine = process_scan(path)
		vec = volume.reshape(-1)
		elements = path.split('/')
		name = elements[-1]
		new_image = nib.Nifti1Image(volume, affine=affine)
		nib.save(new_image, os.path.join('/home/miranda/Documents/code/3DCNN/data_norm/',name))
		f.append(os.path.join('/home/miranda/Documents/code/3DCNN/data_norm/',name))
		label.append(0)
		c_img = crop_image(volume, display=False)
		new_image2 = nib.Nifti1Image(c_img, affine=affine)
		print(os.path.join('/home/miranda/Documents/code/3DCNN/data_norm/', 'crop_'+name))
		nib.save(new_image2, os.path.join('/home/miranda/Documents/code/3DCNN/data_norm/', 'crop_'+name))
		f_crop.append(os.path.join('/home/miranda/Documents/code/3DCNN/data_norm/', 'crop_'+name))

	for path in laa_scan_paths:
		volume, affine = process_scan(path)
		elements = path.split('/')
		name = elements[-1]
		new_image = nib.Nifti1Image(volume, affine=affine)
		nib.save(new_image, os.path.join('/home/miranda/Documents/code/3DCNN/data_norm/',name))
		f.append(os.path.join('/home/miranda/Documents/code/3DCNN/data_norm/',name))
		label.append(1)
		c_img = crop_image(volume, display=False)
		new_image2 = nib.Nifti1Image(c_img, affine=affine)
		print(os.path.join('/home/miranda/Documents/code/3DCNN/data_norm/', 'crop_'+name))
		nib.save(new_image2, os.path.join('/home/miranda/Documents/code/3DCNN/data_norm/', 'crop_'+name))
		f_crop.append(os.path.join('/home/miranda/Documents/code/3DCNN/data_norm/', 'crop_'+name))

	for path in sv_scan_paths:
		volume, affine = process_scan(path)
		elements = path.split('/')
		name = elements[-1]
		new_image = nib.Nifti1Image(volume, affine=affine)
		nib.save(new_image, os.path.join('/home/miranda/Documents/code/3DCNN/data_norm/',name))
		f.append(os.path.join('/home/miranda/Documents/code/3DCNN/data_norm/',name))
		label.append(2)
		c_img = crop_image(volume, display=False)
		new_image2 = nib.Nifti1Image(c_img, affine=affine)
		print(os.path.join('/home/miranda/Documents/code/3DCNN/data_norm/', 'crop_'+name))
		nib.save(new_image2, os.path.join('/home/miranda/Documents/code/3DCNN/data_norm/', 'crop_'+name))
		f_crop.append(os.path.join('/home/miranda/Documents/code/3DCNN/data_norm/', 'crop_'+name))

	d = {'path': f,'label':label} 
	df = pd.DataFrame(data=d)
	df.to_csv('/home/miranda/Documents/code/3DCNN/data_norm/data_info.csv',index=False)

	d = {'path': f_crop,'label':label} 
	df = pd.DataFrame(data=d)
	df.to_csv('/home/miranda/Documents/code/3DCNN/data_norm/data_crop_info.csv',index=False)

	# load data vessel
	ce_vscan_paths = [
	    os.path.join('/home/miranda/Documents/code/3DCNN/', "SSubtype/CE_vessel", x)
	    for x in os.listdir("/home/miranda/Documents/code/3DCNN/SSubtype/CE_vessel")
	]

	laa_vscan_paths = [
	    os.path.join('/home/miranda/Documents/code/3DCNN/', "SSubtype/LAA_vessel", x)
	    for x in os.listdir("/home/miranda/Documents/code/3DCNN/SSubtype/LAA_vessel")
	]

	sv_vscan_paths = [
	    os.path.join('/home/miranda/Documents/code/3DCNN/', "SSubtype/SV_vessel", x)
	    for x in os.listdir("/home/miranda/Documents/code/3DCNN/SSubtype/SV_vessel")
	]

	print("CT scans with CE: " + str(len(ce_vscan_paths)))
	print("CT scans with LAA: " + str(len(laa_vscan_paths)))
	print("CT scans with SV: " + str(len(sv_vscan_paths)))

	# Read and process the scans.
	# Each scan is resized across height, width, and depth and rescaled.

	f2 = []
	label2 = []
	for path in ce_vscan_paths:
		volume, affine = process_scan(path)
		elements = path.split('/')
		name = elements[-1]
		new_image = nib.Nifti1Image(volume, affine=affine)
		nib.save(new_image, os.path.join('/home/miranda/Documents/code/3DCNN/data_norm/','vessel_'+name))
		f2.append(os.path.join('/home/miranda/Documents/code/3DCNN/data_norm/','vessel_'+name))
		label2.append(0)

	for path in laa_vscan_paths:
		volume, affine = process_scan(path)
		elements = path.split('/')
		name = elements[-1]
		new_image = nib.Nifti1Image(volume, affine=affine)
		nib.save(new_image, os.path.join('/home/miranda/Documents/code/3DCNN/data_norm/','vessel_'+name))
		f2.append(os.path.join('/home/miranda/Documents/code/3DCNN/data_norm/','vessel_'+name))
		label2.append(1)

	for path in sv_vscan_paths:
		volume, affine = process_scan(path)
		elements = path.split('/')
		name = elements[-1]
		new_image = nib.Nifti1Image(volume, affine=affine)
		nib.save(new_image, os.path.join('/home/miranda/Documents/code/3DCNN/data_norm/','vessel_'+name))
		f2.append(os.path.join('/home/miranda/Documents/code/3DCNN/data_norm/','vessel_'+name))
		label2.append(2)

	d2 = {'path': f2,'label':label2} 
	df2 = pd.DataFrame(data=d2)
	df2.to_csv('/home/miranda/Documents/code/3DCNN/data_norm/data_vessel_info.csv',index=False)

def scipy_rotate(volume):
	# define some rotation angles
	angles = [-20, -10, -5, 5, 10, 20]
	# pick angles at random
	angle = random.choice(angles)
	# rotate volume
	volume = ndimage.rotate(volume, angle, reshape=False)
	volume[volume < 0] = 0
	volume[volume > 1] = 1
	return volume

def flip(volume):
	axes = [0, 1, 2]
	axis = random.choice(axes)
	volume = np.flip(volume, axis)
	return volume

def gaussian_noise(volume):
	mean = 0
	var = 0.1
	return volume + np.random.normal(mean, var, volume.shape)

def cropND(volume,bounding):
	start = tuple(map(lambda a, da: a//2-da//2, volume.shape, bounding))
	end = tuple(map(operator.add, start, bounding))
	slices = tuple(map(slice, start, end))
	return volume[slices]

def zero_mean():

	df = pd.read_csv('/home/miranda/Documents/code/3DCNN/data_norm/data_info.csv')
	df2 = pd.read_csv('/home/miranda/Documents/code/3DCNN/data_norm/data_vessel_info.csv')

	t_a = np.zeros(shape=(128,128,128))
	i = 0
	for index, row in df.iterrows():
		f = row.tolist()[0]
		scan = nib.load(f)
		data = scan.get_fdata()
		t_a = np.add(t_a,data)
		i = i + 1

	average_a = t_a/i
	average_a_flatten = average_a.reshape(-1)
	mean_pixel_ct = np.mean(average_a_flatten)
	print('CT', mean_pixel_ct) # 0.03289864781412823

	t_b = np.zeros(shape=(128,128,128))
	i = 0
	for index, row in df2.iterrows():
		f = row.tolist()[0]
		scan = nib.load(f)
		data = scan.get_fdata()
		t_b = np.add(t_b,data)
		i = i + 1

	average_b = t_b/i
	average_b_flatten = average_b.reshape(-1)
	mean_pixel_vessle = np.mean(average_b_flatten)
	print('CT vessel', mean_pixel_vessle) # 0.012433479989886424

	return mean_pixel_ct, mean_pixel_vessle

def zero_center():

	# means
	CT, CT_vessel = zero_mean()

	df = pd.read_csv('/home/miranda/Documents/code/3DCNN/data_norm/data_info.csv')
	df2 = pd.read_csv('/home/miranda/Documents/code/3DCNN/data_norm/data_vessel_info.csv')

	paths_a = []
	labels_a = []
	for index, row in df.iterrows():
		f = row.tolist()[0]
		label = row.tolist()[1]
		elements = f.split('/')
		name = elements[-1]
		basename = name.replace('.nii.gz', '') 
		folder = '/'.join(elements[:-1])
		scan = nib.load(f)
		data = scan.get_fdata()
		image = data - CT
		new_image = nib.Nifti1Image(image, affine=scan.affine)
		nib.save(new_image, os.path.join(folder,basename+'_'+'zero'+'.nii.gz'))
		paths_a.append(os.path.join(folder,basename+'_'+'zero'+'.nii.gz'))
		labels_a.append(label)

	d = {'path': paths_a,'label':labels_a} 
	df_img = pd.DataFrame(data=d)
	df_img.to_csv('/home/miranda/Documents/code/3DCNN/data_norm/data_info_0mean.csv',index=False)

	paths_b = []
	labels_b = []
	for index, row in df2.iterrows():
		f = row.tolist()[0]
		label = row.tolist()[1]
		elements = f.split('/')
		name = elements[-1]
		basename = name.replace('.nii.gz', '') 
		folder = '/'.join(elements[:-1])
		scan = nib.load(f)
		data = scan.get_fdata()
		image = data - CT_vessel
		new_image = nib.Nifti1Image(image, affine=scan.affine)
		nib.save(new_image, os.path.join(folder,basename+'_'+'zero'+'.nii.gz'))
		paths_b.append(os.path.join(folder,basename+'_'+'zero'+'.nii.gz'))
		labels_b.append(label)

	d2 = {'path': paths_b,'label':labels_b} 
	df_img2 = pd.DataFrame(data=d2)
	df_img2.to_csv('/home/miranda/Documents/code/3DCNN/data_norm/data_vessel_info_0mean.csv',index=False)


def training_data():

	df = pd.read_csv('/home/miranda/Documents/code/3DCNN/data/wm.csv')

	train, validate, test = np.split(df.sample(frac=1, random_state=42), 
								[int(.6*len(df)), int(.8*len(df))])

	train.to_csv('/home/miranda/Documents/code/3DCNN/data/wm_train.csv', index=False)
	validate.to_csv('/home/miranda/Documents/code/3DCNN/data/wm_val.csv', index=False)
	test.to_csv('/home/miranda/Documents/code/3DCNN/data/wm_test.csv', index=False)

	x_train, x_validation, x_test, y_train, y_validation, y_test = [], [], [], [], [], []


	# x_train made of resize, rotate, flip, crop
	for index, row in train.iterrows():
		path = row['path']
		label = row['label']
		# # binarize
		# if label != 0:
		# 	label = 1
		if label == 'Cardioembolic':
			y = 0
		elif label == 'Large Artery Atherosclerosis':
			y = 1
		elif label == 'Small vessel (lacunar)':
			y = 2
		volume, affine = read_nifti_file(path)
		# argument training data
		# resize
		volume1 = resize_volume(volume, 128, 128, 128)
		x_train.append(volume1)
		y_train.append(y)

		# rotation
		# volume2 = scipy_rotate(volume1)
		# x_train.append(volume2)
		# y_train.append(y)
		# # flip
		# volume3 = flip(volume1)
		# x_train.append(volume3)
		# y_train.append(y)
		# center crop
		# volume4 = resize_volume(volume, 256, 256, 256)
		# volume5 = cropND(volume, (128,128,128))
		# x_train.append(volume5)
		# y_train.append(y)


	x_train = np.asarray(x_train)
	y_train = np.asarray(y_train)



	# x_validation: original volume resize
	for index, row in validate.iterrows():
		path = row['path']
		label = row['label']
		# if label != 0:
		# 	label = 1
		if label == 'Cardioembolic':
			y = 0
		elif label == 'Large Artery Atherosclerosis':
			y = 1
		elif label == 'Small vessel (lacunar)':
			y = 2
		volume, affine = read_nifti_file(path)
		# resize
		volume1 = resize_volume(volume, 128, 128, 128)
		x_validation.append(volume1)
		y_validation.append(y)

	x_validation = np.asarray(x_validation)
	y_validation = np.asarray(y_validation)

	for index, row in test.iterrows():
		path = row['path']
		label = row['label']
		# if label != 0:
		# 	label = 1
		if label == 'Cardioembolic':
			y = 0
		elif label == 'Large Artery Atherosclerosis':
			y = 1
		elif label == 'Small vessel (lacunar)':
			y = 2
		volume, affine = read_nifti_file(path)
		# resize
		volume1 = resize_volume(volume, 128, 128, 128)
		x_test.append(volume1)
		y_test.append(y)

	x_test = np.asarray(x_test)
	y_test = np.asarray(y_test)

	# x_train, x_val, y_train, y_val = train_test_split(scans, labels, test_size=0.4, random_state=42)
	# x_validation, x_test, y_validation, y_test = train_test_split(x_val, y_val, test_size=0.5, random_state=42)
	np.save('/home/miranda/Documents/code/3DCNN/data/x_train_wm_spm.npy', x_train)
	np.save('/home/miranda/Documents/code/3DCNN/data/x_val_wm_spm.npy', x_validation)
	np.save('/home/miranda/Documents/code/3DCNN/data/x_test_wm_spm.npy', x_test)
	np.save('/home/miranda/Documents/code/3DCNN/data/y_train_wm_spm.npy', y_train)
	np.save('/home/miranda/Documents/code/3DCNN/data/y_val_wm_spm.npy', y_validation)
	np.save('/home/miranda/Documents/code/3DCNN/data/y_test_wm_spm.npy', y_test)


	# df2 = pd.read_csv('/home/miranda/Documents/code/3DCNN/data_norm/data_vessel_info_0mean.csv')

	# scans2 = []
	# labels2 = []
	# for index, row in df2.iterrows():
	# 	path = row['path']
	# 	label = row['label']
	# 	volume, affine = read_nifti_file(path)
	# 	scans2.append(volume)
	# 	labels2.append(label)
	# scans2 = np.asarray(scans2)
	# labels2 = np.asarray(labels2)

	# x_train2, x_val2, y_train2, y_val2 = train_test_split(scans2, labels2, test_size=0.4, random_state=42)
	# x_validation2, x_test2, y_validation2, y_test2 = train_test_split(x_val2, y_val2, test_size=0.5, random_state=42)
	# np.save('/home/miranda/Documents/code/3DCNN/data_norm/x_train2_0mean.npy', x_train2)
	# np.save('/home/miranda/Documents/code/3DCNN/data_norm/x_val2_0mean.npy', x_validation2)
	# np.save('/home/miranda/Documents/code/3DCNN/data_norm/x_test2_0mean.npy', x_test2)
	# np.save('/home/miranda/Documents/code/3DCNN/data_norm/y_train2_0mean.npy', y_train2)
	# np.save('/home/miranda/Documents/code/3DCNN/data_norm/y_val2_0mean.npy', y_validation2)
	# np.save('/home/miranda/Documents/code/3DCNN/data_norm/y_test2_0mean.npy', y_test2)


def data_augmentation():

	x_train = np.load('/home/miranda/Documents/code/3DCNN/data_norm/x_train_0mean.npy')
	y_train = np.load('/home/miranda/Documents/code/3DCNN/data_norm/y_train_0mean.npy')

	x_train_t = []
	y_train_t = []
	x_train_arg = []
	y_train_arg = []
	for i in range(len(y_train)):
		volume = x_train[i]	
		x_train_t.append(volume)
		label = y_train[i]
		y_train_t.append(label)
		# rotation
		augmented_volume = scipy_rotate(volume)
		x_train_arg.append(augmented_volume)
		y_train_arg.append(label)
		x_train_t.append(augmented_volume)
		y_train_t.append(label)
		# flip
		augmented_volume2 = flip(volume)
		x_train_arg.append(augmented_volume2)
		y_train_arg.append(label)
		x_train_t.append(augmented_volume2)
		y_train_t.append(label)
		# add noise
		# augmented_volume3 = gaussian_noise(volume)
		# x_train_arg.append(augmented_volume3)
		# y_train_arg.append(label)
		# x_train_t.append(augmented_volume3)
		# y_train_t.append(label)
		# center crop



	x_train_arg = np.asarray(x_train_arg)
	y_train_arg = np.asarray(y_train_arg)

	np.save('/home/miranda/Documents/code/3DCNN/data_norm/x_train_arg_0mean.npy', x_train_arg)
	np.save('/home/miranda/Documents/code/3DCNN/data_norm/y_train_arg_0mean.npy', y_train_arg)

	x_train_t = np.asarray(x_train_t)
	y_train_t = np.asarray(y_train_t)

	np.save('/home/miranda/Documents/code/3DCNN/data_norm/x_train_arg_merg_0mean.npy', x_train_t)
	np.save('/home/miranda/Documents/code/3DCNN/data_norm/y_train_arg_merg_0mean.npy', y_train_t)

	# =========================================
	# x_train2 = np.load('/home/miranda/Documents/code/3DCNN/data_norm/x_train2_0mean.npy')
	# y_train2 = np.load('/home/miranda/Documents/code/3DCNN/data_norm/y_train2_0mean.npy')

	# x_train2_arg = []
	# y_train2_arg = []
	# x_train2_t = []
	# y_train2_t = []
	# for i in range(len(y_train2)):
	# 	volume = x_train2[i]
	# 	x_train2_t.append(volume)
	# 	label = y_train2[i]
	# 	y_train2_t.append(label)
	# 	augmented_volume = scipy_rotate(volume)
	# 	x_train2_arg.append(augmented_volume)
	# 	y_train2_arg.append(label)
	# 	x_train2_t.append(augmented_volume)
	# 	y_train2_t.append(label)

	# x_train2_arg = np.asarray(x_train2_arg)
	# y_train2_arg = np.asarray(y_train2_arg)

	# np.save('/home/miranda/Documents/code/3DCNN/data_norm/x_train2_arg_0mean.npy', x_train2_arg)
	# np.save('/home/miranda/Documents/code/3DCNN/data_norm/y_train2_arg_0mean.npy', y_train2_arg)
	
	# x_train2_t = np.asarray(x_train2_t)
	# y_train2_t = np.asarray(y_train2_t)

	# np.save('/home/miranda/Documents/code/3DCNN/data_norm/x_train2_arg_merg_0mean.npy', x_train2_arg)
	# np.save('/home/miranda/Documents/code/3DCNN/data_norm/y_train2_arg_merg_0mean.npy', y_train2_arg)

	
def MosMedData():

	normal_scan_paths = [
	    os.path.join("/home/miranda/Documents/code/3DCNN", "MosMedData/CT-0", x)
	    for x in os.listdir("../MosMedData/CT-0")
	]
	abnormal_scan_paths = [
	    os.path.join("/home/miranda/Documents/code/3DCNN", "MosMedData/CT-23", x)
	    for x in os.listdir("../MosMedData/CT-23")
	]

	scans = []
	labels = []
	for x in normal_scan_paths:
		volume, affine = process_scan(x)
		scans.append(volume)
		labels.append(0)

	for x in abnormal_scan_paths:
		volume, affine = process_scan(x)
		scans.append(volume)
		labels.append(1)
	scans = np.asarray(scans)
	labels = np.asarray(labels)

	print('scans', scans.shape)
	x_train, x_val, y_train, y_val = train_test_split(scans, labels, test_size=0.4, random_state=42)
	x_validation, x_test, y_validation, y_test = train_test_split(x_val, y_val, test_size=0.5, random_state=42)
	print('x_train', x_train.shape)
	print('x_validation', x_validation.shape)

	np.save('/home/miranda/Documents/code/3DCNN/MosMeddata_norm/x_train.npy', x_train)
	np.save('/home/miranda/Documents/code/3DCNN/MosMeddata_norm/x_val.npy', x_validation)
	np.save('/home/miranda/Documents/code/3DCNN/MosMeddata_norm/x_test.npy', x_test)
	np.save('/home/miranda/Documents/code/3DCNN/MosMeddata_norm/y_train.npy', y_train)
	np.save('/home/miranda/Documents/code/3DCNN/MosMeddata_norm/y_val.npy', y_validation)
	np.save('/home/miranda/Documents/code/3DCNN/MosMeddata_norm/y_test.npy', y_test)


def clinical_info():

	dic = pd.read_excel('/home/miranda/Documents/data/INSPIRE/stats/Miranda project INSPIRE pts.xlsx', header=[3], sheet_name=None)
	df_info = pd.concat(dic.values(), axis=0)

	list_names = ['INSPIRE ID','Baseline ASPECTS',	'Baseline hypodensity volume',	'M1',	'M2',	'I',	'C',	'L',	'IC',\
		'M3',	'M4',	'M5',	'M6',	'Hypodensity outside of the occlusion? / old infarct?',	'occlusion/lesion hemishphere(0=PCA,R=1, L=2)',	\
	'Occlusion location (no visible occlusion=0,ica=1, m1=2, m2=3, m3=4, aca=5, pca=6, basilar=7, vertebral=8)', \
	'Baseline NIHSS',	'Pre-stroke mRS',	'Baseline Blood Glucose (mmol/L)', \
	'Baseline Systolic Blood Pressure (mmHg)',	'Baseline Diastolic Blood Pressure (mmHg)', \
	'Ischemic Core Volume (ml)',	'Penumbra Volume (ml)']

	# 'Occlusion Sites',	'Smoking',	'Hypertension',	'Atrial fibrillation',	'Hypercholestermia',	'Diabetes',	\
	# 'Previous diagnosis of TIA',	'Previous diagnosis of stroke',	'Congestive heart failure',	\
	# 'Family (age<65y) history of vascular disease',	'Ischaemic Heart Disease',	'Aspirin',	'Clopidogrel',	'Dipyridamole',	\
	# 'Warfarin',	'Other anti-thrombotic',	'Statin', 'Baseline total cholesterol level',	


	col_names=['INSPIREID','BaselineASPECTS','Baselinehypodensityvolume','M1','M2','I','C','L','IC','M3','M4','M5','M6',\
	'Hypodensityoutsideoftheocclusion?/oldinfarct?','occlusion/lesionhemishphere(0=PCA,R=1,L=2)',\
	'Occlusionlocation(novisibleocclusion=0,ica=1,m1=2,m2=3,m3=4,aca=5,pca=6,basilar=7,vertebral=8)',\
	'BaselineNIHSS','Pre-strokemRS','BaselineBloodGlucose(mmol/L)',\
	'BaselineSystolicBloodPressure(mmHg)','BaselineDiastolicBloodPressure(mmHg)',\
	'IschemicCoreVolume(ml)','PenumbraVolume(ml)']

	# 'OcclusionSites','Smoking','Hypertension','Atrialfibrillation','Hypercholestermia','Diabetes',\
	# 'PreviousdiagnosisofTIA','Previousdiagnosisofstroke','Congestiveheartfailure',\
	# 'Family(age<65y)historyofvasculardisease','IschaemicHeartDisease','Aspirin','Clopidogrel','Dipyridamole',\
	# 'Warfarin','Otheranti-thrombotic','Statin','Baselinetotalcholesterollevel',

	df = pd.read_csv('/home/miranda/Documents/data/INSPIRE/subtype/CT_BL/f_ncct_thr_sub_2_vessel.csv')

	data = []
	for index, row in df.iterrows():
		f = row.tolist()[0]
		elements = f.split("/")
		fid = elements[8]
		e2 = fid.split("_")
		iid = "_".join(e2[0:2])
		s = df_info.loc[df_info['INSPIRE ID'] == iid]
		columns = s[list_names].values.tolist()[0]
		# for x in columns[0]:
		# 		print(x)
		data.append(columns)

	# print(df_new)
	df_new = pd.DataFrame(data, columns=col_names)
	df_new.to_csv('/home/miranda/Documents/data/INSPIRE/subtype/CT_BL/subtype_info.csv',index=False)

def create_balanced_data():

	df = pd.read_csv('/home/miranda/Documents/code/3DCNN/data_norm/data_info.csv')
	df_ce = df.loc[df['label']==0] # CE
	df_laa = df.loc[df['label']==1] # LAA
	df_sv = df.loc[df['label']==2] # SV
	# print(df_ce) # 43
	# print(df_laa) # 28
	# print(df_sv) # 16

	scans = []
	labels = []
	f = []
	# for row in df_ce.head(30).itertuples():
	for index, row in df_ce[:30].iterrows():
		path = row['path']
		label = row['label']
		volume, affine = read_nifti_file(path)
		scans.append(volume)
		labels.append(label)
		f.append(path)

	for index, row in df_laa.iterrows():
		path = row['path']
		label = row['label']
		volume, affine = read_nifti_file(path)
		scans.append(volume)
		labels.append(label)
		f.append(path)

	for index, row in df_sv.iterrows():
		path = row['path']
		label = row['label']
		volume, affine = read_nifti_file(path)
		scans.append(volume)
		labels.append(label)
		f.append(path)
	scans = np.asarray(scans)
	labels = np.asarray(labels)

	x_train, x_val, y_train, y_val = train_test_split(scans, labels, test_size=0.4, random_state=42)
	x_validation, x_test, y_validation, y_test = train_test_split(x_val, y_val, test_size=0.5, random_state=42)
	np.save('/home/miranda/Documents/code/3DCNN/data_norm/x_train_b.npy', x_train)
	np.save('/home/miranda/Documents/code/3DCNN/data_norm/x_val_b.npy', x_validation)
	np.save('/home/miranda/Documents/code/3DCNN/data_norm/x_test_b.npy', x_test)
	np.save('/home/miranda/Documents/code/3DCNN/data_norm/y_train_b.npy', y_train)
	np.save('/home/miranda/Documents/code/3DCNN/data_norm/y_val_b.npy', y_validation)
	np.save('/home/miranda/Documents/code/3DCNN/data_norm/y_test_b.npy', y_test)



	# scans2 = []
	# labels2 = []
	# for x in f:
	# 	nf = x.replace('ni_NCCT_thr_sub.nii.gz', 'ni_NCCT_thr_sub_vessel.nii.gz')
	# 	volume, affine = read_nifti_file(path)
	# 	scans2.append(volume)
	# 	labels2.append(label)
	# scans2 = np.asarray(scans2)
	# labels2 = np.asarray(labels2)

	df2 = pd.read_csv('/home/miranda/Documents/code/3DCNN/data_norm/data_vessel_info.csv')

	scans2 = []
	labels2 = []
	for index, row in df2.iterrows():
		path = row['path']
		label = row['label']
		volume, affine = read_nifti_file(path)
		scans2.append(volume)
		labels2.append(label)
	scans2 = np.asarray(scans2)
	labels2 = np.asarray(labels2)

	x_train2, x_val2, y_train2, y_val2 = train_test_split(scans2, labels2, test_size=0.4, random_state=42)
	x_validation2, x_test2, y_validation2, y_test2 = train_test_split(x_val2, y_val2, test_size=0.5, random_state=42)
	np.save('/home/miranda/Documents/code/3DCNN/data_norm/x_train2_b.npy', x_train2)
	np.save('/home/miranda/Documents/code/3DCNN/data_norm/x_val2_b.npy', x_validation2)
	np.save('/home/miranda/Documents/code/3DCNN/data_norm/x_test2_b.npy', x_test2)
	np.save('/home/miranda/Documents/code/3DCNN/data_norm/y_train2_b.npy', y_train2)
	np.save('/home/miranda/Documents/code/3DCNN/data_norm/y_val2_b.npy', y_validation2)
	np.save('/home/miranda/Documents/code/3DCNN/data_norm/y_test2_b.npy', y_test2)


	# argumentation

	x_train = np.load('/home/miranda/Documents/code/3DCNN/data_norm/x_train_b.npy')
	y_train = np.load('/home/miranda/Documents/code/3DCNN/data_norm/y_train_b.npy')

	x_train_t = []
	y_train_t = []
	x_train_arg = []
	y_train_arg = []
	for i in range(len(y_train)):
		volume = x_train[i]	
		x_train_t.append(volume)
		label = y_train[i]
		y_train_t.append(label)
		augmented_volume = scipy_rotate(volume)
		x_train_arg.append(augmented_volume)
		y_train_arg.append(label)
		x_train_t.append(augmented_volume)
		y_train_t.append(label)


	x_train_arg = np.asarray(x_train_arg)
	y_train_arg = np.asarray(y_train_arg)

	np.save('/home/miranda/Documents/code/3DCNN/data_norm/x_train_arg_b.npy', x_train_arg)
	np.save('/home/miranda/Documents/code/3DCNN/data_norm/y_train_arg_b.npy', y_train_arg)

	x_train_t = np.asarray(x_train_t)
	y_train_t = np.asarray(y_train_t)

	np.save('/home/miranda/Documents/code/3DCNN/data_norm/x_train_arg_merg_b.npy', x_train_t)
	np.save('/home/miranda/Documents/code/3DCNN/data_norm/y_train_arg_merg_b.npy', y_train_t)

	# =========================================
	x_train2 = np.load('/home/miranda/Documents/code/3DCNN/data_norm/x_train2_b.npy')
	y_train2 = np.load('/home/miranda/Documents/code/3DCNN/data_norm/y_train2_b.npy')

	x_train2_arg = []
	y_train2_arg = []
	x_train2_t = []
	y_train2_t = []
	for i in range(len(y_train2)):
		volume = x_train2[i]
		x_train2_t.append(volume)
		label = y_train2[i]
		y_train2_t.append(label)
		augmented_volume = scipy_rotate(volume)
		x_train2_arg.append(augmented_volume)
		y_train2_arg.append(label)
		x_train2_t.append(augmented_volume)
		y_train2_t.append(label)

	x_train2_arg = np.asarray(x_train2_arg)
	y_train2_arg = np.asarray(y_train2_arg)

	np.save('/home/miranda/Documents/code/3DCNN/data_norm/x_train2_arg_b.npy', x_train2_arg)
	np.save('/home/miranda/Documents/code/3DCNN/data_norm/y_train2_arg_b.npy', y_train2_arg)
	
	x_train2_t = np.asarray(x_train2_t)
	y_train2_t = np.asarray(y_train2_t)

	np.save('/home/miranda/Documents/code/3DCNN/data_norm/x_train2_arg_merg_b.npy', x_train2_t)
	np.save('/home/miranda/Documents/code/3DCNN/data_norm/y_train2_arg_merg_b.npy', y_train2_t)


def plot_slices(num_rows, num_columns, width, height, data):
	"""Plot a montage of 20 CT slices"""
	data = np.rot90(np.array(data))
	data = np.transpose(data)
	data = np.reshape(data, (num_rows, num_columns, width, height))
	rows_data, columns_data = data.shape[0], data.shape[1]
	heights = [slc[0].shape[0] for slc in data]
	widths = [slc.shape[1] for slc in data[0]]
	fig_width = 12.0
	fig_height = fig_width * sum(heights) / sum(widths)
	f, axarr = plt.subplots(
	    rows_data,
	    columns_data,
	    figsize=(fig_width, fig_height),
	    gridspec_kw={"height_ratios": heights},
	)
	for i in range(rows_data):
	    for j in range(columns_data):
	        axarr[i, j].imshow(data[i][j], cmap="gray")
	        axarr[i, j].axis("off")
	plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
	plt.show()
	# plt.savefig('../pic/INSP_CN020400_vessel.png')

def resample_CT():

	# resample CT volumes to 1mm
	df = pd.read_csv('/home/miranda/Documents/data/INSPIRE/subtype/CT_BL/f_ncct_thr_sub_2.csv')
	paths = []
	for index, row in df.iterrows():
		f = row.tolist()[0]
		scan = nib.load(f)
		affine =  scan.affine
		trg_affine = rescale_affine(affine)
		# Get raw data
		img = resample_img(scan, target_affine=trg_affine, target_shape=[182,218,182])	
		elements = f.split('/')
		name = elements[-1]
		basename = name.replace('.nii.gz', '') 
		folder = '/'.join(elements[:-1])
		paths.append(os.path.join(folder, basename+'_1mm'+'.nii.gz'))
		nib.save(img, os.path.join(folder, basename+'_1mm'+'.nii.gz'))
	df_img = pd.DataFrame(paths)
	df_img.to_csv('/home/miranda/Documents/data/INSPIRE/subtype/CT_BL/f_ncct_thr_sub_2_1mm.csv',index=False)


	# orig = nib.load('/home/miranda/Documents/data/VesselAtlas/vesselProbabilities.nii.gz')
	# affine = orig.affine
	# trg_affine = rescale_affine(affine)

	# img = resample_img(orig, target_affine=trg_affine, target_shape=[182,218,182])
	# nib.save(img, '/home/miranda/Documents/data/VesselAtlas/vesselProbabilities1mm.nii.gz')  


def data_dis():

	df_train = pd.read_csv('/home/miranda/Documents/code/3DCNN/data_norm/data_train.csv')
	df_val = pd.read_csv('/home/miranda/Documents/code/3DCNN/data_norm/data_validate.csv')
	df_test = pd.read_csv('/home/miranda/Documents/code/3DCNN/data_norm/data_test.csv')

	class_0 = []
	class_1 = []
	for index, row in df_train.iterrows():
		path = row['path']
		label = row['label']
		if label != 0:
			label = 1
		volume, affine = read_nifti_file(path) 
		newarr = volume.reshape(-1)
		mean_data = np.mean(newarr)
		# max_data = np.max(newarr)
		if label == 0:
			class_0.append(mean_data)
		elif label == 1:
			class_1.append(mean_data)

	n, bins, patches = plt.hist(x=class_0, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
	plt.grid(axis='y', alpha=0.75)
	plt.xlabel('Value')
	plt.ylabel('Frequency')
	plt.title('Histogram')
	plt.text(23, 45, r'$\mu=15, b=3$')
	maxfreq = n.max()
	# Set a clean upper y-axis limit.
	plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

	n, bins, patches = plt.hist(x=class_1, bins='auto', color='red', alpha=0.7, rwidth=0.85)
	plt.grid(axis='y', alpha=0.75)
	plt.xlabel('Value')
	plt.ylabel('Frequency')
	plt.title('Histogram')
	plt.text(23, 45, r'$\mu=15, b=3$')
	maxfreq = n.max()
	# Set a clean upper y-axis limit.
	plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
	# plt.show()
	plt.savefig('/home/miranda/Documents/code/3DCNN/pic/histogram.png')

def sklearn_data():

	# prepare data for sklearn comparision models
	# X (n_samples, n_features)

	df = pd.read_csv('/home/miranda/Documents/code/3DCNN/data/brain.csv')
	# train = pd.read_csv('/home/miranda/Documents/code/3DCNN/data_norm/data_train.csv')
	# validate = pd.read_csv('/home/miranda/Documents/code/3DCNN/data_norm/data_validate.csv')
	# test = pd.read_csv('/home/miranda/Documents/code/3DCNN/data_norm/data_test.csv')

	# 1 PCA dimension reduction

	X_features = []
	y = []
	for index, row in df.iterrows():
		path = row['path']
		label = row['label']
		# if label != 0:
		# 	label = 1
		if label == 'Cardioembolic':
			l = 0
		elif label == 'Large Artery Atherosclerosis':
			l = 1
		elif label == 'Small vessel (lacunar)':
			l = 2
		volume, affine = read_nifti_file(path)
		volume1 = resize_volume(volume, 128, 128, 128)
		vec = volume1.reshape(-1)
		X_features.append(vec)
		y.append(l)

	X_features = np.asarray(X_features)
	y = np.asarray(y)

	np.save('/home/miranda/Documents/code/3DCNN/data/X_features.npy', X_features)
	np.save('/home/miranda/Documents/code/3DCNN/data/y.npy', y)


def decomp():

	X_features = np.load('/home/miranda/Documents/code/3DCNN/data_norm/X_features_spm.npy')
	y = np.load('/home/miranda/Documents/code/3DCNN/data_norm/y_spm.npy')

	n_samples, n_features = X_features.shape

	n_row, n_col = 3, 3
	# idead number need to be tested
	n_components = n_row * n_col
	image_shape = (128, 128, 128)
	rng = RandomState(0)

	def plot_gallery(title, images, n_col=n_col, n_row=n_row, cmap=plt.cm.gray):
		plt.figure(figsize=(2. * n_col, 2.26 * n_row))
		plt.suptitle(title, size=16)
		for i, comp in enumerate(images):
			plt.subplot(n_row, n_col, i + 1)
			vmax = max(comp.max(), -comp.min())
			plt.imshow(comp.reshape(image_shape)[:,:,50], cmap=cmap,
			           interpolation='nearest',
			           vmin=-vmax, vmax=vmax)
			plt.xticks(())
			plt.yticks(())
		plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)

	# global centering
	X_features_centered = X_features - X_features.mean(axis=0)

	# local centering
	X_features_centered -= X_features_centered.mean(axis=1).reshape(n_samples, -1)

	print("Dataset consists of %d CT" % n_samples)

	# Plot a sample of the input data
	plot_gallery("First centered CT", X_features_centered[:n_components])

	# List of the different estimators, whether to center and transpose the
	# problem, and whether the transformer uses the clustering API.
	estimators = [
		('Eigenvectors - PCA using randomized SVD',
		 decomposition.PCA(n_components=n_components, svd_solver='randomized',
		                   whiten=True),
		 True),

		# ('Non-negative components - NMF',
		#  decomposition.NMF(n_components=n_components, init='nndsvda', tol=5e-3),
		#  False),

		('Independent components - FastICA',
		 decomposition.FastICA(n_components=n_components, whiten=True),
		 True),

		('Factor Analysis components - FA',
		 decomposition.FactorAnalysis(n_components=n_components, max_iter=20),
		 True),
	]

	

	for name, estimator, center in estimators:
		print("Extracting the top %d %s..." % (n_components, name))
		t0 = time()
		data = X_features
		if center:
			data = X_features_centered
		estimator.fit(data)
		train_time = (time() - t0)
		print("done in %0.3fs" % train_time)
		if hasattr(estimator, 'cluster_centers_'):
			components_ = estimator.cluster_centers_
		else:
			components_ = estimator.components_

		# Plot an image representing the pixelwise variance provided by the
		# estimator e.g its noise_variance_ attribute. The Eigenfaces estimator,
		# via the PCA decomposition, also provides a scalar noise_variance_
		# (the mean of pixelwise variance) that cannot be displayed as an image
		# so we skip it.
		if (hasattr(estimator, 'noise_variance_') and
				estimator.noise_variance_.ndim > 0):  # Skip the Eigenfaces case
			plot_gallery("Pixelwise variance",
						estimator.noise_variance_.reshape(1, -1), n_col=1,
						n_row=1)
		plot_gallery('%s - Train time %.1fs' % (name, train_time),
				components_[:n_components])

	plt.show()

	# scaler = StandardScaler()
	# X_features = scaler.fit_transform(X_features)
	# pca = PCA(n_components=17)
	# pca.fit(X_features)
	# x_t =pca.transform(X_features)

def classifiers():

	x_t = np.load('/home/miranda/Documents/code/3DCNN/data/x_t.npy')
	y = np.load('/home/miranda/Documents/code/3DCNN/data/y_t.npy')

	elements_count = collections.Counter(y)
	# printing the element and the frequency
	for key, value in elements_count.items():
		print(f"{key}: {value}")

	# 0: 138
	# 1: 112
	# 2: 36


	n_samples, n_features = x_t.shape
	print('x_t.shape', x_t.shape)
	print(np.unique(y))

	y = label_binarize(y, classes=[0, 1, 2]) # one hot
	n_classes = y.shape[1]
	# print(y)

	# feature decomposition
	# X_features_centered = X_features - X_features.mean(axis=0)

	# # local centering
	# X_features_centered -= X_features_centered.mean(axis=1).reshape(n_samples, -1)

	# n_components = 9

	# # scaler = StandardScaler()
	# # X_features = scaler.fit_transform(X_features)
	# fa = decomposition.FactorAnalysis(n_components=n_components, max_iter=20)
	# fa.fit(X_features_centered)
	# x_t =fa.transform(X_features_centered)

	X_train, X_test, y_train, y_test = train_test_split(x_t, y, test_size=.4, random_state=42)
	print('X_train', X_train.shape)

	# # LR
	# clf_lr = OneVsRestClassifier(LogisticRegression(random_state=1234))
	# model_lr = clf_lr.fit(X_train, y_train)
	# yproba = model_lr.predict_proba(X_test)
	# y_pred = model_lr.predict(X_test)

	# print(multilabel_confusion_matrix(y_test, y_pred))

	# # # Print the precision and recall, among other metrics
	# print(metrics.classification_report(y_test, y_pred, digits=3))

	# fpr = dict()
	# tpr = dict()
	# roc_auc = dict()
	# for i in range(n_classes):
	# 	fpr[i], tpr[i], _ = roc_curve(y_test[:, i], yproba[:, i])
	# 	roc_auc[i] = auc(fpr[i], tpr[i])

	# # Compute micro-average ROC curve and ROC area
	# fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), yproba.ravel())
	# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

	# fig = plt.figure(figsize=(8,6))
	# plt.plot(fpr["micro"], tpr["micro"],
	#          label='micro-average ROC curve (area = {0:0.2f})'
	#                ''.format(roc_auc["micro"]))
	# for i in range(n_classes):
	# 	plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
	#                                    ''.format(i, roc_auc[i]))

	# plt.plot([0, 1], [0, 1], 'k--')
	# plt.xlim([0.0, 1.0])
	# plt.ylim([0.0, 1.05])
	# plt.xlabel('False Positive Rate')
	# plt.ylabel('True Positive Rate')
	# plt.title('Some extension of Receiver operating characteristic to multi-class')
	# plt.legend(loc="lower right")
	# # plt.show()
	# fig.savefig('/home/miranda/Documents/code/3DCNN/pic/LR_roc_curve.png')

	clf_mlp = MLPClassifier(learning_rate='adaptive', max_iter=1000)

	# MLP
	param_mlp = {'hidden_layer_sizes': list(range(20,100,20)),
				'activation': ["logistic", "relu"],
				'alpha': [0.0001, 0.001, 0.01]}

	search_mlp = RandomizedSearchCV(
		    estimator=clf_mlp, param_distributions=param_mlp,
		    n_iter=20
		)
	model_mlp = search_mlp.fit(X_train, y_train)
	
	yproba = model_mlp.predict_proba(X_test)

	y_pred = model_mlp.predict(X_test)

	print("### MLP ###")
	print(multilabel_confusion_matrix(y_test, y_pred))

	# # Print the precision and recall, among other metrics
	print(metrics.classification_report(y_test, y_pred, digits=3))

	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	for i in range(n_classes):
		fpr[i], tpr[i], _ = roc_curve(y_test[:, i], yproba[:, i])
		roc_auc[i] = auc(fpr[i], tpr[i])

	# Compute micro-average ROC curve and ROC area
	fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), yproba.ravel())
	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

	fig = plt.figure(figsize=(8,6))
	plt.plot(fpr["micro"], tpr["micro"],
	         label='micro-average ROC curve (area = {0:0.2f})'
	               ''.format(roc_auc["micro"]))
	for i in range(n_classes):
		plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
	                                   ''.format(i, roc_auc[i]))

	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Some extension of Receiver operating characteristic to multi-class')
	plt.legend(loc="lower right")
	# plt.show()
	fig.savefig('/home/miranda/Documents/code/3DCNN/pic/MLP_roc_curve.png')

	# svm
	clf_svm = OneVsRestClassifier(SVC(kernel="poly",probability = True))
	parameters = {
	    "estimator__C": [1,2,4,8],
	    "estimator__kernel": ["poly","rbf"],
	    "estimator__degree":[1, 2, 3, 4],
	}
	svm_tunning = RandomizedSearchCV(clf_svm, param_distributions=parameters,
	                               n_iter = 20)
	svm_tunning.fit(X_train, y_train)

	yproba = svm_tunning.predict_proba(X_test)

	y_pred = svm_tunning.predict(X_test)
	print("### SVM ###")
	print(multilabel_confusion_matrix(y_test, y_pred))

	# # Print the precision and recall, among other metrics
	print(metrics.classification_report(y_test, y_pred, digits=3))

	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	for i in range(n_classes):
		fpr[i], tpr[i], _ = roc_curve(y_test[:, i], yproba[:, i])
		roc_auc[i] = auc(fpr[i], tpr[i])

	# Compute micro-average ROC curve and ROC area
	fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), yproba.ravel())
	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

	fig = plt.figure(figsize=(8,6))
	plt.plot(fpr["micro"], tpr["micro"],
	         label='micro-average ROC curve (area = {0:0.2f})'
	               ''.format(roc_auc["micro"]))
	for i in range(n_classes):
		plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
	                                   ''.format(i, roc_auc[i]))

	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Some extension of Receiver operating characteristic to multi-class')
	plt.legend(loc="lower right")
	# plt.show()
	fig.savefig('/home/miranda/Documents/code/3DCNN/pic/SVC_roc_curve.png')

	# RF
	# Number of trees in random forest
	n_estimators = [int(x) for x in np.linspace(start = 50, stop = 1000, num = 10)]
	# Number of features to consider at every split
	max_features = ['auto', 'sqrt']
	# Maximum number of levels in tree
	max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
	max_depth.append(None)
	# Minimum number of samples required to split a node
	min_samples_split = [2, 5, 10]
	# Minimum number of samples required at each leaf node
	min_samples_leaf = [1, 2, 4]
	# Method of selecting samples for training each tree
	bootstrap = [True, False]# Create the random grid

	clf_rf = OneVsRestClassifier(RandomForestClassifier())
	clf_rf.fit(X_train, y_train)             

	
	yproba = clf_rf.predict_proba(X_test)

	y_pred = clf_rf.predict(X_test)
	print("### RF ###")
	print(multilabel_confusion_matrix(y_test, y_pred))

	# # Print the precision and recall, among other metrics
	print(metrics.classification_report(y_test, y_pred, digits=3))

	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	for i in range(n_classes):
		fpr[i], tpr[i], _ = roc_curve(y_test[:, i], yproba[:, i])
		roc_auc[i] = auc(fpr[i], tpr[i])

	# Compute micro-average ROC curve and ROC area
	fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), yproba.ravel())
	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

	fig = plt.figure(figsize=(8,6))
	plt.plot(fpr["micro"], tpr["micro"],
	         label='micro-average ROC curve (area = {0:0.2f})'
	               ''.format(roc_auc["micro"]))
	for i in range(n_classes):
		plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
	                                   ''.format(i, roc_auc[i]))

	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Some extension of Receiver operating characteristic to multi-class')
	plt.legend(loc="lower right")
	# plt.show()
	fig.savefig('/home/miranda/Documents/code/3DCNN/pic/RF_roc_curve.png')

	# GradientBoosting
	clf_gb =OneVsRestClassifier(GradientBoostingClassifier())
	clf_gb.fit(X_train, y_train)             

	
	yproba = clf_gb.predict_proba(X_test)

	y_pred = clf_gb.predict(X_test)
	print("### GB ###")
	print(multilabel_confusion_matrix(y_test, y_pred))

	# # Print the precision and recall, among other metrics
	print(metrics.classification_report(y_test, y_pred, digits=3))

	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	for i in range(n_classes):
		fpr[i], tpr[i], _ = roc_curve(y_test[:, i], yproba[:, i])
		roc_auc[i] = auc(fpr[i], tpr[i])

	# Compute micro-average ROC curve and ROC area
	fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), yproba.ravel())
	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

	fig = plt.figure(figsize=(8,6))
	plt.plot(fpr["micro"], tpr["micro"],
	         label='micro-average ROC curve (area = {0:0.2f})'
	               ''.format(roc_auc["micro"]))
	for i in range(n_classes):
		plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
	                                   ''.format(i, roc_auc[i]))

	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Some extension of Receiver operating characteristic to multi-class')
	plt.legend(loc="lower right")
	# plt.show()
	fig.savefig('/home/miranda/Documents/code/3DCNN/pic/GB_roc_curve.png')



def transform_to_hu(medical_image, image):
	intercept = medical_image.RescaleIntercept
	slope = medical_image.RescaleSlope
	hu_image = image * slope + intercept

	return hu_image

def window_image(image, window_center, window_width):
	img_min = window_center - window_width // 2
	img_max = window_center + window_width // 2
	window_image = image.copy()
	window_image[window_image < img_min] = img_min
	window_image[window_image > img_max] = img_max

	return window_image

def test():
	img = nib.load('/home/miranda/Documents/code/3DCNN/data_norm/INSP_AU010062_CT_2_Head_1370308286_Brain_Helical_Trauma_20130604101000_2.ni_NCCT_thr_sub.nii.gz')
	# intercept = img.dataobj.inter
	# slope = img.dataobj.slope
	# image = img.get_fdata()
	# hu_image = image * slope + intercept

	# print(img.dataobj.slope, img.dataobj.inter)
	# data = img.get_fdata()
	# hdr = img.header
	# print(hdr)

	# hu_image = transform_to_hu(medical_image, image)
	# brain_image_v = window_image(hu_image, 40, 80)
	
	brain_image_v = img.get_fdata()
	x,y, z = brain_image_v.shape  #(182, 218, 182)

	new_image = []
	msk_img = []
	for i in range(y):
		brain_image = brain_image_v[:,100,:]

		segmentation = morphology.dilation(brain_image, np.ones((5, 5)))
		labels, label_nb = ndimage.label(segmentation)

		label_count = np.bincount(labels.ravel().astype(np.int))
		# The size of label_count is the number of classes/segmentations found

		# We don't use the first class since it's the background
		label_count[0] = 0

		# We create a mask with the class with more pixels
		# In this case should be the brain
		mask = labels == label_count.argmax()

		# Improve the brain mask
		mask = morphology.dilation(mask, np.ones((5, 5)))
		mask = ndimage.morphology.binary_fill_holes(mask)
		mask = morphology.dilation(mask, np.ones((3, 3)))

		# Since the the pixels in the mask are zero's and one's
		# We can multiple the original image to only keep the brain region
		masked_image = mask * brain_image
		new_image.append(masked_image)
		msk_img.append(mask)
		plt.imshow(mask)
		plt.show()

	new_image = np.asarray(new_image)	
	new_image = nib.Nifti1Image(new_image, affine=img.affine)
	nib.save(new_image, '/home/miranda/Desktop/test.nii.gz')
	# nib.save(new_image, os.path.join(folder,basename+'_'+'norm'+'.nii.gz'))
	new_image2 = np.asarray(msk_img)	
	new_image2 = nib.Nifti1Image(new_image2, affine=img.affine)
	nib.save(new_image2, '/home/miranda/Desktop/mask.nii.gz')

def training_data_b():

	# prepare data to predict functional independence and mortality

	df = pd.read_csv('/home/miranda/Documents/data/INSPIRE/subtype/stats/CT_vol_info_index.csv')

	# add brain image path
	brain_path = []
	for index, row in df.iterrows():
		path = row['Path']
		parts = path.split("/")
		base = "_".join(parts[-3:])
		f = Path(os.path.join(path, 'bn__InverseWarped.nii.gz'))
		if f.is_file():
			brain_path.append(f)
			# # copy file 
			# new_name = os.path.join('/home/miranda/Documents/code/3DCNN/data_raw', base+'.nii.gz')
			# shutil.copy(f, new_name)
		else:
			brain_path.append(np.nan)
	df['brain_path'] = brain_path
	df = df.dropna(axis=0, subset=['brain_path'])



	# train, validate, test = np.split(df.sample(frac=1, random_state=42), 
	# 							[int(.6*len(df)), int(.8*len(df))])

	# train.to_csv('/home/miranda/Documents/code/3DCNN/data/outcome_train.csv', index=False)
	# validate.to_csv('/home/miranda/Documents/code/3DCNN/data/outcome_val.csv', index=False)
	# test.to_csv('/home/miranda/Documents/code/3DCNN/data/outcome_test.csv', index=False)

	# x_train, x_validation, x_test, y_train, y_validation, y_test = [], [], [], [], [], []


	# # x_train 
	# for index, row in train.iterrows():
	# 	path = row['brain_path']
	# 	label = row['Functional_outcome']

	# 	volume, affine = read_nifti_file(path)
	# 	# argument training data
	# 	# resize
	# 	volume1 = resize_volume(volume, 128, 128, 128)
	# 	x_train.append(volume1)
	# 	y_train.append(label)

	# x_train = np.asarray(x_train)
	# y_train = np.asarray(y_train)


	# # x_validation: original volume resize
	# for index, row in validate.iterrows():
	# 	path = row['brain_path']
	# 	label = row['Functional_outcome']
		
	# 	volume, affine = read_nifti_file(path)
	# 	# resize
	# 	volume1 = resize_volume(volume, 128, 128, 128)
	# 	x_validation.append(volume1)
	# 	y_validation.append(label)

	# x_validation = np.asarray(x_validation)
	# y_validation = np.asarray(y_validation)

	# for index, row in test.iterrows():
	# 	path = row['brain_path']
	# 	label = row['Functional_outcome']
		
	# 	volume, affine = read_nifti_file(path)
	# 	# resize
	# 	volume1 = resize_volume(volume, 128, 128, 128)
	# 	x_test.append(volume1)
	# 	y_test.append(label)

	# x_test = np.asarray(x_test)
	# y_test = np.asarray(y_test)


	# np.save('/home/miranda/Documents/code/3DCNN/data/x_train_fo.npy', x_train)
	# np.save('/home/miranda/Documents/code/3DCNN/data/x_val_fo.npy', x_validation)
	# np.save('/home/miranda/Documents/code/3DCNN/data/x_test_fo.npy', x_test)
	# np.save('/home/miranda/Documents/code/3DCNN/data/y_train_fo.npy', y_train)
	# np.save('/home/miranda/Documents/code/3DCNN/data/y_val_fo.npy', y_validation)
	# np.save('/home/miranda/Documents/code/3DCNN/data/y_test_fo.npy', y_test)

# sub_groups()
# demographics()
# normalization('/home/miranda/Documents/data/INSPIRE/subtype/CT_BL/brain_files.csv')
# viz_check()
# attention_map()
# resample_template()
# vessel_data()

# read_xlsx('/home/miranda/Documents/data/INSPIRE/_$INSPIRE Imaging and Clinical data_AB.xlsx')
# remove_badscan()
# prepare_data()
# zero_center()
# training_data()
# data_augmentation()
# MosMedData()
# clinical_info()
# create_balanced_data()
# data_dis()
# sklearn_data()
classifiers()

# Visualize montage of slices.
# 4 rows and 10 columns for 100 slices of the CT scan.
# image, affine = read_nifti_file('../data_norm/vessel_INSP_CN020400_CT_4_1.25_493742059_1.25mm_C-_20190430203320_4.ni_NCCT_thr_sub_vessel.nii.gz')
# data = np.load('/home/miranda/Documents/code/3DCNN/data_norm/x_train_0mean.npy')
# print(data.shape)
# plot_slices(4, 10, 128, 128, data[50, :, :, :40])

# image, affine = read_nifti_file('../data_norm/vessel_INSP_CN020400_CT_4_1.25_493742059_1.25mm_C-_20190430203320_4.ni_NCCT_thr_sub_vessel.nii.gz')
# print(np.count_nonzero(image)) # 563561
# image, affine = read_nifti_file('/home/miranda/Documents/data/VesselAtlas/vesselProbabilities1mm.nii.gz')
# print(np.count_nonzero(image)) # 2438602

# resample_CT()
# image, affine = read_nifti_file('/home/miranda/Documents/data/INSPIRE/subtype/CT_BL/INSP_CN020027_10744237/58740_1333078674/CT_1_10mm315_1333078677.3/CT_1_10mm315_1333078677.3_HS_HEAD_-_Axial_10mm_Head_20120330113754_1.ni_NCCT_thr_sub.nii.gz')
# print(np.min(image))

# data = np.load('/home/miranda/Documents/code/3DCNN/data_norm/x_train_spm.npy')
# print(data.shape)
# plot_slices(4, 10, 128, 128, data[52, :, :, 30:70])
# test()
# decomp()
# training_data_b()