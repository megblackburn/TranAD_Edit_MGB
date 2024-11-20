import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
from src.folderconstants import *
from shutil import copyfile

datasets = ['synthetic', 'SMD', 'SWaT', 'SMAP', 'MSL', 'WADI', 'MSDS', 'UCR','UCR_old' 'MEG_TRIAL','MEG_SIG', 'MBA', 'NAB']

wadi_drop = ['2_LS_001_AL', '2_LS_002_AL','2_P_001_STATUS','2_P_002_STATUS']

def load_and_save(category, filename, dataset, dataset_folder):
    temp = np.genfromtxt(os.path.join(dataset_folder, category, filename),
                         dtype=np.float64,
                         delimiter=',')
    print(dataset, category, filename, temp.shape)
    np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)
    return temp.shape

def load_and_save2(category, filename, dataset, dataset_folder, shape):
	temp = np.zeros(shape)
	with open(os.path.join(dataset_folder, 'interpretation_label', filename), "r") as f:
		ls = f.readlines()
	for line in ls:
		pos, values = line.split(':')[0], line.split(':')[1].split(',')
		start, end, indx = int(pos.split('-')[0]), int(pos.split('-')[1]), [int(i)-1 for i in values]
		temp[start-1:end-1, indx] = 1
	print(dataset, category, filename, temp.shape)
	np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)

def normalize(a):
	a = a / np.maximum(np.absolute(a.max(axis=0)), np.absolute(a.min(axis=0)))
	return (a / 2 + 0.5)

def normalize2(a, min_a = None, max_a = None):
	if min_a is None: min_a, max_a = min(a), max(a)
	return (a - min_a) / (max_a - min_a), min_a, max_a

def normalize3(a, min_a = None, max_a = None):
	if min_a is None: min_a, max_a = np.min(a, axis = 0), np.max(a, axis = 0)
	return (a - min_a) / (max_a - min_a + 0.0001), min_a, max_a

def convertNumpy(df):
	x = df[df.columns[3:]].values[::10, :]
	return (x - x.min(0)) / (x.ptp(0) + 1e-4)

def load_data(dataset):
	folder = os.path.join(output_folder, dataset)
	os.makedirs(folder, exist_ok=True)
	if dataset == 'GAUSS':
		dataset_folder = 'data/GAUSS'
		#labels = np.load(dataset_folder+'/labels/1_GAUSS_Anomaly_Gaussnoise_labels.npy')
		#print(np.shape(labels))
		filelist3 = os.listdir(dataset_folder+'/labels/')
		filelist1 = os.listdir(dataset_folder+'/train/')
		filelist2 = os.listdir(dataset_folder+'/test/')
		#np.save(f'{folder}/1_GAUSS_Anomaly_Gaussnoise_labels.npy', labels)
		for f in filelist1:
			train = np.loadtxt(f'{dataset_folder}/train/{f}')
			train = np.array([train])
			train = train.T
			#print(np.shape(train))
			train= normalize(train)
			np.save(f'{folder}/{f[0:-4]}.npy', train)
		for f in filelist2:
			test = np.loadtxt(f'{dataset_folder}/test/{f}')
			test = np.array([test])
			test = test.T
			print(np.shape(test))
			test = normalize(test)
			np.save(f'{folder}/{f[0:-4]}.npy', test)
		for f in filelist3:
			labels = np.load(f'{dataset_folder}/labels/{f}')
			#labels = np.array([labels])
			#labels = labels.T
			np.save(f'{folder}/{f[0:-4]}.npy', labels)
		#for file in ['train', 'test', 'labels']:
		#	np.save(os.path.join(folder, f'{file}.npy'), eval(file).astype('float64'))

	else:
		raise Exception(f'Not Implemented. Check one of {datasets}')

if __name__ == '__main__':
	commands = sys.argv[1:]
	load = []
	if len(commands) > 0:
		for d in commands:
			load_data(d)
	else:
		print("Usage: python preprocess.py <datasets>")
		print(f"where <datasets> is space separated list of {datasets}")