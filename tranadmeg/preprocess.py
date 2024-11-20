import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
from src.folderconstants import *
from shutil import copyfile

datasets = ['GUASS', 'SPIKE']

#wadi_drop = ['2_LS_001_AL', '2_LS_002_AL','2_P_001_STATUS','2_P_002_STATUS']

def load_and_save(category, filename, dataset, dataset_folder):
    temp = np.genfromtxt(os.path.join(dataset_folder, category, filename),
                         dtype=np.float64,
                         delimiter=',')
    print(dataset, category, filename, temp.shape)
    np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)
    return temp.shape

def normalize(a):
	a = a / np.maximum(np.absolute(a.max(axis=0)), np.absolute(a.min(axis=0)))
	return (a / 2 + 0.5)

def load_data(dataset):
	folder = os.path.join(output_folder, dataset)
	os.makedirs(folder, exist_ok=True)
	
	if dataset == 'GAUSS':
		dataset_folder = 'data/GAUSS'
		#labels = np.load(dataset_folder+'/labels/1_GAUSS_Anomaly_labels.npy')
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
			labels = np.array([labels])
			labels = labels.T
			np.save(f'{folder}/{f[0:-4]}.npy', labels)

	if dataset == 'SPIKE':
		dataset_folder = 'data/SPIKE'
		#labels = np.load(dataset_folder+'/labels/1_GAUSS_Anomaly_labels.npy')
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
   


if __name__ == '__main__':
	commands = sys.argv[1:]
	load = []
	if len(commands) > 0:
		for d in commands:
			load_data(d)
	else:
		print("Usage: python preprocess.py <datasets>")
		print(f"where <datasets> is space separated list of {datasets}")