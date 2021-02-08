#!/usr/bin/env python3

"""
MACHINE LEARNING WORKFLOWS - DATA SPLIT

70 % of data for training
10 % for validation
20 % for testing

"""
import cv2
import os
import pickle

DATASET_DIR = ""
DATA_SPLIT_FILE = "data_split_id_list.pickle"




def rename_files(file_list, prefix_file):
    for file in file_list:
        img = cv2.imread(DATASET_DIR+file)
        cv2.imwrite(prefix_file + file,img)


def main():
	data_split = pickle.load( open( DATA_SPLIT_FILE, "rb" ) )

	train, validate, test = data_split["train"], data_split["validate"], data_split["test"]
	all_files = [train, validate, test]
	prefix_list = ["train_" , "val_", "test_"]

	for files, prefix in zip(all_files, prefix_list):
		rename_files(files, prefix)

if __name__ == "__main__":
    main()
