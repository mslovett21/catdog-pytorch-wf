#!/usr/bin/env python3

"""
MACHINE LEARNING WORKFLOWS - DATA SPLIT

70 % of data for training
10 % for validation
20 % for testing

"""

import os
import pickle

DATASET_DIR = ""
DATA_SPLIT_FILE = "data_split_id_list.pickle"


def main():
    data_split = pickle.load( open( DATA_SPLIT_FILE, "rb" ) )
    print(data_split)

    train = data_split["train"]
    validate = data_split["validate"]
    test = data_split["test"]

    # rename files os.rename(src, dst)
    [os.rename(DATASET_DIR + file, "train_" + file) for file in train] 
    [os.rename(DATASET_DIR + file, "val_" + file) for file in validate] 
    [os.rename(DATASET_DIR + file, "test_" + file) for file in test] 

if __name__ == "__main__":
    main()
