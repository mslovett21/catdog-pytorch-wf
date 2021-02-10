#!/usr/bin/env python3

import os
import urllib.request
import shutil
import numpy as np
import zipfile
import random
import pickle
from Pegasus.api import *


def create_tar_and_pkl(model):
    model_tar = model + "_model.tar.gz"
    os.system("tar cvf "+ model_tar + " --files-from /dev/null")
    pkl_filename = "hpo_study_checkpoint_" + model + ".pkl"
    file = open(pkl_filename, 'ab')
    pickle.dump("", file, pickle.HIGHEST_PROTOCOL)
    return model_tar, pkl_filename

def create_pkl(model):
    pkl_filename = "hpo_study_checkpoint_" + model + ".pkl"
    file = open(pkl_filename, 'ab')
    pickle.dump("", file, pickle.HIGHEST_PROTOCOL)
    return pkl_filename

# Helper functions
def download_data(dataset_link, zip_data):
    with urllib.request.urlopen(dataset_link) as response,\
    open(zip_data, "wb") as f:
        shutil.copyfileobj(response, f)
        
def unzip_flatten_data(zip_data, directory_to_extract_to):
    with zipfile.ZipFile(zip_data, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)
        
def return_corrupted_files(filename):
    f = open(filename)
    corrupted_files = []
    line = f.readline()
    while line:
        corrupted_files.append(line.strip())
        line = f.readline()
    f.close()
    return corrupted_files

def return_input_files(corrupted_files, DATASET_SIZE, DATA_DIR, LABELS):
    input_file_names = []
    for label in LABELS:
        i = 0
        for f in os.listdir(label):
            if i < DATASET_SIZE/2: # for dev purposes 
                curr_label = label.split('/')[-1]
                src = os.path.join(label, f)
                filename = curr_label + "_" + f
                dst = DATA_DIR + filename
                if dst not in corrupted_files:
                    os.rename(src,dst)
                    input_file_names.append(filename)
                    i = i+1
                else:
                    print("Passing corrupted file {}".format(dst))
    return input_file_names


def add_input_wf_files(input_file_names,DATA_DIR,rc):
    """ Inputs to data_preprocessing1.py
    """
    all_input_files  = []
    for filename in input_file_names:
        dst = DATA_DIR + filename
        all_input_files.append(File(filename))
        rc.add_replica("local", filename,  os.path.join(os.getcwd(), dst)) 
    return all_input_files


#--------------------------------------------------------------------------------------------
def create_file_objects_postfix(input_file_names, postfix):
    """ Given names of fiels creates
        Pegasus File objects, adds postfix to the names
    """
    output_data = []
    for img_name in input_file_names:
        name = img_name.split(".")[0] +  postfix
        output_data.append(File(name))
    return output_data


def create_file_objects(input_file_names):
    """ Given names of fiels creates
        Pegasus File objects
    """
    output_files= []
    for img_name in input_file_names:
        output_files.append(File(img_name))
    return output_files


def create_file_objects_postfix_range(input_file_names,prefix, postfix, range_num):
    """ Given names of fiels creates
        Pegasus File objects, adds postfix to the names and range
    """
    filenames_data = []
    for img_name in input_file_names:
        for i in range(range_num):
            filename = prefix+img_name.split(".")[0] + postfix + str(i)+ ".jpg"
            filenames_data.append(File(filename))
    return filenames_data


#-----------------------------------------------------------------------------------------------

def add_input_tune_model(input_file_names):
    input_tune_model = []
    for filename in input_file_names:
        input_tune_model.append(File(filename))
    return input_tune_model


def split_data_filenames(filenames):

    random.shuffle(filenames)
    train, validate, test = np.split(filenames, [int(len(filenames)*0.7), int(len(filenames)*0.8)])
    files_split_dict = {}
    files_split_dict["train"] = train
    files_split_dict["test"] = test
    files_split_dict["validate"] = validate
    
    train_filenames = [file for file in train] 
    val_filenames = [file for file in validate] 
    test_filenames =  [file for file in test] 
    return train_filenames,val_filenames,test_filenames, files_split_dict