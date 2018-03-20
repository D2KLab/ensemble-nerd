import os
from os import listdir
from os.path import isfile, join
import pickle
import pandas as pd
import numpy as np
from keras.preprocessing import sequence
from keras.models import load_model
from copy import deepcopy
from functools import reduce
import getopt
import sys
from api_pkg.utils.tokenization import splitInTokens
from api_pkg.utils.output import *
import json

try:
    base = sys.argv[1]
except:
    print('You have to specify the input base')
    exit()

try:
    model_name = sys.argv[2]
except:
    print('You have to specify the model to use')
    exit()

extractors_types=['alchemy', 'adel', 'opencalais', 'meaning_cloud','dandelion','dbspotlight', 'babelfy', 'textrazor']
extractors_disambiguation = ['dandelion','dbspotlight', 'babelfy', 'textrazor']






training_folder = 'data/training_data/'+base+'/'
features_folder_train = training_folder + 'train/features_files/'
features_folder_test = training_folder + 'test/features_files/'
output_folder_train = training_folder + 'train/output/'
output_folder_test = training_folder + 'test/output/'
text_folder_train = training_folder + 'train/txt_files/'
text_folder_test = training_folder + 'test/txt_files/'
try:
    os.makedirs(output_folder_train)
except:
    pass
try:
    os.makedirs(output_folder_test)
except:
    pass        
features_paths_train = [features_folder_train+f for f in listdir(features_folder_train) if isfile(join(features_folder_train, f)) and '.p' in f]
features_paths_train.sort()
features_paths_test = [features_folder_test+f for f in listdir(features_folder_test) if isfile(join(features_folder_test, f)) and '.p' in f]
features_paths_test.sort()
for f in features_paths_train:
    filenamebase = f.split('/')[-1].split('.')[0]
    outputfilebase = output_folder_train + filenamebase
    getAnnotationsOutput(f,text_folder_train+filenamebase+'.txt',model_name_recognition=model_name,model_name_disambiguation=model_name,outputfilebase=outputfilebase,eval_flag=True)

for f in features_paths_test:
    filenamebase = f.split('/')[-1].split('.')[0]
    outputfilebase = output_folder_test + filenamebase
    getAnnotationsOutput(f,text_folder_test+filenamebase+'.txt',model_name_recognition=model_name,model_name_disambiguation=model_name,outputfilebase=outputfilebase,eval_flag=True)







