import os
import multiprocessing
import sys
from copy import copy
from os import listdir
from os.path import isfile, join
import re
from itertools import combinations
import pickle
import time
import os.path
from time import gmtime, strftime
import numpy as np
import getopt
from api_pkg.utils.representation import *


def getPathName(folder,original_filepath,ext='p'):
    base = os.path.splitext(os.path.basename(original_filepath))[0]
    if folder[-1]!='/':
        folder += '/'
    new_path = '.'.join([folder+base,ext])
    return new_path

def readTextsFromFolder(folder,outputfolder):
    inputfilepaths = [folder+f for f in listdir(folder) if isfile(join(folder, f)) and '.txt' in f]
    text_path_list = list()
    for path in inputfilepaths:
        outpath = getPathName(outputfolder,path)
        with open(path) as f:
            text = f.read()
        text_path_list.append((text,outpath))
    return text_path_list

args = sys.argv

ground_truth = sys.argv[1]

optlist, args = getopt.getopt(sys.argv[2:], '',['workers=','lang='
                                                ])
lang = None

for opt, arg in optlist:
        if opt == '--workers':
            n_workers = int(arg)
        if opt == '--lang':
            lang = arg



def worker_per_file(q,ground_truth,lang):
    flag = True
    while flag:
        text,path = q.get()
        if text == -1:
            flag = False
        else:
            if not os.path.isfile(path):
                print(path)
                features_dict= getFeatures(text,lang=lang,model_setting=ground_truth)
                pickle.dump( features_dict, open( path, "wb" ) )



inputfolder_train = 'data/training_data/'+ground_truth+'/train/txt_files/'
outputfolder_train = 'data/training_data/'+ground_truth+'/train/features_files/'
inputfolder_test = 'data/training_data/'+ground_truth+'/test/txt_files/'
outputfolder_test = 'data/training_data/'+ground_truth+'/test/features_files/'

try:
    os.makedirs(outputfolder_train)
except:
    pass

try:
    os.makedirs(outputfolder_test)
except:
    pass


text_path_list = readTextsFromFolder(inputfolder_train,outputfolder_train) + readTextsFromFolder(inputfolder_test,outputfolder_test)

try:
    n_workers = n_workers
except:
    n_cpus = multiprocessing.cpu_count()

    if n_cpus >= 8:
        n_workers = int(multiprocessing.cpu_count() / 4) + 1
    else:
        n_workers = max([min([3,multiprocessing.cpu_count()-1]),1])

print("n_workers",n_workers)


queue = multiprocessing.Queue()
jobs = []


for i in range(n_workers):
    p = multiprocessing.Process(target=worker_per_file, args=(queue,ground_truth,lang))
    jobs.append(p)

for job in jobs:
    job.start()



counter = 0
for item in text_path_list:
    queue.put(item)
            

for i in range(n_workers):
    queue.put((-1,-1))


for job in jobs:
    job.join()

