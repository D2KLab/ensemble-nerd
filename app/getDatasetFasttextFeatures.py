import os
import multiprocessing
import sys
from copy import copy
from os import listdir
from os.path import isfile, join
from langdetect import detect
from api_pkg.utils.tokenization import *
import re
from itertools import combinations
import pickle
import time
import os.path
from langdetect import detect
from time import gmtime, strftime
import numpy as np
import getopt

from pyfasttext import FastText

FASTTEXT_EN = FastText('data/fasttext_data/wiki.en.bin')
FASTTEXT_FR1= FastText('data/fasttext_data/wiki.fr.bin')
FASTTEXT_FR2= FastText('data/fasttext_data/my_corpus_model.bin')



def getFastTextFeatures(text,lang):
    tokens = splitInTokens(text)
    if lang == 'fr':
        fasttextfeatures = np.array([
            np.append(FASTTEXT_FR1[token[0]],FASTTEXT_FR2[token[0]])
            for i,token in enumerate(tokens)])
    if lang == 'en':
        fasttextfeatures = np.array([FASTTEXT_EN[token[0]] for i,token in enumerate(tokens)])
    return fasttextfeatures



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

ground_truths = sys.argv[1].split(',')
optlist, args = getopt.getopt(sys.argv[2:], '',['workers=','lang='
                                                ])
lang = None

for opt, arg in optlist:
        if opt == '--workers':
            n_workers = int(arg)
        if opt == '--lang':
            lang = arg




'''


'''
def getFeaturesFasttext_(text,lang):
    if not bool(lang):
        lang = detect(text)
    return getFastTextFeatures(text,lang)


def worker_per_file(q,ground_truth,lang):
        flag = True
        while flag:
            text,path = q.get()
            if text == -1:
                flag = False
            else:
                if os.path.isfile(path):
                    print(path)
                    features_dict_all = pickle.load(open(path,'rb'))
                    features_dict_all['features']['fasttext'] = getFeaturesFasttext_(text,lang=lang)
                    pickle.dump( features_dict_all, open( path, "wb" ) )


for ground_truth in ground_truths:
    inputfolder_train = 'data/training_data/'+ground_truth+'/train/txt_files/'
    outputfolder_train = 'data/training_data/'+ground_truth+'/train/features_files/'
    inputfolder_test = 'data/training_data/'+ground_truth+'/test/txt_files/'
    outputfolder_test = 'data/training_data/'+ground_truth+'/test/features_files/'



    text_path_list = readTextsFromFolder(inputfolder_train,outputfolder_train) + readTextsFromFolder(inputfolder_test,outputfolder_test)

    try:
        n_workers = n_workers
    except:
        n_workers = 1


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

