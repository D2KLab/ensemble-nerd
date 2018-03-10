import os
import multiprocessing
import sys
from copy import copy
from os import listdir
from os.path import isfile, join
from langdetect import detect
import re
from itertools import combinations
import pickle
import time
import os.path
from langdetect import detect
from api_pkg import dandelion,dbspotlight,opencalais,babelfy,adel,meaning_cloud,alchemy,textrazor
from api_pkg.utils.representation import *
from time import gmtime, strftime
import numpy as np


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




def getAnnotations(text,lang=None,model_setting='default'):
    extractors_list = [
        alchemy.ALCHEMY(),
        adel.ADEL(),
        dbspotlight.DBSPOTLIGHT(),
        opencalais.OPENCALAIS(),
        meaning_cloud.MEANINGCLOUD(),
        dandelion.DANDELION(),
        babelfy.BABELFY(),
        textrazor.TEXTRAZOR()
    ]
    
    #print(strftime("%H:%M:%S", gmtime()))
    
    for ext in extractors_list:
        try:
            if ext.name == 'adel':
                ext.extract(text,lang=lang,setting=model_setting)
            else:
                ext.extract(text,lang=lang)
        except:
            raise Exception("The extractor",ext.name,"presented an error during the API request phase\n"
                            +str(sys.exc_info()[1]))

            
    #print(strftime("%H:%M:%S", gmtime()))
    extractors_responses = [ext.get_annotations() for ext in extractors_list]
            
    for ext in extractors_list:
        try:
            ext.parse()
        except:
            print(ext.get_annotations())
            raise Exception("The extractor",ext.name,"presented an error during the API response parsing phase\n"+
                            str(sys.exc_info()[1]))
    #print(strftime("%H:%M:%S", gmtime()))
    
    for ext in extractors_list:
        try:
            ext.tokenize()
        except:
            print("The extractor",ext.name,"presented an error during the API response tokenizing phase")
            print(sys.exc_info()[1])
    
    #print(strftime("%H:%M:%S", gmtime()))
    type_list_dict = {}
    entity_list_dict = {}
    score_list_dict = {}
    ontology_type_dict = {}
    ontology_entity_dict = {}
    for ext in extractors_list:
        annotations_ext = ext.get_annotations()
        try:
            ontology_type_dict[ext.name] = ext.ontology
            ontology_entity_dict[ext.name] = ext.ontology
        except:
            ontology_type_dict[ext.name] = ext.ontology_type
            ontology_entity_dict[ext.name] = ext.ontology_uri

        if ext.recognition:
            type_list_dict[ext.name] = [a['type'] for a in annotations_ext]
        if ext.disambiguation:
            entity_list_dict[ext.name] = [a['uri'] for a in annotations_ext]
            
        ext_scores = []
        for i,a in enumerate(annotations_ext):
            scores_list = list()
            try:
                scores_list.append(a['relevance'])
            except:
                pass
            try:
                scores_list.append(a['confidence'])
            except:
                pass
            if i == 0 and not bool(scores_list):
                break
            ext_scores.append(scores_list)
            
        if bool(ext_scores):
            score_list_dict[ext.name] = np.array(ext_scores)
        
    return extractors_responses,type_list_dict,score_list_dict,entity_list_dict,ontology_type_dict,ontology_entity_dict


'''


'''
def getFeatures(text,features_dict_all = {'features':{}},lang=None,model_setting='default'):
    if not bool(lang):
        lang = detect(text)
    
    extractors_responses,type_list_dict,score_list_dict,entity_list_dict,ontology_type_dict,ontology_entity_dict = getAnnotations(text,lang=lang,model_setting=model_setting)
    features_dict_all["entity_list"] = entity_list_dict
    features_dict_all["type_list"] = type_list_dict
    features_dict_all["extractors_responses"] = extractors_responses
    print('Forming features')
    
    if 'fasttext' not in features_dict_all['features']:
        features_dict_all['features']['fasttext']=getFastTextFeatures(text,lang)
    print('Formed Fasttext features')
        
    if 'type' not in features_dict_all['features']:
        features_dict_all['features']['type']=getTypeFeatures(type_list_dict,ontology_type_dict)
    print('Formed type features')
    if 'score' not in features_dict_all['features']:
        features_dict_all['features']['score']=score_list_dict
        
    print('Formed score features')
    if 'entity' not in features_dict_all['features']:
        features_dict_all['features']['entity'],features_dict_all['features']['entity_MATRIX']=getEntityFeatures(entity_list_dict,ontology_entity_dict,lang)
    print('Formed entity features')
    return features_dict_all


def worker_per_file(q,ground_truth):
    flag = True
    while flag:
        text,path = q.get()
        if text == -1:
            flag = False
        else:
            if not os.path.isfile(path):
                features_dict_all = getFeatures(text)
                pickle.dump( features_dict_all, open( path, "wb" ) )



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


n_cpus = multiprocessing.cpu_count()
if n_cpus >= 8:
    n_workers = int(multiprocessing.cpu_count() / 4) + 1
else:
    n_workers = max([min([3,multiprocessing.cpu_count()-1]),1])
n_workers = 1
print("n_workers",n_workers)


queue = multiprocessing.Queue()
jobs = []


for i in range(n_workers):
    p = multiprocessing.Process(target=worker_per_file, args=(queue,ground_truth))
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

