
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







def getPathName(folder,original_filepath,ext='p'):
    base = os.path.splitext(os.path.basename(original_filepath))[0]
    if folder[-1]!='/':
        folder += '/'
    new_path = '.'.join([folder+base,ext])
    return new_path

def readTextsFromFolder(folder,outputfolder):
    inputfilepaths = [folder+f for f in listdir(folder) if isfile(join(inputfolder, f)) and '.txt' in f]
    text_path_list = list()
    for path in inputfilepaths:
        outpath = getPathName(outputfolder,path)
        with open(path) as f:
            text = f.read()
        text_path_list.append((text,outpath))
    return text_path_list





def worker_per_file(q,objects_extractors):
    PRECOMPUTED_SIM = {}
    flag = True
    while flag:
        extractors = copy(objects_extractors)
        text,path = q.get()
        flag_fasttext = True
        flag_all = True
        features_dict_all = pickle.load( open( path, "rb" ) )

        N = len(splitInTokens(text))
        lang = detect(text)
        print('path',path)


        for ext in extractors:
        	if ext.name == 'adel':
        		ext.extract(text,lang=lang,setting='oke2016')
        		ext.parse()
        		ext.tokenize()
        		get_type_features = ext.get_type_features()
        		get_score_features = ext.get_score_features()
        		features_dict_all["features"]['type'][ext.name] = get_type_features
        		features_dict_all["features"]['score'][ext.name] = get_score_features
        		type_l= ext.get_type_list()
        		features_dict_all["features"]["type_list"]['adel'] = type_l
            	pickle.dump( features_dict_all, open( path, "wb" ) )
            	break
        del extractors

args = sys.argv


inputfolder,outputfolder=args[1:3]
print('inputfolder',inputfolder)
print('outputfolder',outputfolder)
try:
    os.makedirs(outputfolder)
except:
    pass
text_path_list = readTextsFromFolder(inputfolder,outputfolder)



from api_pkg import dandelion,dbspotlight,opencalais,babelfy,adel,meaning_cloud,alchemy,textrazor
from api_pkg.utils import *
from langdetect import detect

objects_extractors = [
    alchemy.ALCHEMY(),
    adel.ADEL(),
    opencalais.OPENCALAIS(),
    meaning_cloud.MEANINGCLOUD(),
    dandelion.DANDELION(),
    dbspotlight.DBSPOTLIGHT(),
    babelfy.BABELFY(),
    textrazor.TEXTRAZOR()
]

n_cpus = multiprocessing.cpu_count()
if n_cpus >= 8:
    n_workers = int(multiprocessing.cpu_count() / 4) + 1
else:
    n_workers = max([min([3,multiprocessing.cpu_count()-1]),1])
print("n_workers",n_workers)


queue = multiprocessing.Queue()
jobs = []


for i in range(n_workers):
    p = multiprocessing.Process(target=worker_per_file, args=(queue,objects_extractors))
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
