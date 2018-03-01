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
        if os.path.isfile(path):
            features_dict_all = pickle.load( open( path, "rb" ) )
            features_obj = features_dict_all["features"]
            if 'fasttext' in features_obj:
                flag_fasttext = False
            if "type" in features_obj:
                print(path)
                flag_all = False
        else:
            features_dict_all = {}
        if text == -1:
            flag = False
        elif flag_all:
            N = len(splitInTokens(text))
            lang = detect(text)
            print('path',path)
            if flag_fasttext:
                features_obj = {}
                features_obj['fasttext'] = getFastTextFeatures(text,lang)
            else:
                features_obj['fasttext'] = features_dict_all["features"]['fasttext']

            uris_list = {}
            type_list = {}
            features_obj['type'] =  {}
            features_obj['score'] =  {}


            for ext in extractors:
                flag_q = True
                while flag_q:
                    try:
                        ext.extract(text,lang=lang)
                        flag_q = False
                    except:
                        print('Requiried',ext.name,text)
                        time.sleep(10)
                        pass
                try:
                    ext.parse()
                except:
                    print('PARSING ERR',ext,path)
                    raise Exception
                try:
                    ext.tokenize()
                except:
                    print('TOKENIZE ERR')
                    raise Exception
                try:
                    get_type_features = ext.get_type_features()
                    get_score_features = ext.get_score_features()
                except:
                    print('NO PROCESSED TEXT')
                    print(ext)
                    print(text)
                    raise Exception
                if not (len(get_type_features) == len(get_score_features) == N):
                    print('WWWW',ext.name,path,text)
                    print(N,len(get_type_features),len(get_score_features))
                    raise Exception
                features_obj['type'][ext.name] = get_type_features
                features_obj['score'][ext.name] = get_score_features

                uris_l= ext.get_uris_list()
                if bool(uris_l):
                    uris_list[ext.name] = uris_l

                type_l= ext.get_type_list()
                if bool(type_l):
                    type_list[ext.name] = type_l
            print('End types')

            extractors_uris = list(uris_list.keys())
            com_ext = [sorted(item) for item in list(combinations(extractors_uris,2))] + [(ext,ext) for ext in extractors_uris]


            features_obj['uris'] =  {ext1:{ext2:[] for ext2 in extractors_uris} for ext1 in extractors_uris}
            features_obj['uris_MATRIX'] = dict()

            length_list = len(uris_list[extractors_uris[0]])
            for k in range(length_list):
                for comb in com_ext:
                    uri1,uri2 = uris_list[comb[0]][k],uris_list[comb[1]][k]
                    if not bool(uri1):
                        uri1 = np.NAN
                    if not bool(uri2):
                        uri2 = np.NAN

                    key = str(uri1)+'_'+str(uri2)+'_'+lang
                    try:
                        sim = PRECOMPUTED_SIM[key]
                    except:
                        sim = list(getUrisSimilarityVector(uri1,uri2,lang=lang)) + [int(type(uri1)==type(uri2)==float or uri1==uri2)]
                        PRECOMPUTED_SIM[key] = sim
                    if (uri1,uri2) not in features_obj['uris_MATRIX']:
                        features_obj['uris_MATRIX'][(uri1,uri2)] = sim
                    if (uri2,uri1) not in features_obj['uris_MATRIX']:
                        features_obj['uris_MATRIX'][(uri2,uri1)] = sim
                    features_obj['uris'][comb[0]][comb[1]].append(sim)
                    if comb[0] != comb[1]:
                        features_obj['uris'][comb[1]][comb[0]].append(sim)

            for ext in features_obj['uris']:
                for z,key in enumerate(extractors_uris):
                    if z!=0:
                        X_ext = np.append(X_ext,features_obj['uris'][ext][key],axis=1)
                    else:
                        X_ext = np.array(features_obj['uris'][ext][key])
                features_obj['uris'][ext] = X_ext

            print('End Uris')

            features_dict_all["uris_list"] = uris_list
            features_dict_all["type_list"] = type_list
            features_dict_all["features"] = features_obj
            pickle.dump( features_dict_all, open( path, "wb" ) )
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
