import os
import multiprocessing
import sys
from copy import copy
from os import listdir
from os.path import isfile, join
from langdetect import detect
import re
from itertools import combinations
import numpy as np
import pickle
import string
import numpy as np
def isGoodChar(c):
    return c.isalpha() or c.isdigit()

splitting_chars = list()
for ch in string.printable:
    if not isGoodChar(ch):
        splitting_chars.append(ch)

def formatText(string):
    string = string.lower()
    for ch in splitting_chars:
        if ch != ' ':
            string = string.replace(ch,' '+ch+' ')
    return string.replace('  ',' ')

def splitInTokens(string):
    s=formatText(string)
    final_tuples = [(m.group(0), m.start(),m.start()+len(m.group(0))-1) for m in re.finditer(r'\S+', s)]
    final_tuples.sort(key=lambda x:x[1])
    return final_tuples
from pyfasttext import FastText
FASTTEXT_EN = FastText('fasttext_data/wiki.en.bin')
FASTTEXT_FR1= FastText('fasttext_data/wiki.fr.bin')
FASTTEXT_FR2= FastText('fasttext_data/my_corpus_model.bin')

def getFastTextFeatures(text,lang):
    tokens = splitInTokens(text)
    if lang == 'fr':
        for i,token in enumerate(tokens):
            token_text = token[0]
            features_token = np.append(FASTTEXT_FR1[token_text],FASTTEXT_FR2[token_text])
            if i == 0:
                fasttextfeatures = [features_token]
            else:
                fasttextfeatures.append(features_token)
    if lang == 'en':
        for i,token in enumerate(tokens):
            token_text = token[0]
            features_token = FASTTEXT_EN[token_text]
            if i == 0:
                fasttextfeatures = [features_token]
            else:
                fasttextfeatures.append(features_token)
    return np.array(fasttextfeatures)



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





def worker_per_file(q):
    flag = True
    while flag:
        text,path = q.get()
        if text == -1:
            flag = False
        else:
            uris_list = {}
            lang = detect(text)
            try:
                features_dict_all = pickle.load( open( path, "rb" ) )
                features_obj = features_dict_all["features"]
                print(path)
            except:
                features_dict_all = {}
                features_obj = {}
            features_obj['fasttext'] = getFastTextFeatures(text,lang)
            features_dict_all["features"] = features_obj
            pickle.dump( features_dict_all, open( path, "wb" ) )

args = sys.argv


inputfolder,outputfolder=args[1:3]
print('inputfolder',inputfolder)
print('outputfolder',outputfolder)
try:
    os.makedirs(outputfolder)
except:
    pass
text_path_list = readTextsFromFolder(inputfolder,outputfolder)


from langdetect import detect


n_cpus = multiprocessing.cpu_count()
if n_cpus >= 8:
    n_workers = int(multiprocessing.cpu_count() / 4) + 1
else:
    n_workers = max([min([3,multiprocessing.cpu_count()-1]),1])
print("n_workers",n_workers)


queue = multiprocessing.Queue()
jobs = []


for i in range(n_workers):
    p = multiprocessing.Process(target=worker_per_file, args=(queue,))
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
