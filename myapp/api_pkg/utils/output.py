import os
from os import listdir
from os.path import isfile, join, isdir
import pickle
import pandas as pd
import numpy as np
from keras.preprocessing import sequence
from keras.models import load_model
from copy import deepcopy
from functools import reduce
import getopt
import sys
from api_pkg.utils.tokenization import *
from api_pkg.utils.request import *
import json


extractors_types=['alchemy', 'adel', 'opencalais', 'meaning_cloud','dandelion','dbspotlight', 'babelfy', 'textrazor']
extractors_disambiguation = ['dandelion','dbspotlight', 'babelfy', 'textrazor']

model_folder_NER = 'data/models/recognition/'
model_folder_NEL = 'data/models/disambiguation/'

model_recognition_names = set([f for f in listdir(model_folder_NER) if isdir(join(model_folder_NER, f))])
model_disambiguation_names = set([f for f in listdir(model_folder_NEL) if isdir(join(model_folder_NEL, f))])


MODELS_DICT = {'recognition':{m:load_model(model_folder_NER+m+'/model.h5') for m in model_recognition_names},
'disambiguation':{m:load_model(model_folder_NEL+m+'/model.h5') for m in model_disambiguation_names}
}




def ConcatenateArray(list_arrays):
    array = list()
    for l in list_arrays:
        array += l
    return array

def getTypesPerFile(f,inv_types_map):
    types = list()
    for k,line in enumerate(f):
        i_max = list(line).index(max(line))
        line_round = line.round()
        if tuple(line_round) in inv_types_map:
            type_ = inv_types_map[tuple(line_round)]
        elif 1 in line_round:
            type_ = inv_types_map[tuple([int(i_max==n) for n in range(len(line_round))])]
        else:
            type_ = '0'
        continue_ = int(line[-1])
        types.append(type_)
    return types

def setURI(t):
    if bool(t) and type(t)==str:
        return t
    else:
        return np.NAN
    
def getURISListPerExtractor(f,extractor_list=['dandelion', 'dbspotlight', 'babelfy', 'textrazor']):
    obj = pickle.load(open(f,'rb'))
    uri_list = obj['entity_list']
    return [{ext:setURI(uri_list[ext][i]) for ext in uri_list} for i in range(len(uri_list['dandelion']))]

def getLISTPREDICTED(P,candidates):
    count = 0
    list_uris = []
    for item in candidates:
        pred_obj = {}
        for uri in item:
            pred_obj[uri] = P[count][0]
            count += 1
        if 1 in [round(v) for v in list(pred_obj.values())]:
            selected = max(pred_obj, key=pred_obj.get)
        else:
            selected = '0'
        if type(selected) != str:
            selected = '0'
        list_uris.append(selected)
    return list_uris

def built_X_sample_disambiguation(f,extractors_disambiguation=extractors_disambiguation,extractors_types=extractors_types):

    if type(f) == dict:
        obj = f
    else:
        obj = pickle.load(open(f,'rb'))

    similarites = obj['features']['entity_MATRIX']
    sim_dict = dict()
    for item in similarites.items():
        sim_dict[(str(item[0][0]),str(item[0][1]))] = item[1]
        sim_dict[(str(item[0][1]),str(item[0][0]))] = item[1]

    uri_list = obj['entity_list']
    features_obj_type = obj['features']['type']
    uris_list_per_extractor_flatten = [{ext:setURI(uri_list[ext][i]) for ext in uri_list} for i in range(len(uri_list['dandelion']))]
    candidates_list = []

    X_dict_test= {ext:[] for ext in extractors_types}
    X_dict_test['entity'] = []

    for i,line in enumerate(uris_list_per_extractor_flatten):
        candidates = set([line[ext] for ext in extractors_disambiguation if type(line[ext])==str])
        if True in [type(line[ext]) == float for ext in extractors_disambiguation]:
            candidates.add(np.NAN)
        candidates = list(candidates)
        candidates.sort(key=lambda x:str(x))
        X_p = [ConcatenateArray([sim_dict[(str(line[ext_1]),str(uri))] for ext_1 in extractors_disambiguation])
               for uri in candidates]
        
        X_dict_test['entity'] += X_p
        candidates_list.append(candidates)
        for ext in extractors_types:
            for c in candidates:
                X_dict_test[ext].append(features_obj_type[ext][i])
    for key in X_dict_test:
        X_dict_test[key] = np.array(X_dict_test[key])
    return X_dict_test,candidates_list



def built_X_partial(features_obj,features,extractors):
    for f in features:
        if type(features_obj[f]) == dict:
            for extractor in features_obj[f]:
                if extractor in extractors:
                    try:
                        X_file = np.append(X_file,features_obj[f][extractor],axis=1)
                    except:
                        X_file = features_obj[f][extractor]
        else:
            try:
                #print(len(X_file))
                X_file = np.append(X_file,features_obj[f],axis=1)
            except:
                X_file = features_obj[f]
    return X_file

def built_X_sample_recognition(features_path,
                     features=['type', 'score', 'entity', 'fasttext'],
                     extractors_types=['alchemy', 'adel', 'opencalais', 'meaning_cloud',
                                       'dandelion', 'dbspotlight', 'babelfy', 'textrazor'],
                     extractors_disambiguation=['dandelion', 'dbspotlight', 'babelfy', 'textrazor'],
                     continue_flag = False,
                     type_flag = True
                    ):

    X_dict = {}
    if type(features_path) == dict:
        obj = features_path
    else:
        obj = pickle.load(open(features_path,'rb'))
    features_obj = obj['features']
    uris_list = obj['entity_list']


    for feat in features:
        if feat in ['type','score']:
            for ext in extractors_types:
                X_dict[feat+ext] = built_X_partial(features_obj,[feat],[ext])
        elif feat == 'entity':
            for ext in extractors_disambiguation:
                X_dict[feat+ext] = built_X_partial(features_obj,[feat],[ext])
        else:
            X_dict[feat] = built_X_partial(features_obj,[feat],[])         
    
    #print(len(uris_list['babelfy']),len(features_obj['type']['babelfy']),len(ground_truth_pd))
    #print(Y_file)
    #print(list(X_file[0]))
    return X_dict

def estimateContinueInfo(type_list_predicted,entity_list_predicted):
    offsets = list()
    offsets_type = list()
    offsets_entity = list()

    for j,type_ in enumerate(type_list_predicted):
        entity_ = entity_list_predicted[j]
        continue_type = type(type_)==str and type_ != '0' and j != len(type_list_predicted) - 1 and type_list_predicted[j+1] == type_
        continue_entity = type(entity_)==str and entity_ != '0' and j != len(entity_list_predicted) - 1 and entity_list_predicted[j+1] == entity_
        if continue_type or continue_entity:
            offsets.append(1)
            if continue_type:
                offsets_type.append(1)
            else:
                offsets_type.append(0)
            if continue_entity:
                offsets_entity.append(1)
            else:
                offsets_entity.append(0)
        else:
            offsets.append(0)
            offsets_type.append(0)
            offsets_entity.append(0)
    return offsets,offsets_type,offsets_entity

def fillMissing(list_values,offsets,other_list):
    continue_index = []
    for i,value in enumerate(list_values):
        offset = offsets[i]
        other_list_value = other_list[i]

        if offset == 0 and i!=0 and offsets[i-1] == 1:
            continue_index.append(i)
            possible_values = [list_values[j] for j in continue_index]
            possible_values_set = set(possible_values)
            fill_val = sorted([(v,possible_values.count(v)) for v in possible_values_set], key=lambda tup: -tup[1])[0][0]
            if fill_val == '0':
                fill_val = '1'
            for j in continue_index:
                list_values[j] = fill_val
            continue_index = []
        elif offset == 0 and value=='0' and other_list_value != '0':
            list_values[i] = '1'
        elif offset == 1:
            continue_index.append(i)
    return list_values
def getNamedEntities(records,text):
    entities = list()
    type_entities = list()
    uri_entitities = list()
    start = None
    for rec in records:
        cond_type = rec['type'] not in ['1','0']
        cond_uri = rec['uri'] not in ['1','0']
        #print(rec['surface'],rec['type'],rec['uri'],cond_type,cond_uri,rec['offset']) 
        if cond_uri or cond_type:
            if rec['offset'] == 1:
                if not bool(start):
                    start = rec['start']
            else:
                if not bool(start):
                    start = rec['start']
                e = {
                    'surface':text[start:rec['end']+1],
                    'start':start,
                    'end':rec['end']
                }
                if cond_type:
                    e['type'] = rec['type']
                    e_t = deepcopy(e)
                    if 'uri' in e_t:
                        del e_t['uri']
                    type_entities.append(e_t)                        
                if cond_uri:
                    e['uri'] = rec['uri']
                    e_u = deepcopy(e)
                    if 'type' in e_u:
                        del e_u['type']
                    uri_entitities.append(e_u)
                entities.append(e)
                start = None
    return entities,type_entities,uri_entitities

def getTypeNamedEntities(records,text):
    type_entities = list()
    start = None
    for rec in records:
        cond_type = rec['type'] not in ['1','0']
        if cond_type:
            if rec['offset_type'] == 1:
                start = rec['start']
            else:
                if not bool(start):
                    start = rec['start']
                e = {
                    'surface':text[start:rec['end']+1],
                    'start':start,
                    'end':rec['end'],
                    'type':rec['type_raw']
                }
                type_entities.append(e)
                start = None

    return type_entities

def getUriNamedEntities(records,text):
    uri_entities = list()
    start = None
    for rec in records:
        cond_uri = rec['uri'] not in ['1','0']
        if cond_uri:
            if rec['offset_uri'] == 1:
                start = rec['start']
            else:
                if not bool(start):
                    start = rec['start']
                e = {
                    'surface':text[start:rec['end']+1],
                    'start':start,
                    'end':rec['end'],
                    'uri':rec['uri_raw']
                }
                uri_entities.append(e)
                start = None

    return uri_entities

def formBRATandSave(entities,output_path):
    BASIC_BRAT_ANN = 'TINDEX\tTYPE START END\tSURFACE'
    brat_lines = list()
    for i,e in enumerate(entities):
        if 'type' in e:
            index_ = str(i+1)
            type_ = e['type']
            start_ = str(e['start'])
            end_ = str(e['end']+1)
            text_ = e['surface']
            b_l = BASIC_BRAT_ANN.replace('INDEX',index_).replace('TYPE',type_).replace('START',start_).replace('END',end_).replace('SURFACE',text_)
            brat_lines.append(b_l)
    with open(output_path,'w+') as f_out:
        f_out.write('\n'.join(brat_lines))

def getAnnotationsOutput(features_file,text_file,model_name_recognition='oke2016',model_name_disambiguation='oke2016',outputfilebase=None,return_flag=False,eval_flag=False,normalize_flag=True):



    ground_truth_folder_train = 'data/training_data/' + model_name_recognition+'/train/csv_ground_truth/'
    ground_truth_folder_test = 'data/training_data/'+ model_name_recognition +'/test/csv_ground_truth/'

    ground_truth_paths = (
        [ground_truth_folder_train+f for f in listdir(ground_truth_folder_train) if isfile(join(ground_truth_folder_train, f)) and '.csv' in f]
        +
        [ground_truth_folder_test+f for f in listdir(ground_truth_folder_test) if isfile(join(ground_truth_folder_test, f)) and '.csv' in f]
    )

    types = set()
    for g in ground_truth_paths:
        truth_pd = pd.read_csv(g)
        types = types | set(truth_pd['type'])
        
    types.remove(np.NAN)
    types = list(types)
    types.sort()

    try:
        text = open(text_file,'r').read()
    except:
        text = text_file
    tokens = splitInTokens(text)

    types_dict = {t:[int(i==j) for j in range(len(types))] for i,t in enumerate(types)}
    inv_types_map = {tuple(v): k for k, v in types_dict.items()}
    model_recognition_path = model_folder_NER+model_name_recognition+'/model.h5'
    model_recognition_path = model_folder_NEL+model_name_disambiguation+'/model.h5'


    if model_name_recognition in MODELS_DICT['recognition']:
        model_recognition= MODELS_DICT['recognition'][model_name_recognition]
        X_dict_recogniton = built_X_sample_recognition(features_file)
        predicted= model_recognition.predict(X_dict_recogniton,verbose=0)
        type_list_predicted = getTypesPerFile(predicted,inv_types_map)
        #print('ok recognition')
    else:
        #print('NO recognition')
        type_list_predicted = ['0' for i in range(len(tokens))]

    if model_name_disambiguation in MODELS_DICT['disambiguation']:
        model_disambiguation = MODELS_DICT['disambiguation'][model_name_disambiguation]
        X_dict_disambiguation,candidates_list =built_X_sample_disambiguation(features_file)
        predicted= model_disambiguation.predict(X_dict_disambiguation,verbose=0)
        entity_list_predicted = getLISTPREDICTED(predicted,candidates_list)
        #print('ok disambiguation')
    else:
        entity_list_predicted = ['0' for i in range(len(tokens))]
        #print('NO disambiguation')


    offsets,offsets_type,offsets_entity = estimateContinueInfo(type_list_predicted,entity_list_predicted)


    type_list_predicted_filled = fillMissing(deepcopy(type_list_predicted),offsets,entity_list_predicted)
    entity_list_predicted_filled = fillMissing(deepcopy(entity_list_predicted),offsets,type_list_predicted)

    #print(len(tokens),len(type_list_predicted_filled),len(entity_list_predicted_filled))



    records = []
    for i,t in enumerate(tokens):
        records.append({
            'surface':t[0],
            'start':t[1],
            'end':t[2],
            'offset':offsets[i],
            'offset_type':offsets_type[i],
            'offset_uri':offsets_entity[i],
            'type':type_list_predicted_filled[i],
            'uri':entity_list_predicted_filled[i],
            'type_raw':type_list_predicted[i],
            'uri_raw':entity_list_predicted[i]
        })

    if eval_flag:
        pd.DataFrame(records).to_csv(outputfilebase+'.csv',index=False)


    wd_uris = ['http://www.wikidata.org/entity/'+uri for uri in set(pd.DataFrame(records)['uri'])]






    entities,type_entities,uri_entitities = getNamedEntities(records,text)

    #print('len entitities',len(entities))

    if normalize_flag:
        uri_dict = fromWikidataToDbpediaUri(wd_uris).set_index('wd_uri')['db_uri'].to_dict()
        for i,e in enumerate(entities):
            if 'uri' not in e:
                entities[i]['Wikidata identifier'] = ''
                entities[i]['Dbpedia link'] = ''
            else:
                wd_id = e['uri']
                wd_uri ='http://www.wikidata.org/entity/'+wd_id
                entities[i]['Wikidata identifier'] = wd_id
                if wd_uri in uri_dict:
                    entities[i]['Dbpedia link'] = uri_dict[wd_uri]
                else:
                    entities[i]['Dbpedia link'] = ''
                del entities[i]['uri']
            if 'type' not in e:
                entities[i]['type'] = ''



    json_obj = {'text':text,'entities':entities}
    if return_flag:
        return json_obj


    with open(outputfilebase+'.json',"w") as f:
      json.dump(json_obj, f)


    if eval_flag:

        uri_entities_raw = getUriNamedEntities(records,text)
        type_entities_raw = getTypeNamedEntities(records,text)

        json_obj = {'text':text,'entities':type_entities}
        with open(outputfilebase+'_type.json',"w") as f:
          json.dump(json_obj, f)
        json_obj = {'text':text,'entities':uri_entitities}
        with open(outputfilebase+'_uri.json',"w") as f:
          json.dump(json_obj, f)
        json_obj = {'text':text,'entities':type_entities_raw}
        with open(outputfilebase+'_type_raw.json',"w") as f:
          json.dump(json_obj, f)
        json_obj = {'text':text,'entities':uri_entities_raw}
        with open(outputfilebase+'_uri_raw.json',"w") as f:
          json.dump(json_obj, f)


        formBRATandSave(entities,outputfilebase+'.ann')
        formBRATandSave(type_entities,outputfilebase+'_type.ann')
        formBRATandSave(type_entities_raw,outputfilebase+'_type_raw.ann')


