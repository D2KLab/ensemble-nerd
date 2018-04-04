from os import listdir
from os.path import isfile, join,isdir
import os
import pickle
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from sklearn.metrics import f1_score,recall_score,precision_score
from api_pkg.utils.tokenization import splitInTokens
from functools import reduce
from IPython.display import display, HTML
sns.set_style("whitegrid", {"axes.grid": False})

try:
    base = sys.argv[1]
except:
    print('You have to specify the ground truth')
    exit()

scores = {
    'f1':f1_score,
    'recall':recall_score,
    'precision':precision_score
}

def createFolder(folder):
    try:
        os.makedirs(folder)
    except:
        pass
    
def getScoresToken(gt_list,p_list,score):
    types = [g for g in set(gt_list) if type(g) == str and g not in ['0','1']]
    score_obj= dict()
    scores_res = score(gt_list, p_list, labels=types,average=None)
    for i,t in enumerate(types):
        score_obj[t]=scores_res[i]
    for avg in ['micro', 'macro']:
        sc = score(gt_list, p_list, labels=types,average=avg)
        score_obj[avg]=sc
    return score_obj

def getTypesListGT(groundtruth_paths):
    type_df_per_file = list()
    for g in groundtruth_paths:
        recs_gt = pd.read_csv(g).to_dict(orient='records')
        types = list()
        for r in recs_gt:
            if type(r['type']) == str:
                type_ = r['type']
            else:
                type_ = '0'
            types.append(type_)
        type_df_per_file.append(types)
    return type_df_per_file

def getOutputFile(features_file,extension,to_append=''):
    path_file = features_file.split('features_files/')[0] + 'output/'+features_file.split('/')[-1].split('.')[0]
    if extension == 'csv':
        path_file += '.csv'
        df = pd.read_csv(path_file)
        return df
    elif extension == 'json':
        path_file += to_append + '.json'
        data = json.load(open(path_file))
        return data
    elif extension == 'brat':
        path_file += to_append + '.ann'
        return path_file
        
    
def getOutputTypeList(features_paths,raw=False):
    if raw:
        col = 'type_raw'
    else:
        col = 'type'
    for i,f in enumerate(features_paths):
        types_list_f = list(getOutputFile(f,'csv')[col])
        if i == 0:
            types_list = types_list_f
        else:
            types_list += types_list_f
    
    for i,t in enumerate(types_list):
        if type(t)!=str or t == '1':
            types_list[i] = '0'
    return types_list

def getOutputEntityList(features_paths,raw=False):
    if raw:
        col = 'uri_raw'
    else:
        col = 'uri'
    for i,f in enumerate(features_paths):
        types_list_f = list(getOutputFile(f,'csv')[col])
        if i == 0:
            types_list = types_list_f
        else:
            types_list += types_list_f
    
    for i,t in enumerate(types_list):
        if type(t)!=str or t == '1':
            types_list[i] = '0'
    return types_list


def getExtractorsTypesLists(features_paths,extractors_type,base):
    extractors_type_lists = dict()
    for ext in extractors_type:
        extractors_type_lists[ext] = []
        if ext == 'adel' and base in ['oke2016','aida','oke2015']:
            col = 'type_list'
        else:
            col = 'type_list_normalized'
        for path in features_paths:
            features_dict= pickle.load(open(path,'rb'))
            list_ext = features_dict[col][ext]
            for i,l in enumerate(list_ext):
                if type(l) != str or l == '1':
                    list_ext[i] = '0'
            extractors_type_lists[ext].append(list_ext)
    return extractors_type_lists
            
def generateHistogram(x,path,ext_names):
    N = len(x)
    x = np.transpose(x)
    f1_score = tuple(x[0])
    precision = tuple(x[1])
    recall = tuple(x[2])


    ind = np.arange(N)  # the x locations for the groups
    width = 0.3      # the width of the bars

    fig, ax = plt.subplots(figsize=(20,9))

    rects1 = ax.bar(ind, f1_score, width, color='r')

    rects2 = ax.bar(ind + width, precision, width, color='y')

    rects3 = ax.bar(ind + width*2, recall, width, color='g')

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Scores')
    ax.set_title('Scores by extractor')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(tuple(ext_names))

    ax.legend((rects1[0], rects2[0],rects3[0]), ('F1_score', 'Precision','Recall'))


    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    '',
                    ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    plt.savefig(path, dpi=600)
    plt.show()

training_folder = 'data/training_data/'+base+'/'
evaluation_folder = training_folder + 'evaluation/'
evaluation_folder_NER = evaluation_folder + 'recognition/'
evaluation_folder_NEL = evaluation_folder + 'disambiguation/'
evaluation_folder_NER_bin = evaluation_folder_NER + 'binary/'
evaluation_folder_NEL_bin = evaluation_folder_NEL + 'binary/'
createFolder(evaluation_folder_NER_bin)
createFolder(evaluation_folder_NEL_bin)
ground_truth_folder = training_folder + 'test/csv_ground_truth/'
features_folder_test = training_folder + 'test/features_files/'
features_paths= [features_folder_test+f for f in listdir(features_folder_test) if isfile(join(features_folder_test, f)) and '.p' in f]
features_paths.sort()
groundtruth_paths = [path.replace(features_folder_test,ground_truth_folder).replace('.p','.csv') for path in features_paths]
groundtruth_paths.sort()

extractors_type = list(pickle.load(open(features_paths[0],'rb'))['type_list_normalized'].keys())
extractors_entity = list(pickle.load(open(features_paths[0],'rb'))['entity_list'].keys())

types = set()
for g in groundtruth_paths:
    truth_pd = pd.read_csv(g)
    types = types | set(truth_pd['type'])

types.remove(np.NAN)
types = list(types)
types.sort()

gt = getTypesListGT(groundtruth_paths)
gt_flatten = reduce(lambda x,y: x+y,gt)

ensemble_type_list = getOutputTypeList(features_paths)
ensemble_type_list_raw = getOutputTypeList(features_paths,raw=True)

extractors_type_lists_per_file = getExtractorsTypesLists(features_paths,extractors_type,base)
extractors_type_lists = {key:reduce(lambda x,y: x+y,extractors_type_lists_per_file[key]) for key in extractors_type_lists_per_file}

for_hist_micro = [[0,0,0] for ext in extractors_type_lists]
for_hist_micro.append([0,0,0])
for_hist_micro.append([0,0,0])
for_hist_macro = [[0,0,0] for ext in extractors_type_lists]
for_hist_macro.append([0,0,0])
for_hist_macro.append([0,0,0])
for j,key in enumerate(['f1','precision','recall']):
    recs = []
    for k,ext in enumerate(extractors_type):
        obj = getScoresToken(gt_flatten,extractors_type_lists[ext],scores[key])
        obj['extractor'] = ext 
        recs.append(obj)
        for_hist_micro[k][j] = obj['micro']
        for_hist_macro[k][j] = obj['macro']

    obj = getScoresToken(gt_flatten,ensemble_type_list,scores[key])
    obj['extractor'] = 'ensemble'
    for_hist_micro[-2][j] = obj['micro']
    for_hist_macro[-2][j] = obj['macro']
    recs.append(obj)
    obj = getScoresToken(gt_flatten,ensemble_type_list_raw,scores[key])
    obj['extractor'] = 'ensemble_raw'
    for_hist_micro[-1][j] = obj['micro']
    for_hist_macro[-1][j] = obj['macro']
    recs.append(obj)
    writer = pd.ExcelWriter(evaluation_folder_NER_bin+key+'.xlsx')
    df = pd.DataFrame(recs)[['extractor','Organization','Person','Place','Role','macro','micro']]
    df = df.round(2)
    print(key)
    display(df)
    df.to_excel(writer)
    writer.save()


generateHistogram(for_hist_micro,evaluation_folder_NER_bin+'micro_hist.png',extractors_type+['ensemble','esemble_raw'])
generateHistogram(for_hist_macro,evaluation_folder_NER_bin+'macro_hist.png',extractors_type+['ensemble','esemble_raw'])

def generateBRATFiles(output_folder,segments_names,texts_paths,offsets_per_file,types_per_file,texts_tokens_per_file):
    for i,t_list in enumerate(types_per_file):
        entities = getNamedEntities(t_list,offsets_per_file[i],texts_tokens_per_file[i])
        text = open(texts_paths[i]).read()
        output_path = output_folder + segments_names[i] + '.ann'
        formBRATandSave(entities,text,output_path)
        
def getOffsetListGT(groundtruth_paths):
    offset_df_per_file = list()
    for g in groundtruth_paths:
        recs_gt = pd.read_csv(g).to_dict(orient='records')
        offsets = [r['continue'] for r in recs_gt]
        offset_df_per_file.append(offsets)
    return offset_df_per_file

def getTextTokensList(texts_paths):
    texts_splitted_per_file = list()
    for path in texts_paths:
        with open(path) as f:
            texts_splitted_per_file.append(splitInTokens(f.read()))
    return texts_splitted_per_file

def getNamedEntities(type_tokens,offsets,texts_tokens):
    entities = list()
    start_end = list()
    for i,t in enumerate(type_tokens):
        if t != '0':
            start = texts_tokens[i][1]
            end = texts_tokens[i][2]
            if bool(start_end):
                start_end[1] = end
            else:
                start_end = [start,end]
            if bool(offsets[i]):
                pass
            else:
                entities.append({
                    'type':t,
                    'start':start_end[0],
                    'end':start_end[1]
                })
                start_end = []
        else:
            start_end = []
        
    return entities

def formBRATandSave(entities,text,output_path):
    BASIC_BRAT_ANN = 'TINDEX\tTYPE START END\tSURFACE'
    brat_lines = list()
    for i,e in enumerate(entities):
        index_ = str(i+1)
        type_ = e['type']
        start_ = str(e['start'])
        end_ = str(e['end']+1)
        text_ = text[e['start']:e['end']+1]
        b_l = BASIC_BRAT_ANN.replace('INDEX',index_).replace('TYPE',type_).replace('START',start_).replace('END',end_).replace('SURFACE',text_)
        brat_lines.append(b_l)
    with open(output_path,'w+') as f_out:
        f_out.write('\n'.join(brat_lines))

text_folder = training_folder + 'test/txt_files/'
features_folder_test = training_folder + 'test/features_files/'
texts_paths = [path.replace(features_folder_test,text_folder).replace('.p','.txt') for path in features_paths]
texts_paths.sort()
segments = [t.split('/')[-1].split('.')[0] for t in groundtruth_paths]
gt_offset = getOffsetListGT(groundtruth_paths)
brat_truth_folder  = training_folder + 'test/brat_ground_truth/'
texts_splitted_per_file = getTextTokensList(texts_paths)
createFolder(brat_truth_folder)
generateBRATFiles(brat_truth_folder,
                  segments,texts_paths,
                  gt_offset,gt,texts_splitted_per_file)



evaluation_folder_NER_brat = training_folder + 'test/brat_extractors/'
createFolder(evaluation_folder_NER_brat)
scores_brat = ['pre','rec','fsc']


brat_extractors_folder = training_folder + 'test/brat_extractors/'

def estimateContinueInfo(predicted):
    offsets = list()
    for j,type_ in enumerate(predicted):
        if type_ == '0':
            offsets.append(0)
        else:
            if j != len(predicted) - 1 and predicted[j+1] == type_:
                offsets.append(1)
            else:
                offsets.append(0)
    return offsets

def getTypeNamedEntities(records,text):
    type_entities = list()
    start = None
    for rec in records:
        if rec['offset'] == 1:
            start = rec['start']

        else:
            if bool(start):
                e = {
                    'surface':text[start:rec['end']+1],
                    'start':start,
                    'end':rec['end'],
                    'type':rec['type']
                }
                type_entities.append(e)
                start = None

    return type_entities

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

def formExtractorsBrat(features_paths,extractors_type_lists_per_file,types,brat_extractors_folder):
    texts_paths = [path.replace(features_folder_test,text_folder).replace('.p','.txt') for path in features_paths]
    texts_paths.sort()
    for z,ext in enumerate(extractors_type_lists_per_file):
        for i,f in enumerate(extractors_type_lists_per_file[ext]):
            createFolder(brat_extractors_folder +'/'+ext+'/')
            out_path = brat_extractors_folder +'/'+ext+'/'+features_paths[i].split('/')[-1].split('.')[0]+'.ann'
            offsets = estimateContinueInfo(f)
            text = open(texts_paths[i],'r').read()
            tokens = splitInTokens(text)
            records = list()
            for j,t in enumerate(tokens):
                records.append({
                    'surface':t[0],
                    'start':t[1],
                    'end':t[2],
                    'offset':offsets[j],
                    'type':f[j]
                })
            entities = getTypeNamedEntities(records,text)
            formBRATandSave(entities,out_path)

formExtractorsBrat(features_paths,extractors_type_lists_per_file,types,brat_extractors_folder)

from bratutils import agreement as a

standard_gold_paths = [brat_truth_folder+f for f in listdir(brat_truth_folder) if isfile(join(brat_truth_folder, f)) and '.ann' in f]

standard_gold_dict = {p.split('/')[-1].split('.')[0]:p for p in standard_gold_paths}


output_path_hist = evaluation_folder_NER_brat +'hist_brat.png'

output_path_xlsx = evaluation_folder_NER_brat+'brat_scores.xlsx'

eval_names = [f for f in listdir(brat_extractors_folder) if isdir(join(evaluation_folder_NER_brat, f))]


records = list()
hist_data = list()

to_divide = len(features_paths)
for n in eval_names:
    fold = brat_extractors_folder+n+'/'
    paths = [fold+f for f in listdir(fold) if isfile(join(fold, f)) and '.ann' in f]

    scores_dict = {score:0 for score in scores_brat}
    for p in paths:
        name_file = p.split('/')[-1].split('.')[0]
        standard_gold_p = standard_gold_dict[name_file]
        doc = a.Document(standard_gold_p)
        doc2 = a.Document(p)
        doc.make_gold()
        statistics = doc2.compare_to_gold(doc)
        for score in scores_dict:
            exec('sc = statistics.'+score)
            scores_dict[score] += sc

    rec_dict = {score:scores_dict[score]/to_divide for score in scores_dict}
    hist_data.append([scores_dict[score]/to_divide for score in ['fsc','pre','rec']])
    rec_dict['extractor'] = n
    records.append(rec_dict)


scores_dict = {score:0 for score in scores_brat}
for f in features_paths:
    name_file = f.split('/')[-1].split('.')[0]
    p = getOutputFile(f,'brat')
    standard_gold_p = standard_gold_dict[name_file]
    doc = a.Document(standard_gold_p)
    doc2 = a.Document(p)
    doc.make_gold()
    statistics = doc2.compare_to_gold(doc)
    print(p,standard_gold_p)
    for score in scores_dict:
        exec('sc = statistics.'+score)
        scores_dict[score] += sc


rec_dict = {score:scores_dict[score]/to_divide for score in scores_dict}
hist_data.append([scores_dict[score]/to_divide for score in ['fsc','pre','rec']])
rec_dict['extractor'] = 'ensemble'
records.append(rec_dict)

scores_dict = {score:0 for score in scores_brat}

for f in features_paths:
    name_file = f.split('/')[-1].split('.')[0]
    p = getOutputFile(f,'brat',to_append='_type_raw')
    standard_gold_p = standard_gold_dict[name_file]
    doc = a.Document(standard_gold_p)
    doc2 = a.Document(p)
    doc.make_gold()
    statistics = doc2.compare_to_gold(doc)
    for score in scores_dict:
        exec('sc = statistics.'+score)
        scores_dict[score] += sc


rec_dict = {score:scores_dict[score]/to_divide for score in scores_dict}
hist_data.append([scores_dict[score]/to_divide for score in ['fsc','pre','rec']])
rec_dict['extractor'] = 'ensemble_raw'
records.append(rec_dict)


df = pd.DataFrame(records)


generateHistogram(hist_data,output_path_hist,eval_names+['ensemble','ensemble_raw'])

df = df.round(2)
writer_1 = pd.ExcelWriter(output_path_xlsx)
df.to_excel(writer_1)
writer_1.save()



def getExtractorsEntityLists(features_paths,extractors_entity):
    extractors_entity_lists = dict()
    col = 'entity_list'
    for ext in extractors_entity:
        extractors_entity_lists[ext] = []
        for path in features_paths:
            features_dict= pickle.load(open(path,'rb'))
            list_ext = features_dict[col][ext]
            for i,l in enumerate(list_ext):
                if type(l) != str or l == '1':
                    list_ext[i] = '0'
            extractors_entity_lists[ext].append(list_ext)
    return extractors_entity_lists

def getScoresDisambiguation(standard_gold_list,predicted_list):
    for i,p in enumerate(predicted_list):
        if p in ['0','1']:
            predicted_list[i] = np.NAN
    for i,p in enumerate(standard_gold_list):
        if p in ['0','1']:
            standard_gold_list[i] = np.NAN      
    
    true_negative = 0 
    true_positive = 0
    false_negative = 0
    false_positive = 0
    for i in range(len(standard_gold_list)):
        item_gold, item = standard_gold_list[i],predicted_list[i]
        if type(item_gold) != str and type(item) != str:
            true_negative += 1
        elif type(item_gold) == str and type(item) != str:
            false_negative += 1
        elif type(item_gold) != str and type(item) == str:
            false_positive += 1
        elif type(item_gold) == str and type(item) == str:
            if item == item_gold:
                true_positive += 1
            else:
                false_positive += 1
                false_negative += 1
    precision = true_positive / (true_positive+false_positive)
    recall = true_positive / (true_positive+false_negative)
    f1 = 2*(precision* recall)/(precision + recall)
    score_obj = {
        'precision':precision,
        'recall':recall,
        'f1':f1
    }
    return score_obj



extractors_entity_lists_per_file = getExtractorsEntityLists(features_paths,extractors_entity)
extractors_entity_lists = {key:reduce(lambda x,y: x+y,extractors_entity_lists_per_file[key]) for key in extractors_entity_lists_per_file}

def getURISListGT(groundtruth_paths):
    uris_df_per_file = list()
    for g in groundtruth_paths:
        recs_gt = pd.read_csv(g).to_dict(orient='records')
        uris = list()
        for r in recs_gt:
            uri = r['wd_uri']
            uris.append(uri)
        uris_df_per_file.append(uris)
    return uris_df_per_file

gt_test = getURISListGT(groundtruth_paths)
gt_test_flatten = reduce(lambda x,y: x+y,gt_test)

ensemble_type_list = getOutputEntityList(features_paths)
ensemble_type_list_raw = getOutputEntityList(features_paths,raw=True)

for_hist = [[0,0,0] for ext in extractors_entity_lists]
for_hist.append([0,0,0])
for_hist.append([0,0,0])


recs = []
for k,ext in enumerate(extractors_entity):
    obj = getScoresDisambiguation(gt_test_flatten,extractors_entity_lists[ext])
    obj['extractor'] = ext 
    recs.append(obj)
    for_hist[k][0] = obj['f1']
    for_hist[k][1] = obj['precision']
    for_hist[k][2] = obj['recall']

obj = getScoresDisambiguation(gt_test_flatten,ensemble_type_list)
obj['extractor'] = 'ensemble'
for_hist[-2][0] = obj['f1']
for_hist[-2][1] = obj['precision']
for_hist[-2][2] = obj['recall']
recs.append(obj)
obj = getScoresDisambiguation(gt_test_flatten,ensemble_type_list_raw)
obj['extractor'] = 'ensemble_raw'
for_hist[-1][0] = obj['f1']
for_hist[-1][1] = obj['precision']
for_hist[-1][2] = obj['recall']
recs.append(obj)
writer = pd.ExcelWriter(evaluation_folder_NEL_bin+'scores.xlsx')
df = pd.DataFrame(recs)[['extractor','f1','precision','recall']]
df = df.round(2)
display(df)
df.to_excel(writer)
writer.save()


generateHistogram(for_hist,evaluation_folder_NEL_bin+'hist.png',extractors_entity+['ensemble','esemble_raw'])