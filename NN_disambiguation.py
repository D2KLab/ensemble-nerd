from os import listdir
from os.path import isfile, join
import os
import pickle
import pandas as pd
import numpy as np
from keras.preprocessing import sequence
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU,Concatenate
from keras.layers.core import Activation
from keras.regularizers import l2
from keras.layers.wrappers import TimeDistributed
from keras.layers.wrappers import Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout
from keras.callbacks import Callback
from sklearn.metrics import precision_recall_fscore_support,fbeta_score
import keras.backend as K
from keras.callbacks import ModelCheckpoint,EarlyStopping
from copy import deepcopy
from functools import reduce
import getopt
import sys


base = 'OKE2016'
saving_folder = 'OKE2016/models_disambiguation/'

reg_alpha = 0.000
dropout = 0.45
dropout_input = 0.0
epochs=1000
batch=50
eg_alpha=0.0 
units=400
layers=2
#loss_function = 'categorical_crossentropy'
loss_function = 'mse'
features=['type', 'score', 'uris', 'fasttext']
continue_flag = False
type_flag = True
optimizer = 'adam'
#activation = 'softmax'
activation = 'sigmoid'
activation_middle = 'relu'
extractors_types=['alchemy', 'adel', 'opencalais', 'meaning_cloud','dandelion', 'dbspotlight', 'babelfy', 'textrazor']
extractors_disambiguation = ['dandelion', 'dbspotlight', 'babelfy', 'textrazor']
architecture = 'blstm'
patience = 50
types = set()
O_type = False

optlist, args = getopt.getopt(sys.argv[1:], '',['reg_alpha=',
                                                'dropout=',
                                                'epochs=',
                                                'batch=',
                                                'eg_alpha=',
                                                'units=',
                                                'layers=',
                                                'loss_function=',
                                                'input_base=',
                                                'saving_folder=',
                                                'features=',
                                                'optimizer=',
                                                'type_filter=',
                                                'extractors_types=',
                                                'extractors_disambiguation=',
                                                'architecture=',
                                                'activation_middle=',
                                                'patience=',
                                                'types=',
                                                'activation=',
                                                'dropout_input=',
                                                'no_match_value'
                                                '0_type'
                                                ])

for opt, arg in optlist:
        if opt == '--reg_alpha':
            reg_alpha = float(arg)
        elif opt == "--dropout":
            dropout = float(arg)
        elif opt == "--epochs":
            epochs = int(arg)
        elif opt == "--batch":
            batch = int(arg)
        elif opt == '--eg_alpha':
            eg_alpha = float(arg)
        if opt == '--units':
            units = int(arg)
        elif opt == "--layers":
            layers = int(arg)
        elif opt== "--loss_function":
            loss_function = arg
        elif opt == "--input_base":
            base = arg
        elif opt == "--saving_folder":
            saving_folder = arg  
        elif opt == "--features":
            features= arg.split(',')
        elif opt == '--optimizer':
            optimizer = arg
        elif opt == '--type_filter':
            types = set(arg.split(','))
        elif opt == "--extractors_types":
            extractors_types= arg.split(',')
        elif opt == "--extractors_disambiguation":
            extractors_disambiguation= arg.split(',')
        elif opt == "--architecture":
            architecture = arg
        elif opt == "--activation_middle":
            activation_middle = arg
        elif opt == "--activation":
            activation = arg
        elif opt == "--patience":
            patience = int(arg)   
        elif opt == "--types":
            types = arg.split(',')
        elif opt == "--dropout_input":
            dropout_input = float(arg) 
        elif opt == "--0_type":
            O_type = True

try:
    os.makedirs(saving_folder)
except:
    pass


print(optlist)

def de_flattenData(flat_X,max_fragment_len):
    return np.reshape(flat_X,(int(len(flat_X)/max_fragment_len),max_fragment_len,flat_X.shape[1]))

def flatten_data(X):
    offset = len(X[0])
    final_X = []
    for x in X:
        for line in x:
            final_X.append(line)
    return np.array(final_X)

def built_X_sample(features_obj,features,extractors_types,extractors_disambiguation):
    for f in features:
        if type(features_obj[f]) == dict:
            for extractor in features_obj[f]:
                if f == 'uris':
                    extractors = extractors_disambiguation
                else:
                    extractors = extractors_types
                if extractor in extractors:
                    #if f == 'uris':
                     #   print(features_obj[f][extractor].shape)
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

def built_Y_sample(list_uris,ground_truth_pd,extractors,no_match_value=1):
    right_uris = list(ground_truth_pd['wd_uri'])
    #print('list_uris',list_uris)
    #print('right_uris',right_uris)
    Y_file = []
    for i in range(len(right_uris)):
        Y_file_p = []
        u_2 = right_uris[i]
        for key in extractors:
            u_1 = list_uris[key][i]
            if type(u_1) == type(u_2):
                if type(u_1) == float:
                    Y_file_p.append(1)
                else:
                    Y_file_p.append(int(u_1==u_2))
            else:
                Y_file_p.append(0)
        Y_file.append(Y_file_p)
    return np.array(Y_file)

def built_XY_samples(features_paths,groundtruth_paths,max_fragment_len,
                     features=['type', 'score', 'uris', 'fasttext'],
                     extractors_types=['alchemy', 'adel', 'opencalais', 'meaning_cloud',
                                       'dandelion', 'dbspotlight', 'babelfy', 'textrazor'],
                     extractors_disambiguation=['dandelion', 'dbspotlight', 'babelfy', 'textrazor']
                    ):
    for i,f_p in enumerate(features_paths):
        #print(f_p)
        obj = pickle.load(open(f_p,'rb'))
        features_obj = obj['features']
        uris_list = obj['uris_list']
        X_file = built_X_sample(features_obj,features,extractors_types,extractors_disambiguation)
        if i!=0:
            pad_length = max_fragment_len - X_file.shape[0]
            nil_np_X = np.array((pad_length)*[nil_X])
            X.append(np.append(X_file,nil_np_X,axis=0))
        else:
            nil_X = np.zeros(X_file.shape[-1])
            pad_length = max_fragment_len - X_file.shape[0]
            nil_np_X = np.array((pad_length)*[nil_X])
            X=[np.append(X_file,nil_np_X,axis=0)]
        
        path_gt = groundtruth_paths[i]
        ground_truth_pd = pd.read_csv(path_gt)
        #print(len(uris_list['babelfy']),len(features_obj['type']['babelfy']),len(ground_truth_pd))
        Y_file = built_Y_sample(uris_list,ground_truth_pd,extractors_disambiguation)
        #print(Y_file)
        #print(list(X_file[0]))
        if i!=0:
            pad_length = max_fragment_len - Y_file.shape[0]
            nil_np_Y = np.array((pad_length)*[nil_Y])
            Y.append(np.append(Y_file,nil_np_Y,axis=0))
        else:
            nil_Y = np.zeros(Y_file.shape[-1])
            pad_length = max_fragment_len - Y_file.shape[0]
            nil_np_Y = np.array((pad_length)*[nil_Y])
            Y= [np.append(Y_file,nil_np_Y,axis=0)]
    return np.asarray(X),np.asarray(Y)
        

def getScoresDisambiguation(standard_gold_list,predicted_list):
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

def getUrisListPerFile(predicted,features_paths):
    uri_list_end= list()
    predicted_round = predicted.round()
    for j,f in enumerate(predicted_round):
        f_p = features_paths[j]
        uri_list_per_file= []
        obj = pickle.load(open(f_p,'rb'))
        uri_list = obj['uris_list']
        for k,line in enumerate(f[:len(uri_list['dandelion'])]):
            line_dict = dict()
            if 1 in line:
                for i,l in enumerate(line):
                    uri= obj['uris_list'][extractors_disambiguation[i]][k]
                    if type(uri) == float:
                        if '0' in line_dict: 
                            line_dict['0'] += l
                        else:
                            line_dict['0'] = l
                    else:  
                        if uri in line_dict: 
                            line_dict[uri] += l
                        else:
                            line_dict[uri] = l
                #print(line_dict)
                selected = max(line_dict, key=line_dict.get)
                if selected == '0':
                    selected = np.NAN
            else:
                selected = np.NAN
            uri_list_per_file.append(selected)
        uri_list_end.append(uri_list_per_file)
    return uri_list_end


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

def getURISListPerExtractor(features_paths):
    uris_list_per_extractor = list()
    for j,f in enumerate(features_paths):
        uri_list_file = list()
        obj = pickle.load(open(f,'rb'))
        uri_list = obj['uris_list']
        uris_list_per_extractor.append([{ext:uri_list[ext][i] for ext in uri_list} for i in range(len(uri_list['dandelion']))])
    return uris_list_per_extractor

def getBestResult(uris_list_per_extractor,gt_test):
    best_result = list()
    for i,file in enumerate(gt_test):
        best_result_p = list()
        for j,uri in enumerate(file):
            if True in [uris_list_per_extractor[i][j][key]==uri for key in uris_list_per_extractor[i][j]]:
                best_result_p.append(uri)
            else:
                best_result_p.append(np.NAN)
        best_result.append(best_result_p)
    return best_result








training_folder = 'training_data/'+base+'/'
ground_truth_folder_train = training_folder + 'train/csv_ground_truth/'
ground_truth_folder_test = training_folder + 'test/csv_ground_truth/'

features_folder_train = training_folder + 'train/features_files/'
features_folder_test = training_folder + 'test/features_files/'

features_paths_train = [features_folder_train+f for f in listdir(features_folder_train) if isfile(join(features_folder_train, f)) and '.p' in f]
features_paths_train.sort()
features_paths_test = [features_folder_test+f for f in listdir(features_folder_test) if isfile(join(features_folder_test, f)) and '.p' in f]
features_paths_test.sort()
groundtruth_paths_train = [path.replace(features_folder_train,ground_truth_folder_train).replace('.p','.csv') for path in features_paths_train]
groundtruth_paths_train.sort()
groundtruth_paths_test = [path.replace(features_folder_test,ground_truth_folder_test).replace('.p','.csv') for path in features_paths_test]
groundtruth_paths_test.sort()
max_fragment_len = max([len(pd.read_csv(g)) for g in groundtruth_paths_train+groundtruth_paths_test]) + 5
if not bool(types):
    for g in groundtruth_paths_train+groundtruth_paths_test:
        truth_pd = pd.read_csv(g)
        types = types | set(truth_pd['type'])
        
    types.remove(np.NAN)
    types = list(types)
types.sort()
if O_type == True:
    O_list = ['0']
else:
    O_list = []
types_dict = {t:[int(i==j) for j in range(len(types+O_list))] for i,t in enumerate(types+O_list)}
inv_types_map = {tuple(v): k for k, v in types_dict.items()}

train_X,train_Y = built_XY_samples(features_paths_train,groundtruth_paths_train,max_fragment_len,features=features,extractors_types=extractors_types,extractors_disambiguation=extractors_disambiguation)

test_X,test_Y = built_XY_samples(features_paths_test,groundtruth_paths_test,max_fragment_len,features=features,extractors_types=extractors_types,extractors_disambiguation=extractors_disambiguation)

gt_test = getURISListGT(groundtruth_paths_test)
uris_list_per_extractor_test = getURISListPerExtractor(features_paths_test)
best_result_test = getBestResult(uris_list_per_extractor_test,gt_test)

model_obj = dict()
model_obj['train_X'] = train_X
model_obj['train_Y'] = train_Y
model_obj['test_X'] = test_X
model_obj['test_Y'] = test_Y
model_obj['path'] = saving_folder+'model.h5'
model_obj['max_fragment_len'] = max_fragment_len
model_obj['features_paths_test'] = features_paths_test

dim_in_1 = model_obj['train_X'].shape[1]
dim_in_2 = model_obj['train_X'].shape[2]
dim_out = model_obj['train_Y'].shape[2]


class TestCallback(Callback):
    def __init__(self,model_obj, gt_test,saving_folder,best_result_test):
        self.model_obj = model_obj
        self.saving_folder = saving_folder
        self.gt_test = gt_test
        self.gt_test_flatten = reduce(lambda x,y: x+y,gt_test)
        self.history_scores = list()
        self.f1_max = 0
        self.val_loss_min = 20000
        self.best_result_test = best_result_test
        self.best_result_test_flatten = reduce(lambda x,y: x+y,best_result_test)

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.model_obj['test_X'],self.model_obj['test_Y']
        predicted_test = self.model.predict(x,verbose=0)
        if len(predicted_test.shape) == 2:
            predicted_test = de_flattenData(predicted_test,self.model_obj['max_fragment_len'])
        comb_test = getUrisListPerFile(predicted_test,model_obj['features_paths_test'])
        #getScoresDisambiguation(standard_gold_list,predicted_list)
        comb_test_flatten = reduce(lambda x,y: x+y,comb_test)
        val_loss = model.evaluate(self.model_obj['test_X'],self.model_obj['test_Y'])[0]
        score_obj = getScoresDisambiguation(self.gt_test_flatten,comb_test_flatten)
        print('\nSCORE_1',score_obj)
        score_obj_2 = getScoresDisambiguation(self.best_result_test_flatten,comb_test_flatten)
        print('\nSCORE_2',score_obj_2)
        for key in score_obj_2:
            score_obj[key+'_reachable'] = score_obj_2[key]
        score_obj['val_loss']=val_loss
        self.history_scores.append(score_obj)
        f1_score = score_obj_2['f1']
        if ((f1_score > self.f1_max) and (val_loss <= self.val_loss_min)) or ((val_loss < self.val_loss_min) and (f1_score >= self.f1_max)):
            self.f1_max = f1_score
            self.val_loss_min = val_loss
            self.model.save(self.model_obj['path'])

    def on_train_end(self,logs={}):
        self.model_obj['history_scores'] = self.history_scores
        md = load_model(model_obj['path'])
        predicted_test = md.predict(model_obj['test_X'],verbose=0)
        predicted_train= md.predict(model_obj['train_X'],verbose=0)
        self.model_obj['predicted_train'],self.model_obj['predicted_test'] = predicted_train,predicted_test
        pickle.dump( self.model_obj, open( self.saving_folder+'model_obj.p', "wb" ) )





model = Sequential()
if architecture == 'blstm':
    model.add(Dropout(dropout_input,input_shape=(dim_in_1,dim_in_2)))
    for i in range(layers):
        model.add(Bidirectional(LSTM(units, return_sequences=True, \
                                        W_regularizer=l2(reg_alpha), \
                                        U_regularizer=l2(reg_alpha), \
                                        b_regularizer=l2(reg_alpha))))
        model.add(Dropout(dropout))
    model.add(TimeDistributed(Dense(dim_out, activation=activation, \
                                    W_regularizer=l2(reg_alpha), \
                                    b_regularizer=l2(reg_alpha))))
elif architecture == 'simple':
    model_obj['train_X'] = flatten_data(model_obj['train_X'])
    model_obj['train_Y'] = flatten_data(model_obj['train_Y'])
    model_obj['test_X'] = flatten_data(model_obj['test_X'])
    model_obj['test_Y'] = flatten_data(model_obj['test_Y'])
    model.add(Dropout(dropout_input,input_shape=(dim_in_2,)))
    for i in range(layers):
        model.add(Dense(units, activation=activation_middle,kernel_regularizer=l2(reg_alpha), bias_regularizer=l2(reg_alpha)))
        model.add(Dropout(dropout))
    model.add(Dense(dim_out, activation=activation,kernel_regularizer=l2(reg_alpha), bias_regularizer=l2(reg_alpha)))

print(model_obj['train_X'].shape)
model.compile(loss=loss_function, optimizer=optimizer, metrics=['mae','accuracy'])
print(model.summary())
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=0)
model.fit(model_obj['train_X'], model_obj['train_Y'], epochs=epochs, batch_size=batch,callbacks=[TestCallback(model_obj, gt_test,saving_folder,best_result_test),early_stop],validation_data = (model_obj['test_X'], model_obj['test_Y']))





