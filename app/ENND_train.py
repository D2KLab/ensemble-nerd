import os
from os import listdir
from os.path import isfile, join
import pickle
import pandas as pd
import numpy as np
from functools import reduce
from keras.preprocessing import sequence
from keras.models import load_model
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense,Input,concatenate
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

from sklearn.metrics import f1_score as f1_score_func

from sklearn.metrics import accuracy_score


extractors_types=['alchemy', 'adel', 'opencalais', 'meaning_cloud','dandelion','dbspotlight', 'babelfy', 'textrazor']
extractors_disambiguation = ['dandelion','dbspotlight', 'babelfy', 'textrazor']

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
optimizer = 'adam'
#activation = 'softmax'
activation = 'sigmoid'
activation_middle = 'selu'
architecture = 'simple'
patience = 50
dim_concatenation = 5

try:
    base = sys.argv[1]
except:
    print('You have to specify the ground truth')
    exit()


optlist, args = getopt.getopt(sys.argv[2:], '',['reg_alpha=',
                                                'dropout=',
                                                'epochs=',
                                                'model_name=',
                                                'batch=',
                                                'eg_alpha=',
                                                'units=',
                                                'loss_function=',
                                                'saving_folder=',
                                                'features=',
                                                'optimizer=',
                                                'activation_middle=',
                                                'patience=',
                                                'activation='
                                                ])


for opt, arg in optlist:
        if opt == '--reg_alpha':
            reg_alpha = float(arg)
        elif opt == "--dropout":
            dropout = float(arg)
        elif opt == "--dropout_input":
            dropout_input = float(arg)
        elif opt == "--epochs":
            epochs = int(arg)
        elif opt == "--batch":
            batch = int(arg)
        elif opt == '--eg_alpha':
            eg_alpha = float(arg)
        if opt == '--units':
            units = int(arg)
        elif opt== "--loss_function":
            loss_function = arg
        elif opt == "--saving_folder":
            saving_folder = arg  
        elif opt == "--features":
            features= arg.split(',')
        elif opt == '--optimizer':
            optimizer = arg
        elif opt == "--activation_middle":
            activation_middle = arg
        elif opt == "--activation":
            activation = arg
        elif opt == "--patience":
            patience = int(arg)


features = set(['type', 'score', 'entity', 'fasttext'])


def get_similarities_dict(features_paths_train,features_paths_test):
    sim_dict = dict()
    features_paths = features_paths_train+features_paths_test
    for i,f in enumerate(features_paths):
        similarites = pickle.load(open(f,'rb'))['features']['entity_MATRIX']
        for item in similarites.items():
            sim_dict[(str(item[0][0]),str(item[0][1]))] = item[1]
            sim_dict[(str(item[0][1]),str(item[0][0]))] = item[1]
    return sim_dict


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


def setURI(t):
    if bool(t) and type(t)==str:
        return t
    else:
        return np.NAN
    
def getURISListPerExtractor(features_paths,extractor_list=['dandelion', 'dbspotlight', 'babelfy', 'textrazor']):
    uris_list_per_extractor = list()
    for j,f in enumerate(features_paths):
        uri_list_file = list()
        obj = pickle.load(open(f,'rb'))
        uri_list = obj['entity_list']
        uris_list_per_extractor.append([{ext:setURI(uri_list[ext][i]) for ext in uri_list if ext in extractor_list} for i in range(len(uri_list['dandelion']))])
    return uris_list_per_extractor

def ConcatenateArray(list_arrays):
    array = list()
    for l in list_arrays:
        array += l
    return array

def isSameUri(wd1,wd2):
    return int((type(wd1)==type(wd2)==float) or (wd1==wd2))


def builtXY_b(uris_list_per_extractor_flatten,extractors_disambiguation,gt_flatten):
    X = []
    Y = []
    candidates_list = []
    for i,line in enumerate(uris_list_per_extractor_flatten):
        
        candidates = set([line[ext] for ext in extractors_disambiguation if type(line[ext])==str])
        if True in [type(line[ext]) == float for ext in extractors_disambiguation]:
            candidates.add(np.NAN)
        candidates = list(candidates)
        candidates.sort(key=lambda x:str(x))
        X_p = [ConcatenateArray([[isSameUri(line[ext_1],uri)] for ext_1 in extractors_disambiguation])
               for uri in candidates]
        Y_p = [[isSameUri(uri,gt_flatten[i])]
               for uri in candidates]
        candidates_list.append(candidates)
        X += X_p
        Y += Y_p
    X = np.array(X)
    Y = np.array(Y)
    return X,Y,candidates_list

def builtXY(uris_list_per_extractor_flatten,extractors_disambiguation,gt_flatten,similarities_dict):
    X = []
    Y = []
    candidates_list = []
    for i,line in enumerate(uris_list_per_extractor_flatten):
        candidates = set([line[ext] for ext in extractors_disambiguation if type(line[ext])==str])
        if True in [type(line[ext]) == float for ext in extractors_disambiguation]:
            candidates.add(np.NAN)
        candidates = list(candidates)
        candidates.sort(key=lambda x:str(x))
        X_p = [ConcatenateArray([similarities_dict[(str(line[ext_1]),str(uri))] for ext_1 in extractors_disambiguation])
               for uri in candidates]
        Y_p = [[isSameUri(uri,gt_flatten[i])]
               for uri in candidates]
        candidates_list.append(candidates)
        X += X_p
        Y += Y_p
    X = np.array(X)
    Y = np.array(Y)
    return X,Y,candidates_list

def createModelObject(extractors_disambiguation,
                      X_test,X_train,Y_test,Y_train,
                      gt_test_flatten,gt_train_flatten,
                      uris_list_per_extractor_train_flatten,uris_list_per_extractor_test_flatten,features_paths_train,
                      features_paths_test,candidates_train,candidates_test
                     ):
    model_obj = {
        'features_paths_train':features_paths_train,
        'features_paths_test':features_paths_test,
        'extractors_disambiguation':extractors_disambiguation
    }
    
    model_obj['candidates_train'] = candidates_train
    model_obj['candidates_test'] = candidates_test
    model_obj['train_X']= np.array(X_train)
    model_obj['train_Y']= np.array(Y_train)
    model_obj['test_X']= np.array(X_test)
    model_obj['test_Y']= np.array(Y_test)
    return model_obj

def addExtractorsTypesRepresentation(model_obj,extractors_types):
    candidates_test_iter = iter(model_obj['candidates_test'])
    candidates_train_iter = iter(model_obj['candidates_train'])
    paths_test = model_obj['features_paths_test']
    paths_train = model_obj['features_paths_train']
    model_obj['extractors_types'] = extractors_types
    model_obj['type_dict_test'] = {ext:[] for ext in extractors_types}
    model_obj['type_dict_train'] = {ext:[] for ext in extractors_types}
    for p in paths_test:
        obj = pickle.load(open(p,'rb'))
        features_obj = obj['features']['type']
        flag_g = True
        for ext in extractors_types:
            if flag_g:
                repeats = [len(next(candidates_test_iter)) for line in obj['features']['type'][ext]]
                flag_g = False
            for j,line in enumerate(obj['features']['type'][ext]):
                for k in range(repeats[j]):
                    model_obj['type_dict_test'][ext].append(line)
    
    for p in paths_train:
        obj = pickle.load(open(p,'rb'))
        features_obj = obj['features']['type']
        flag_g = True
        for ext in extractors_types:
            if flag_g:
                repeats = [len(next(candidates_train_iter)) for line in obj['features']['type'][ext]]
                flag_g = False
            for j,line in enumerate(obj['features']['type'][ext]):
                for k in range(repeats[j]):
                    model_obj['type_dict_train'][ext].append(line)
    for ext in extractors_types:
        model_obj['type_dict_test'][ext] = np.array(model_obj['type_dict_test'][ext])
        model_obj['type_dict_train'][ext] = np.array(model_obj['type_dict_train'][ext])
    return model_obj
    

training_folder = 'data/training_data/'+base+'/'
ground_truth_folder_train = training_folder + 'train/csv_ground_truth/'
ground_truth_folder_test = training_folder + 'test/csv_ground_truth/'

features_folder_train = training_folder + 'train/features_files/'
features_folder_test = training_folder + 'test/features_files/'

text_folder_train = training_folder + 'train/txt_files/'
text_folder_test = training_folder + 'test/txt_files/'

features_paths_train = [features_folder_train+f for f in listdir(features_folder_train) if isfile(join(features_folder_train, f)) and '.p' in f]
features_paths_train.sort()
features_paths_test = [features_folder_test+f for f in listdir(features_folder_test) if isfile(join(features_folder_test, f)) and '.p' in f]
features_paths_test.sort()
groundtruth_paths_train = [path.replace(features_folder_train,ground_truth_folder_train).replace('.p','.csv') for path in features_paths_train]
groundtruth_paths_train.sort()
groundtruth_paths_test = [path.replace(features_folder_test,ground_truth_folder_test).replace('.p','.csv') for path in features_paths_test]
groundtruth_paths_test.sort()


gt_test = getURISListGT(groundtruth_paths_test)
gt_test_flatten = reduce(lambda x,y: x+y,gt_test)
gt_train = getURISListGT(groundtruth_paths_train)
gt_train_flatten = reduce(lambda x,y: x+y,gt_train)


uris_list_per_extractor_train = getURISListPerExtractor(features_paths_train)
uris_list_per_extractor_train_flatten = reduce(lambda x,y: x+y,uris_list_per_extractor_train)
uris_list_per_extractor_test = getURISListPerExtractor(features_paths_test)
uris_list_per_extractor_test_flatten = reduce(lambda x,y: x+y,uris_list_per_extractor_test)



similarities_dict = get_similarities_dict(features_paths_train,features_paths_test)
X_train,Y_train,candidates_train = builtXY(uris_list_per_extractor_train_flatten,extractors_disambiguation,gt_train_flatten,similarities_dict)
X_test,Y_test,candidates_test = builtXY(uris_list_per_extractor_test_flatten,extractors_disambiguation,gt_test_flatten,similarities_dict)

model_obj = createModelObject(extractors_disambiguation,
                      X_test,X_train,Y_test,Y_train,
                      gt_test_flatten,gt_train_flatten,
                      uris_list_per_extractor_train_flatten,uris_list_per_extractor_test_flatten,features_paths_train,
                      features_paths_test,candidates_train,candidates_test
                     )

model_obj = addExtractorsTypesRepresentation(model_obj,extractors_types)




saving_folder = 'data/models/disambiguation/'+base+'/'

try:
    os.makedirs(saving_folder)
except:
    pass
if set(features) == set(['type', 'score', 'entity', 'fasttext']):
    model_file_path = saving_folder+'model.h5'
else:
    model_file_path = saving_folder+'_'.join(features)+'_model.h5'


class TestCallback(Callback):
    def __init__(self):
        self.history_scores = list()
        self.f1_max = 0
        self.val_loss_min = 20000

    def on_epoch_end(self, epoch, logs={}):
        global model_obj
        x, y = model_obj['X_dict_test'],model_obj['Y_dict_test']
        val_loss = model.evaluate(x,y)[0]
        predicted_test = self.model.predict(x,verbose=0)
        score_obj = dict()


        score_obj['val_loss']=val_loss
        f1_score = f1_score_func(np.squeeze(model_obj['test_Y']),np.squeeze(predicted_test).round())
        score_obj['f1_score'] = f1_score

        print(score_obj,len(predicted_test))

        self.history_scores.append(score_obj)
        if ((f1_score > self.f1_max) and (val_loss <= self.val_loss_min)) or ((val_loss < self.val_loss_min) and (f1_score >= self.f1_max)):
            self.f1_max = f1_score
            self.val_loss_min = val_loss
            self.model.save(model_obj['path'])

    def on_train_end(self,logs={}):
        global model_obj
        model_obj['history_scores'] = self.history_scores
        md = load_model(model_obj['path'])
        predicted_test = md.predict(model_obj['X_dict_test'],verbose=0)
        predicted_train= md.predict(model_obj['X_dict_train'],verbose=0)
        model_obj['predicted_train'] = predicted_train
        model_obj['predicted_test'] = predicted_test



model_obj['path'] = model_file_path
 
dim_in_1 = model_obj['train_X'].shape[-2]
dim_in_2 = model_obj['train_X'].shape[-1]
dim_out = model_obj['train_Y'].shape[-1]

def generateConcactPartSimple(X,ext_name):
    dim_input = X.shape[-1]
    input_tensor = Input(shape=(X.shape[-1],), name=ext_name)
    dense_middle = Dense(dim_concatenation, activation=activation_middle,kernel_regularizer=l2(reg_alpha), bias_regularizer=l2(reg_alpha))(input_tensor)
    return input_tensor,dense_middle




if 'type_dict_test' not in model_obj:
    model = Sequential()
    if architecture == 'blstm':
        raise Exception('Not defined BLSTM')
    elif architecture == 'simple':
        model.add(Dropout(dropout_input,input_shape=(dim_in_2,)))
        for i in range(layers):
            model.add(Dense(units, activation=activation_middle,kernel_regularizer=l2(reg_alpha), bias_regularizer=l2(reg_alpha)))
            model.add(Dropout(dropout))
        model.add(Dense(dim_out, activation=activation,kernel_regularizer=l2(reg_alpha), bias_regularizer=l2(reg_alpha)))


    model.compile(loss=loss_function, optimizer=optimizer, metrics=['mae','accuracy'])
    print(model.summary())
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=0)
    model_obj['X_dict_test'] = model_obj['test_X']
    model_obj['Y_dict_test'] = model_obj['test_Y']
    model_obj['X_dict_train'] = model_obj['train_X']
    model.fit(model_obj['train_X'], model_obj['train_Y'], epochs=epochs, batch_size=batch,callbacks=[TestCallback(),early_stop],validation_data = (model_obj['test_X'], model_obj['test_Y']))

else:

    X_dict_train = dict()
    X_dict_train['entity'] = model_obj['train_X']
    for ext in model_obj['type_dict_train']:
        X_dict_train[ext] = model_obj['type_dict_train'][ext]

    X_dict_test= dict()
    X_dict_test['entity'] = model_obj['test_X']
    for ext in model_obj['type_dict_test']:
        X_dict_test[ext] = model_obj['type_dict_test'][ext]

    uris_tensor = Input(shape=(dim_in_2,), name='entity')
    uris_tensor_d = Dropout(dropout_input)(uris_tensor)
    to_concatenate_layers = [generateConcactPartSimple(X=model_obj['type_dict_train'][ext],ext_name=ext) for ext in model_obj['type_dict_train']]
    to_concatenate_layers_input = [uris_tensor]+[c[0] for c in to_concatenate_layers]
    to_concatenate_layers_out = [uris_tensor] + [c[1] for c in to_concatenate_layers]
    concatenation = concatenate(to_concatenate_layers_out)
    for i in range(layers):
        concatenation = Dense(units, activation=activation_middle,kernel_regularizer=l2(reg_alpha), bias_regularizer=l2(reg_alpha))(concatenation)
        concatenation = Dropout(dropout)(concatenation)
    main_output = Dense(dim_out, activation=activation,kernel_regularizer=l2(reg_alpha), bias_regularizer=l2(reg_alpha), name='main_output')(concatenation)
    model = Model(inputs=to_concatenate_layers_input, outputs=[main_output])

    model.compile(loss=loss_function, optimizer=optimizer, metrics=['mae','accuracy'])
    print(model.summary())
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=0)

    model_obj['X_dict_test'] = X_dict_test
    model_obj['Y_dict_test'] = {'main_output': model_obj['test_Y']}
    model_obj['X_dict_train'] = X_dict_train


    model.fit(X_dict_train,
          {'main_output': model_obj['train_Y']},
          validation_data=(X_dict_test, 
                 {'main_output': model_obj['test_Y']}),
          callbacks=[TestCallback(),early_stop],
          epochs=epochs, batch_size=batch)
    



