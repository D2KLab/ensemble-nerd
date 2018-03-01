from os import listdir
from os.path import isfile, join
import os
import pickle
import pandas as pd
import numpy as np
from keras.models import Model
from keras.preprocessing import sequence
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,Embedding
from keras.layers import GRU,Concatenate,Input,concatenate,Flatten,Reshape
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
from sklearn.metrics import f1_score

base = 'aida'

units = 400
reg_alpha = 0.000
eg_alpha=0.0 
epochs=1000
batch=50
dropout =0.3
loss_function = 'mse'
features=['type', 'score', 'uris', 'fasttext']
optimizer = 'adam'
activation = 'sigmoid'
activation_middle = 'relu'
extractors_types=['alchemy', 'adel', 'opencalais', 'meaning_cloud','dandelion', 'babelfy', 'textrazor']
patience = 100
types = set()



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
                                                'architecture=',
                                                'activation_middle=',
                                                'patience=',
                                                'types=',
                                                'activation=',
                                                'dropout_input=',
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

saving_folder = 'training_data/'+base+'/models/best/'
try:
    os.makedirs(saving_folder)
except:
    pass


def built_X_sample(features_obj,features,extractors):
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

def built_Y_sample(list_uris,ground_truth_pd,extractors,continue_flag=True,type_flag=True):
    right_types = list(ground_truth_pd['type'])
    right_continue = list(ground_truth_pd['continue'])
    #print('list_uris',list_uris)
    #print('right_uris',right_uris)
    Y_file = []
    for i in range(len(right_types)):
        type_ = right_types[i]
        if type_ in types_dict:
            Y_file_p = deepcopy(types_dict[type_])
        elif '0' in types_dict:
            Y_file_p = deepcopy(types_dict['0'])
        else:
            Y_file_p = [0 for k in range(len(types_dict))]
        #if 1 in Y_file_p:
         #   Y_file_p.append(0)
        #else:
         #   Y_file_p.append(1)
        if not type_flag:
            Y_file_p = []
        if continue_flag:
            continue_ = right_continue[i]
            Y_file_p.append(continue_)
        Y_file.append(Y_file_p)
    return np.array(Y_file)

def flatten_data(X):
    offset = len(X[0])
    final_X = []
    for x in X:
        for line in x:
            final_X.append(line)
    return np.array(final_X)

def built_XY_samples(features_paths,groundtruth_paths,max_fragment_len,
                     features=['type', 'score', 'uris', 'fasttext'],
                     extractors_types=['alchemy', 'adel', 'opencalais', 'meaning_cloud',
                                       'dandelion', 'dbspotlight', 'babelfy', 'textrazor'],
                     extractors_disambiguation=['dandelion', 'dbspotlight', 'babelfy', 'textrazor'],
                     continue_flag = True,
                     type_flag = True
                    ):

    X_dict = {}

    for i,f_p in enumerate(features_paths):
        #print(f_p)
        obj = pickle.load(open(f_p,'rb'))
        features_obj = obj['features']
        uris_list = obj['uris_list']


        for feat in features:
            if feat in ['type','score']:
                for ext in extractors_types:
                    X_file = built_X_sample(features_obj,[feat],[ext])
                    nil_X = np.zeros(X_file.shape[-1])
                    pad_length = max_fragment_len - X_file.shape[0]
                    nil_np_X = np.array((pad_length)*[nil_X])
                    if i!=0:
                        X_dict[feat+ext].append(np.append(X_file,nil_np_X,axis=0))
                    else:
                        X_dict[feat+ext]=[np.append(X_file,nil_np_X,axis=0)]
            elif feat == 'uris':
                for ext in extractors_disambiguation:
                    X_file = built_X_sample(features_obj,[feat],[ext])
                    nil_X = np.zeros(X_file.shape[-1])
                    pad_length = max_fragment_len - X_file.shape[0]
                    nil_np_X = np.array((pad_length)*[nil_X])
                    if i!=0:
                        X_dict[feat+ext].append(np.append(X_file,nil_np_X,axis=0))
                    else:
                        X_dict[feat+ext]=[np.append(X_file,nil_np_X,axis=0)]
            else:
                X_file = built_X_sample(features_obj,[feat],[])
                nil_X = np.zeros(X_file.shape[-1])
                pad_length = max_fragment_len - X_file.shape[0]
                nil_np_X = np.array((pad_length)*[nil_X])
                if i!=0:
                    X_dict[feat].append(np.append(X_file,nil_np_X,axis=0))
                else:
                    X_dict[feat]=[np.append(X_file,nil_np_X,axis=0)]              
        
        path_gt = groundtruth_paths[i]
        ground_truth_pd = pd.read_csv(path_gt)
        #print(len(uris_list['babelfy']),len(features_obj['type']['babelfy']),len(ground_truth_pd))
        Y_file = built_Y_sample(uris_list,ground_truth_pd,extractors_disambiguation,continue_flag=continue_flag,type_flag = type_flag)
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

    for key in X_dict:
        if key != 'fasttext.':
            X_dict[key] = flatten_data(np.asarray(X_dict[key]))
        else:
            X_dict[key] = np.asarray(X_dict[key])
    Y = flatten_data(np.asarray(Y))
    return X_dict,Y


def getTypesListCombination(predicted,inv_types_map):
    type_df_per_file = list()
    for j,f in enumerate(predicted):
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
        type_df_per_file.append(types)
    return type_df_per_file

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

def getTypeExtractor(ext,features_paths):
    type_df_per_file = list()
    for f in features_paths:
        obj_features = pickle.load(open( f, "rb" ) )
        type_list_obj = obj_features['type_list'][ext]
        types = list()
        for t in type_list_obj:
            if type(t) == str:
                type_ = t
            else:
                type_ = '0'
            types.append(type_)
        type_df_per_file.append(types)
    return type_df_per_file

def flatten_data(X):
    offset = len(X[0])
    final_X = []
    for x in X:
        for line in x:
            final_X.append(line)
    return np.array(final_X)

def de_flattenData(flat_X,max_fragment_len):
    return np.reshape(flat_X,(int(len(flat_X)/max_fragment_len),max_fragment_len,flat_X.shape[1]))
def deletePadding(comb,gt):
    for i,f in enumerate(comb):
        comb[i]=f[:len(gt[i])]
    return comb
 

def getScores(gt_list,p_list,types,return_flag=False):
    score_obj= dict()
    scores = f1_score(gt_list, p_list, labels=types,average=None)
    for i,t in enumerate(types):
        score_obj[t]=scores[i]
        print('F1 on type',t+':',scores[i])
    for avg in ['micro', 'macro', 'weighted']:
        sc = f1_score(gt_list, p_list, labels=types,average=avg)
        score_obj[avg]=sc
        print('Global F1','(',avg,')',':',sc)
    if return_flag:
        return score_obj

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

types_dict = {t:[int(i==j) for j in range(len(types))] for i,t in enumerate(types)}
inv_types_map = {tuple(v): k for k, v in types_dict.items()}

gt_test = getTypesListGT(groundtruth_paths_test)

extractors_disambiguation=['dandelion', 'babelfy', 'textrazor']

X_dict_train,train_Y = built_XY_samples(features_paths_train,groundtruth_paths_train,max_fragment_len,features=features,continue_flag=False,type_flag=True,extractors_types=extractors_types,extractors_disambiguation = extractors_disambiguation)

X_dict_test,test_Y = built_XY_samples(features_paths_test,groundtruth_paths_test,max_fragment_len,features=features,continue_flag=False,type_flag=True,extractors_types=extractors_types,extractors_disambiguation = extractors_disambiguation)

print('Parsed')

def generateConcactPartSimple(X,feat,ext_name,activation_middle,reg_alpha,eg_alpha,types):
    dim_input = X.shape[-1]
    dim_out = len(types)
    input_tensor = Input(shape=(X.shape[-1],), name=feat+ext_name)
    dense_middle = Dense(dim_out, activation=activation_middle,kernel_regularizer=l2(reg_alpha), bias_regularizer=l2(reg_alpha),name=feat+ext_name+'out')(input_tensor)
    return input_tensor,dense_middle


def generateConcactPartLSTM(X,feat,ext_name,activation_middle,reg_alpha,eg_alpha,types):
    dim_input_1 = X.shape[-2]
    dim_input_2 = X.shape[-1]
    dim_out = len(types)
    input_tensor = Input(shape=(dim_input_1,dim_input_2), name=feat+ext_name)
    x = Bidirectional(LSTM(units, return_sequences=True, \
                                        W_regularizer=l2(reg_alpha), \
                                        U_regularizer=l2(reg_alpha), \
                                        b_regularizer=l2(reg_alpha)))(input_tensor)
    x = Dropout(dropout)(x)
    dense_middle = TimeDistributed(Dense(dim_out, activation=activation_middle,kernel_regularizer=l2(reg_alpha), bias_regularizer=l2(reg_alpha)))(x)
    dense_middle = Reshape((dim_out,))(dense_middle)
    return input_tensor,dense_middle


model_obj = dict()
model_obj['train_X'] = X_dict_train
model_obj['train_Y'] = {'main_output': train_Y}
model_obj['test_X'] = X_dict_test
model_obj['test_Y'] = {'main_output': test_Y}
model_obj['path'] = saving_folder+'model.h5'
model_obj['inv_types_map'] = inv_types_map
model_obj['types_dict'] = types_dict
model_obj['max_fragment_len'] = max_fragment_len
model_obj['types'] = types

to_concatenate_layers = []
for feat in features:
    if feat in ['type','score']:
        for ext in extractors_types:
            to_concatenate_layers.append(generateConcactPartSimple(X_dict_train[feat+ext],feat,ext,activation_middle,reg_alpha,eg_alpha,types))
    elif feat == 'uris':
        for ext in extractors_disambiguation:
            to_concatenate_layers.append(generateConcactPartSimple(X_dict_train[feat+ext],feat,ext,activation_middle,reg_alpha,eg_alpha,types))
    elif feat == 'fasttext':
        to_concatenate_layers.append(generateConcactPartSimple(X_dict_train[feat],feat,'',activation_middle,reg_alpha,eg_alpha,types))





to_concatenate_layers_input = [c[0] for c in to_concatenate_layers]
to_concatenate_layers_out = [c[1] for c in to_concatenate_layers]
concatenation = concatenate(to_concatenate_layers_out)

main_output = Dense(len(types), activation=activation, name='main_output')(concatenation)

model = Model(inputs=to_concatenate_layers_input, outputs=[main_output])


model.compile(loss=loss_function, optimizer=optimizer, metrics=['mae','accuracy'])

print(model.summary())


class TestCallback(Callback):
    def __init__(self,model_obj, gt_test,saving_folder):
        self.model_obj = model_obj
        self.saving_folder = saving_folder
        self.gt_test = gt_test
        self.gt_test_flatten = reduce(lambda x,y: x+y,gt_test)
        self.max_fragment_len = model_obj['max_fragment_len']
        self.inv_types_map = model_obj['inv_types_map']
        self.history_scores = list()
        self.types = model_obj['types']
        self.f1_max = 0
        self.val_loss_min = 20000

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.model_obj['test_X'],self.model_obj['test_Y']
        predicted_test = self.model.predict(x,verbose=0)
        if len(predicted_test.shape) == 3 and predicted_test.shape[-1] == 1:
            predicted_test = np.squeeze(predicted_test)
        if len(predicted_test.shape) == 2:
            predicted_test = de_flattenData(predicted_test,self.max_fragment_len)
        comb_test = getTypesListCombination(predicted_test,self.inv_types_map)
        comb_test = deletePadding(comb_test,self.gt_test)
        comb_test_flatten = reduce(lambda x,y: x+y,comb_test)
        val_loss = model.evaluate(self.model_obj['test_X'],self.model_obj['test_Y'])[0]
        loss = model.evaluate(self.model_obj['train_X'],self.model_obj['train_Y'])[0]
        score_obj = getScores(self.gt_test_flatten,comb_test_flatten,self.types,return_flag=True)
        score_obj['val_loss']=val_loss
        score_obj['loss']=loss
        self.history_scores.append(score_obj)
        f1_score = score_obj['micro']
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

        self.model_obj['extractors_mapping'] = dict()
        

        for ext in extractors_types:
            layer_name = 'type'+ext+'out'
            intermediate_layer_model = Model(inputs=model.input,
                                             outputs=model.get_layer(layer_name).output)

            intermediate_output_test = intermediate_layer_model.predict(model_obj['test_X'])
            intermediate_output_test = de_flattenData(intermediate_output_test,self.max_fragment_len)
            comb_test = getTypesListCombination(intermediate_output_test,self.inv_types_map)

            intermediate_output_train = intermediate_layer_model.predict(model_obj['train_X'])
            intermediate_output_train = de_flattenData(intermediate_output_train,self.max_fragment_len)
            comb_train = getTypesListCombination(intermediate_output_train,self.inv_types_map)

            self.model_obj['extractors_mapping'][ext] = {
                'test':comb_test,
                'train':comb_train
            }

        pickle.dump( self.model_obj, open( self.saving_folder+'model_obj.p', "wb" ) )


early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=0)

#model.fit(model_obj['train_X'], model_obj['train_Y'], epochs=epochs, batch_size=batch,callbacks=[TestCallback(model_obj, gt_test,saving_folder),early_stop],validation_data = (model_obj['test_X'], model_obj['test_Y']))



model.fit(X_dict_train,
          {'main_output': train_Y},
          validation_data=(X_dict_test, 
                 {'main_output': test_Y}),
          callbacks=[TestCallback(model_obj, gt_test,saving_folder),early_stop],
          epochs=epochs, batch_size=batch)



