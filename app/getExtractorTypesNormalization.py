import os
import pickle
import sys
from copy import deepcopy
from functools import reduce
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.layers.core import Dropout
from keras.models import Sequential
from keras.models import load_model
from keras.regularizers import l2
from sklearn.metrics import f1_score

try:
    base = sys.argv[1]
except:
    print('You have to specify the ground truth')
    exit()

saving_folder = 'data/models/recognition/' + base + '/single_extractors/'

try:
    os.makedirs(saving_folder)
except:
    pass

reg_alpha = 0.000
dropout = 0.45
dropout_input = 0.0
epochs = 1000
batch = 50
eg_alpha = 0.0
units = 400
layers = 2
# loss_function = 'categorical_crossentropy'
loss_function = 'mse'
continue_flag = False
type_flag = True
optimizer = 'adam'
# activation = 'softmax'
activation = 'sigmoid'
activation_middle = 'relu'
extractors_types = ['alchemy', 'adel', 'opencalais', 'meaning_cloud', 'dandelion', 'dbspotlight', 'babelfy',
                    'textrazor']
architecture = ''
patience = 50
types = set()
O_type = False


def built_X_sample(features_obj, features, extractors):
    for f in features:
        if type(features_obj[f]) == dict:
            for extractor in features_obj[f]:
                if extractor in extractors:
                    try:
                        X_file = np.append(X_file, features_obj[f][extractor], axis=1)
                    except:
                        X_file = features_obj[f][extractor]
        else:
            try:
                # print(len(X_file))
                X_file = np.append(X_file, features_obj[f], axis=1)
            except:
                X_file = features_obj[f]

    return X_file


def built_Y_sample(ground_truth_pd, extractors, continue_flag=True, type_flag=True):
    right_types = list(ground_truth_pd['type'])
    right_continue = list(ground_truth_pd['continue'])
    # print('list_uris',list_uris)
    # print('right_uris',right_uris)
    Y_file = []
    for i in range(len(right_types)):
        type_ = right_types[i]
        if type_ in types_dict:
            Y_file_p = deepcopy(types_dict[type_])
        elif '0' in types_dict:
            Y_file_p = deepcopy(types_dict['0'])
        else:
            Y_file_p = np.zeros(len(types_dict))
        # if 1 in Y_file_p:
        #   Y_file_p.append(0)
        # else:
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


def de_flattenData(flat_X, max_fragment_len):
    return np.reshape(flat_X, (int(len(flat_X) / max_fragment_len), max_fragment_len, flat_X.shape[1]))


def fromPredictedToList(predicted_test, inv_types_map, max_fragment_len, gt, flat=True):
    predicted_test_flatten = de_flattenData(predicted_test, max_fragment_len)
    predicted_test_flatten = getTypesListCombination(predicted_test_flatten, inv_types_map)
    predicted_test_flatten = deletePadding(predicted_test_flatten, gt)
    if flat:
        predicted_test_flatten = reduce(lambda x, y: x + y, predicted_test_flatten)
    return predicted_test_flatten


def built_XY_samples(features_paths, groundtruth_paths, max_fragment_len,
                     features=['type', 'score', 'uris', 'fasttext'],
                     extractors_types=['alchemy', 'adel', 'opencalais', 'meaning_cloud',
                                       'dandelion', 'dbspotlight', 'babelfy', 'textrazor'],
                     extractors_disambiguation=['dandelion', 'dbspotlight', 'babelfy', 'textrazor'],
                     continue_flag=True,
                     type_flag=True
                     ):
    for i, f_p in enumerate(features_paths):
        # print(f_p)
        obj = pickle.load(open(f_p, 'rb'))
        features_obj = obj['features']
        X_file = built_X_sample(features_obj, features, extractors_types)
        if i != 0:
            pad_length = max_fragment_len - X_file.shape[0]
            nil_np_X = np.array((pad_length) * [nil_X])
            X.append(np.append(X_file, nil_np_X, axis=0))
        else:
            nil_X = np.zeros(X_file.shape[-1])
            pad_length = max_fragment_len - X_file.shape[0]
            nil_np_X = np.array((pad_length) * [nil_X])
            X = [np.append(X_file, nil_np_X, axis=0)]

        path_gt = groundtruth_paths[i]
        ground_truth_pd = pd.read_csv(path_gt)
        # print(len(uris_list['babelfy']),len(features_obj['type']['babelfy']),len(ground_truth_pd))
        Y_file = built_Y_sample(ground_truth_pd, extractors_disambiguation, continue_flag=continue_flag,
                                type_flag=type_flag)
        # print(Y_file)
        # print(list(X_file[0]))
        if i != 0:
            pad_length = max_fragment_len - Y_file.shape[0]
            nil_np_Y = np.array((pad_length) * [nil_Y])
            Y.append(np.append(Y_file, nil_np_Y, axis=0))
        else:
            nil_Y = np.zeros(Y_file.shape[-1])
            pad_length = max_fragment_len - Y_file.shape[0]
            nil_np_Y = np.array((pad_length) * [nil_Y])
            Y = [np.append(Y_file, nil_np_Y, axis=0)]
    return np.asarray(X), np.asarray(Y)


def getTypesListCombination(predicted, inv_types_map):
    type_df_per_file = list()
    for j, f in enumerate(predicted):
        types = list()
        for k, line in enumerate(f):
            i_max = list(line).index(max(line))
            line_round = line.round()
            if tuple(line_round) in inv_types_map:
                type_ = inv_types_map[tuple(line_round)]
            elif 1 in line_round:
                type_ = inv_types_map[tuple([int(i_max == n) for n in range(len(line_round))])]
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


def getTypeExtractor(ext, features_paths):
    type_df_per_file = list()
    for f in features_paths:
        obj_features = pickle.load(open(f, "rb"))
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


def deletePadding(comb, gt):
    for i, f in enumerate(comb):
        comb[i] = f[:len(gt[i])]
    return comb


def getScores(gt_list, p_list, types, return_flag=False):
    score_obj = dict()
    scores = f1_score(gt_list, p_list, labels=types, average=None)
    for i, t in enumerate(types):
        score_obj[t] = scores[i]
        print('F1 on type', t + ':', scores[i])
    for avg in ['micro', 'macro', 'weighted']:
        sc = f1_score(gt_list, p_list, labels=types, average=avg)
        score_obj[avg] = sc
        print('Global F1', '(', avg, ')', ':', sc)
    if return_flag:
        return score_obj


training_folder = 'data/training_data/' + base + '/'
ground_truth_folder_train = training_folder + 'train/csv_ground_truth/'
ground_truth_folder_test = training_folder + 'test/csv_ground_truth/'

features_folder_train = training_folder + 'train/features_files/'
features_folder_test = training_folder + 'test/features_files/'

features_paths_train = [features_folder_train + f for f in listdir(features_folder_train) if
                        isfile(join(features_folder_train, f)) and '.p' in f]
features_paths_train.sort()
features_paths_test = [features_folder_test + f for f in listdir(features_folder_test) if
                       isfile(join(features_folder_test, f)) and '.p' in f]
features_paths_test.sort()
groundtruth_paths_train = [path.replace(features_folder_train, ground_truth_folder_train).replace('.p', '.csv') for path
                           in features_paths_train]
groundtruth_paths_train.sort()
groundtruth_paths_test = [path.replace(features_folder_test, ground_truth_folder_test).replace('.p', '.csv') for path in
                          features_paths_test]
groundtruth_paths_test.sort()
max_fragment_len = max([len(pd.read_csv(g)) for g in groundtruth_paths_train + groundtruth_paths_test]) + 5
if not bool(types):
    for g in groundtruth_paths_train + groundtruth_paths_test:
        truth_pd = pd.read_csv(g)
        types = types | set(truth_pd['type'])

    types.remove(np.NAN)
    types = list(types)
types.sort()

O_list = []
types_dict = {t: [int(i == j) for j in range(len(types + O_list))] for i, t in enumerate(types + O_list)}
inv_types_map = {tuple(v): k for k, v in types_dict.items()}

for ext in extractors_types:
    train_X, train_Y = built_XY_samples(features_paths_train, groundtruth_paths_train, max_fragment_len,
                                        features=['type'], continue_flag=continue_flag, type_flag=type_flag,
                                        extractors_types=[ext])
    test_X, test_Y = built_XY_samples(features_paths_test, groundtruth_paths_test, max_fragment_len, features=['type'],
                                      continue_flag=continue_flag, type_flag=type_flag, extractors_types=[ext])


    class TestCallback(Callback):
        def __init__(self, model_obj, gt_test, max_fragment_len, inv_types_map, types, saving_folder):
            self.model_obj = model_obj
            self.saving_folder = saving_folder
            self.gt_test = gt_test
            self.gt_test_flatten = reduce(lambda x, y: x + y, gt_test)
            self.max_fragment_len = max_fragment_len
            self.inv_types_map = inv_types_map
            self.history_scores = list()
            self.types = types
            self.f1_max = 0
            self.val_loss_min = 20000

        def on_epoch_end(self, epoch, logs={}):
            x, y = self.model_obj['test_X'], self.model_obj['test_Y']
            predicted_test = self.model.predict(x, verbose=0)
            if len(predicted_test.shape) == 2:
                predicted_test = de_flattenData(predicted_test, self.max_fragment_len)
            comb_test = getTypesListCombination(predicted_test, self.inv_types_map)
            comb_test = deletePadding(comb_test, self.gt_test)
            comb_test_flatten = reduce(lambda x, y: x + y, comb_test)
            val_loss = model.evaluate(self.model_obj['test_X'], self.model_obj['test_Y'])[0]
            score_obj = getScores(self.gt_test_flatten, comb_test_flatten, self.types, return_flag=True)
            score_obj['val_loss'] = val_loss
            self.history_scores.append(score_obj)
            f1_score = score_obj['micro']
            if ((f1_score > self.f1_max) and (val_loss <= self.val_loss_min)) or (
                    (val_loss < self.val_loss_min) and (f1_score >= self.f1_max)):
                self.f1_max = f1_score
                self.val_loss_min = val_loss
                self.model.save(self.model_obj['path'])

        def on_train_end(self, logs={}):
            self.model_obj['history_scores'] = self.history_scores
            md = load_model(model_obj['path'])
            predicted_test = md.predict(model_obj['test_X'], verbose=0)
            if len(predicted_test.shape) == 2:
                predicted_test = de_flattenData(predicted_test, self.max_fragment_len)
            comb_test = getTypesListCombination(predicted_test, self.inv_types_map)
            comb_test = deletePadding(comb_test, self.gt_test)
            for i, path in enumerate(self.model_obj['features_paths_test']):
                features_dict = pickle.load(open(path, 'rb'))
                if 'type_list_normalized' not in features_dict:
                    features_dict['type_list_normalized'] = dict()
                features_dict['type_list_normalized'][self.model_obj['ext']] = comb_test[i]
                pickle.dump(features_dict, open(path, "wb"))


    gt_test = getTypesListGT(groundtruth_paths_test)

    print('shape', train_X.shape)
    model_obj = dict()
    model_obj['train_X'] = train_X
    model_obj['train_Y'] = train_Y
    model_obj['test_X'] = test_X
    model_obj['test_Y'] = test_Y
    model_obj['path'] = saving_folder + ext + '_model.h5'
    model_obj['inv_types_map'] = inv_types_map
    model_obj['types_dict'] = types_dict
    model_obj['max_fragment_len'] = max_fragment_len
    model_obj['types'] = types
    model_obj['ext'] = ext
    model_obj['features_paths_test'] = features_paths_test

    dim_in_1 = model_obj['train_X'].shape[1]
    dim_in_2 = model_obj['train_X'].shape[2]
    dim_out = model_obj['train_Y'].shape[2]

    model = Sequential()

    model_obj['train_X'] = flatten_data(model_obj['train_X'])
    model_obj['train_Y'] = flatten_data(model_obj['train_Y'])
    model_obj['test_X'] = flatten_data(model_obj['test_X'])
    model_obj['test_Y'] = flatten_data(model_obj['test_Y'])
    model.add(Dropout(dropout_input, input_shape=(dim_in_2,)))
    for i in range(layers):
        model.add(Dense(units, activation=activation_middle, kernel_regularizer=l2(reg_alpha),
                        bias_regularizer=l2(reg_alpha)))
        model.add(Dropout(dropout))
    model.add(Dense(dim_out, activation=activation, kernel_regularizer=l2(reg_alpha), bias_regularizer=l2(reg_alpha)))

    print(model_obj['train_X'].shape)
    model.compile(loss=loss_function, optimizer=optimizer, metrics=['mae', 'accuracy'])
    print(model.summary())
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=0)
    model.fit(model_obj['train_X'], model_obj['train_Y'], epochs=epochs, batch_size=batch,
              callbacks=[TestCallback(model_obj, gt_test, max_fragment_len, inv_types_map, types, saving_folder),
                         early_stop], validation_data=(model_obj['test_X'], model_obj['test_Y']))
