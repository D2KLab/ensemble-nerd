from igraph import *
import pickle
import json
from api_pkg.utils.request import *
from api_pkg.utils.metrics import *
from api_pkg.utils.tokenization import *
from api_pkg import dandelion,dbspotlight,opencalais,babelfy,adel,meaning_cloud,alchemy,textrazor
from itertools import combinations
from langdetect import detect
import time



EMBEDDING_DATA_PATH = 'data/embedding_data/'
WIKIDATA_CLASSES = pickle.load(open(EMBEDDING_DATA_PATH+'all_classes.p','rb'))
WIKIDATA_DICT = json.load(open(EMBEDDING_DATA_PATH+'wikidata_types_mapping.json'))
VIRTUAL_NODES = json.load(open(EMBEDDING_DATA_PATH+'vitual_nodes.json'))
PATH_MAPPING_TYPES = "data/types_representation/types_mapping.json"

MAPPING_TYPES_OBJ = json.load(open(PATH_MAPPING_TYPES))


def getGraphNodes_and_Properties():
    embdedding_specifications_dict = json.load(open(EMBEDDING_DATA_PATH+'embedding_specifications.json'))
    graph_path = embdedding_specifications_dict['structural']
    graph_obj = pickle.load(open(graph_path,'rb'))
    MAX_DISTANCE_NODES = graph_obj['max_distance']
    structural_graph = graph_obj['graph']
    properties_list = embdedding_specifications_dict['semantic']
    return structural_graph,properties_list,MAX_DISTANCE_NODES

structural_graph,properties_list,MAX_DISTANCE_NODES = getGraphNodes_and_Properties()


def getWikidataTypeRepresentation(types_ordered_list):
    features_array = list()
    n_features = len(WIKIDATA_DICT.keys())
    resource_dict_representation = {t:[0 for i in range(n_features)]+[int(t not in WIKIDATA_CLASSES)] for t in (set(types_ordered_list)-{np.NAN})}
    for i,tag in enumerate(WIKIDATA_DICT.keys()):
        set_intersection = set(WIKIDATA_DICT[tag]) & set(resource_dict_representation.keys())
        for r in set_intersection:
            resource_dict_representation[r][i] = 1
    for t in types_ordered_list:
        if t in resource_dict_representation:
            features_array.append(resource_dict_representation[t])
        else:
            features_array.append([0 for i in range(n_features)]+[0])
    return np.array(features_array)





def getTypeRepresentation(types_ordered_list,ontology_name,mapping_types_obj=MAPPING_TYPES_OBJ,raise_flag=True):
    if ontology_name == 'wikidata':
        return getWikidataTypeRepresentation(types_ordered_list)
    else:
        features_array = list()
        ontology_path = MAPPING_TYPES_OBJ[ontology_name]
        ontology_representation = json.load(open(ontology_path))
        n_features = len(list(ontology_representation.values())[0])
        for t_info in types_ordered_list:
            vector = [0 for i in range(n_features)]
            if type(t_info)==str:
                for t in t_info.split(','):
                    if t in ontology_representation:
                        vector_p = ontology_representation[t]
                        vector = [min([x + y,1]) for x, y in zip(vector, vector_p)]
                    elif '-1' in ontology_representation:
                        print('AIA',ontology_name,t)
                        vector_p = ontology_representation['-1']
                        vector = [min([x + y,1]) for x, y in zip(vector, vector_p)]
            features_array.append(vector)
        return np.array(features_array)
'''

    types_ordered_list1 = [a['uri'] for a in annotations]
    type_features1 = getTypeRepresentation(types_ordered_list1,self.ontology_uri)
    types_ordered_list2 = [a['type'] for a in annotations]
    type_features2 = getTypeRepresentation(types_ordered_list2,self.ontology_type)
    type_features = np.append(type_features1, type_features2, axis=1)


'''

def getTypeFeatures(type_list_dict,entity_list_dict,ontology_type_dict,ontology_entity_dict):
    type_features_obj = {}
    extractors_names = set(type_list_dict.keys()) | set(entity_list_dict.keys())
    for ext in extractors_names:
        if ext in type_list_dict and ext in entity_list_dict:
            type_features_obj[ext] = np.append(getTypeRepresentation(type_list_dict[ext],ontology_type_dict[ext]),
             getTypeRepresentation(entity_list_dict[ext],ontology_entity_dict[ext]), axis=1)
        elif ext in type_list_dict:
            type_features_obj[ext] = getTypeRepresentation(type_list_dict[ext],ontology_type_dict[ext])
        elif ext in entity_list_dict:
            type_features_obj[ext] = getTypeRepresentation(entity_list_dict[ext],ontology_entity_dict[ext])
    return type_features_obj





def getDistanceNodes(wd_id_1,wd_id_2,structural_graph):
    virtuals = 0
    try:
        n1 = VIRTUAL_NODES[n1]
        virtuals+=1
    except:
        n1 = wd_id_1
    try:
        n2 = VIRTUAL_NODES[n2]
        virtuals+=1
    except:
        n2 = wd_id_2
    if virtuals == 2:
        if wd_id_1 != wd_id_2:
            dist = 2
        else:
            dist = 0
    else:
        dist = structural_graph.shortest_paths(n1,n2)[0][0] + virtuals
    return dist

def getStructSimilarity(wd_id_1,wd_id_2,structural_graph,MAX_DISTANCE_NODES):
    if type(wd_id_1) != str or type(wd_id_2) != str:
        return 0.0
    else:
        try:
            d_struct = min([1.0,getDistanceNodes(wd_id_1,wd_id_2,structural_graph)/MAX_DISTANCE_NODES])
        except:
            return 0.0
        sim_struct = abs(1.0 - d_struct)
        return sim_struct

def assignMetric(s):
    if s == "one-hot":
        return oneHotSimilarity
    elif s == 'fuzzy':
        return fuzzSimilarity
    elif s == 'tf-idf':
        return ffIdfSimilarity

def getLabelsFeaturesToken(labels,metric=fuzzSimilarity):
    label_features = []
    for t in combinations(labels, 2):
        l1,l2 = t
        label_features.append(metric(l1,l2))
    return label_features
    
def getSematicSimilarity(wd_id_1,wd_id_2,properties,lang):
    semantic_similarity_array = []
    for p in properties:
        if 'filter' in p:
            items_1 = set()
            items_2 = set()
            if lang in p['filter']:
                items_1 = items_1 | set(wikiQuery(wd_id_1,p['property'],filter_q=p['filter'][lang])['o'])
                items_2 = items_2 | set(wikiQuery(wd_id_2,p['property'],filter_q=p['filter'][lang])['o'])
        else:
            items_1 = set(wikiQuery(wd_id_1,p['property'])['o'])
            items_2 = set(wikiQuery(wd_id_2,p['property'])['o'])
        sim_func = assignMetric(p['metric'])
        similarities = [sim_func(i1,i2) for i2 in items_2 for i1 in items_1]
        if len(similarities)!=0:
            semantic_similarity_array.append(max(similarities))
        else:
            semantic_similarity_array.append(0.0)
    return semantic_similarity_array


def getUrisSimilarityVector(wd_id_1,wd_id_2,G=structural_graph,properties_list=properties_list,MAX_DISTANCE_NODES=MAX_DISTANCE_NODES,lang='fr'):
    if type(wd_id_1) == str and type(wd_id_2) == str:
        sim_struct = getStructSimilarity(wd_id_1,wd_id_2,structural_graph,MAX_DISTANCE_NODES)
        sim_semantic = getSematicSimilarity(wd_id_1,wd_id_2,properties_list,lang)
        return np.array([sim_struct] + sim_semantic)
    else:
        return np.array([0 for i in range(len(properties_list)+1)])



def getEntityFeatures(entity_list_dict,ontology_entity_dict,lang):
    PRECOMPUTED_SIM = {}
    extractors_uris = list(entity_list_dict.keys())
    com_ext = [sorted(item) for item in list(combinations(extractors_uris,2))] + [(ext,ext) for ext in extractors_uris]


    entity_features =  {ext1:{ext2:[] for ext2 in extractors_uris} for ext1 in extractors_uris}
    entity_MATRIX= dict()

    length_list = len(entity_list_dict[extractors_uris[0]])
    for k in range(length_list):
        for comb in com_ext:
            uri1,uri2 = entity_list_dict[comb[0]][k],entity_list_dict[comb[1]][k]
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
            if (uri1,uri2) not in entity_MATRIX:
                entity_MATRIX[(uri1,uri2)] = sim
            if (uri2,uri1) not in entity_MATRIX:
                entity_MATRIX[(uri2,uri1)] = sim
            entity_features[comb[0]][comb[1]].append(sim)
            if comb[0] != comb[1]:
                entity_features[comb[1]][comb[0]].append(sim)

    for ext in entity_features:
        for z,key in enumerate(extractors_uris):
            if z!=0:
                X_ext = np.append(X_ext,entity_features[ext][key],axis=1)
            else:
                X_ext = np.array(entity_features[ext][key])
        entity_features[ext] = X_ext
    return entity_features,entity_MATRIX







def fromOntologyEdglist_to_ClassRapresentation_notexclusive(edgelist_pd,set_root,limit):
    classes_rapresentation_dict = dict()
    roots = deepcopy(set_root)
    total_len_rapresentation = 0
    flag = False
    while len(classes_rapresentation_dict) != limit:
        df = edgelist_pd[edgelist_pd["class"].isin(roots)]
        roots =set(df['subclass'])
        len_rapresentation = len(roots)
        for i,r in enumerate(roots):
            past_features = []
            if flag:
                to_add_list = [classes_rapresentation_dict[c] for c in set(df[df['subclass']==r]['class'])]
                for to_add in to_add_list:
                    if len(past_features)!=0:
                        past_features = [x + y for x, y in zip(past_features, to_add)]
                    else:
                        past_features = to_add
            classes_rapresentation_dict[r] = past_features + [int(i==j) for j in range(len_rapresentation)]
        flag = True
            
        for cl in (set(classes_rapresentation_dict.keys()) - roots):
            classes_rapresentation_dict[cl] += [0 for i in range(len_rapresentation)]
    return classes_rapresentation_dict

def fromOntologyEdglist_to_ClassRapresentation(edgelist_pd):
    subclasses = set(edgelist_pd["subclass"])
    classes = set(edgelist_pd["class"])
    set_root = classes - subclasses
    if len(set_root) != 1:
        for r in set_root:
            edgelist_pd = edgelist_pd.append([{
                "class":"_root_",
                "subclass":r
            }]
                , ignore_index=True)
        subclasses = set(edgelist_pd["subclass"])
        classes = set(edgelist_pd["class"])
        set_root = classes - subclasses
    return fromOntologyEdglist_to_ClassRapresentation_notexclusive(edgelist_pd,set_root,len(classes | subclasses)-1)



from pyfasttext import FastText
FASTTEXT_EN = FastText('data/fasttext_data/wiki.en.bin')
FASTTEXT_FR1= FastText('data/fasttext_data/wiki.fr.bin')
FASTTEXT_FR2= FastText('data/fasttext_data/my_corpus_model.bin')



def getFastTextFeatures(text,lang):
    tokens = splitInTokens(text)
    if lang == 'fr':
        fasttextfeatures = np.array([
            np.append(FASTTEXT_FR1[token[0]],FASTTEXT_FR2[token[0]])
            for i,token in enumerate(tokens)])
    if lang == 'en':
        fasttextfeatures = np.array([FASTTEXT_EN[token[0]] for i,token in enumerate(tokens)])
    return fasttextfeatures






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
    limit_failures = 3
    waiting_secs = 7
    print('Request phase')
    for ext in extractors_list:
        print(ext.name)
        counter_failures = 0
        while counter_failures >= 0 and counter_failures < limit_failures:
            try:
                if ext.name == 'adel':
                    ext.extract(text,lang=lang,setting=model_setting)
                else:
                    ext.extract(text,lang=lang)
                counter_failures = -1
            except:
                counter_failures += 1
                if counter_failures == limit_failures:
                    raise Exception("The extractor",ext.name,"presented an error during the API request phase\n"
                                    +str(sys.exc_info()[1]))
                else:
                    time.sleep(waiting_secs)

            
    #print(strftime("%H:%M:%S", gmtime()))
    extractors_responses = {ext.name:ext.get_annotations() for ext in extractors_list}
    print('Parsing phase')
    for ext in extractors_list:
        print(ext.name)
        try:
            ext.parse()
        except:
            print(ext.get_annotations())
            raise Exception("The extractor",ext.name,"presented an error during the API response parsing phase\n"+
                            str(sys.exc_info()[1]))
    #print(strftime("%H:%M:%S", gmtime()))
    print('Tokenization phase')
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
            type_list_dict[ext.name]=[a['type'] for a in annotations_ext]
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
def getFeatures(text,lang=None,model_setting='default'):
    if not bool(lang):
        lang = detect(text)
    features_dict_all = {'features':{}}
    extractors_responses,type_list_dict,score_list_dict,entity_list_dict,ontology_type_dict,ontology_entity_dict = getAnnotations(text,lang=lang,model_setting=model_setting)
    features_dict_all["entity_list"] = entity_list_dict
    features_dict_all["type_list"] = type_list_dict
    features_dict_all["extractors_responses"] = extractors_responses
    print('Forming features')
    
    if 'fasttext' not in features_dict_all['features']:
        features_dict_all['features']['fasttext']=getFastTextFeatures(text,lang)
    print('Formed Fasttext features')
        
    if 'type' not in features_dict_all['features']:
        features_dict_all['features']['type']=getTypeFeatures(type_list_dict,entity_list_dict,ontology_type_dict,ontology_entity_dict)
    print('Formed type features')
    if 'score' not in features_dict_all['features']:
        features_dict_all['features']['score']=score_list_dict
        
    print('Formed score features')
    if 'entity' not in features_dict_all['features']:
        features_dict_all['features']['entity'],features_dict_all['features']['entity_MATRIX']=getEntityFeatures(entity_list_dict,ontology_entity_dict,lang)
    print('Formed entity features')
    return features_dict_all
