import requests
import numpy as np
from api_pkg.utils import *
import unicodedata

def removeAccents(s):
    s = ''.join((c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn'))
    s = s.replace('–','-')
    return s

class ADEL(object):
    def __init__(self, endpoint="adel.eurecom.fr"):
        self.name = "adel"
        self.ontology = "adel"
        self.endpoint = endpoint
        self.lang = None
        self.annotations = None
        self.text = None


    def extract(self, text,extractors="entities,topics",lang="fr",min_confidence=0.0,setting='default'):
        self.lang=lang
        self.text=text
        headers = {
            'Content-Type': 'application/json;charset=utf-8',
            'Accept': 'application/json',
        }
        params = (
            ('setting', setting),
            ('lang', lang),
        )
        text = removeAccents(text)
        if setting == 'oke2016':
            self.ontology = 'adelOKE2016'

        data = '{ "content": "'+text.replace('"','\\"')+'", "input": "raw", "output": "brat"}'
        response = requests.post('http://'+self.endpoint+'/v1/extract',headers=headers, params=params,data=data)

        self.annotations = response.text

    def parse(self):
        text = self.text
        string = self.annotations
        annotations_dict = dict()
        lines = string.splitlines()
        for l in lines:
            split_1 = l.split('\t')
            if len(l) > 0:
                if l[0] == 'T':
                    annotation_key = split_1[0]
                    annotation_text = split_1[-1]
                    try:
                        split_2 = split_1[1].split(' ')
                    except:
                        print(l.split_1)
                        raise Exception
                    start_char = int(split_2[1])
                    end_char = int(split_2[-1])
                    category = split_2[0].split('#')[-1]
                    annotations_dict[annotation_key] = {
                        "chars":set([i for i in range(start_char,end_char)]),
                        "type":category,
                        "text":annotation_text,
                        "uri":np.NAN
                    }
        if len(annotations_dict) == 0:
            print(lines)
        annotations = [annotations_dict[key] for key in annotations_dict]
        cleaned_annotations = removeDoubleOccurences(annotations)
        if not doubleCheck:
            raise Exception("Double check parse false")
        if not consistencyText(cleaned_annotations,text):
            cleaned_annotations = createConsistencyText(cleaned_annotations,text)
        self.annotations = cleaned_annotations

    def tokenize(self):
        text = self.text
        annotations = self.annotations
        annotations = addMissingText(annotations,text)
        annotations = fromAnnotationToTokens(annotations)
        self.annotations = annotations

    def get_type_features(self):
        annotations = self.annotations
        types_ordered_list = [a['type'] for a in annotations]
        type_features = getTypeRepresentation(types_ordered_list,self.ontology)
        return type_features

    def get_score_features(self):
        annotations = self.annotations
        return np.array([[a['relevance']] for a in annotations])


    def get_uris_list(self):
        return []

    def get_type_list(self):
        annotations = self.annotations
        types_ordered_list = [a['type'] for a in annotations]
        return types_ordered_list

    def set_annotations(self,annotations):
        self.annotations=annotations

    def get_annotations(self):
        return self.annotations

    def get_text(self):
        return self.text

    def get_info(self):
        return self.recognition,self.disambiguation

    def clear_annotations(self):
        self.annotations = None

