import requests
import numpy as np
from api_pkg.utils.parsing import *
from api_pkg.utils.tokenization import *
from api_pkg.utils import *
import unicodedata


def removeAccents(s):
    s = ''.join((c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn'))
    s = s.replace('–', '-')
    return s


class ADEL(object):
    def __init__(self, endpoint="adel.eurecom.fr"):
        self.name = "adel"
        self.ontology = "adel"
        self.endpoint = endpoint
        self.lang = None
        self.annotations = None
        self.text = None
        self.disambiguation = False
        self.recognition = True

    def extract(self, text, extractors="entities,topics", lang="fr", min_confidence=0.0, setting='default'):
        self.lang = lang
        self.text = text
        headers = {
            'accept': 'text/plain;charset=utf-8',
            'content-type': 'application/json;charset=utf-8',
        }
        params = (
            ('setting', setting),
            ('lang', lang)
        )
        text = removeAccents(text)
        if setting in ['oke2016', 'oke2015']:
            self.ontology = 'adelOKE'
        if setting in ['neel2015']:
            self.ontology = 'adelNEEL'
        data = '{ "content": "' + text.replace('"', '\\"') + '", "input": "raw", "output": "brat"}'
        try:
            response = requests.post('http://' + self.endpoint + '/v1/extract', headers=headers, params=params,
                                     data=data)
            self.annotations = response.text
        except:
            self.annotations = ''

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
                        "chars": set([i for i in range(start_char, end_char)]),
                        "type": category,
                        "text": annotation_text,
                        "uri": np.NAN
                    }
        if len(annotations_dict) == 0:
            print(lines)
            self.annotations = []
        else:
            annotations = [annotations_dict[key] for key in annotations_dict]
            cleaned_annotations = removeDoubleOccurences(annotations)
            if not doubleCheck:
                raise Exception("Double check parse false")
            if not consistencyText(cleaned_annotations, text):
                cleaned_annotations = createConsistencyText(cleaned_annotations, text)
            self.annotations = cleaned_annotations

    def tokenize(self):
        text = self.text
        annotations = self.annotations
        annotations = addMissingText(annotations, text)
        annotations = fromAnnotationToTokens(annotations)
        self.annotations = annotations

    def set_annotations(self, annotations):
        self.annotations = annotations

    def get_annotations(self):
        return self.annotations

    def get_text(self):
        return self.text

    def get_info(self):
        return self.recognition, self.disambiguation

    def clear_annotations(self):
        self.annotations = None
