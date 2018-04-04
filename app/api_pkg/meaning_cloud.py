import requests
from api_pkg.utils.parsing import *
from api_pkg.utils.tokenization import *
from api_pkg.utils import *
import numpy as np


class MEANINGCLOUD(object):
    def __init__(self, endpoint="api.meaningcloud.com"):
        self.name = "meaning_cloud"
        self.ontology = "meaning_cloud"
        self.api_key = getCredentials(self.name)
        self.endpoint = endpoint
        self.lang = None
        self.annotations = None
        self.text = None
        self.disambiguation = False
        self.recognition = True

    def extract(self, text,extractors="entities,topics",lang="fr",min_confidence=0.0):
        self.lang=lang
        self.text=text
        params = (
            ('key', self.api_key),
            ('of', 'json'),
            ('lang', lang),
            ('txt', text),
            ('tt', 'a')
        )
        response = requests.post('https://api.meaningcloud.com/topics-2.0', params=params)
        self.annotations = response.json()["entity_list"]

    def parse(self):
        text = self.text
        annotations = self.annotations
        occurrences = list()
        for ann in annotations:
            type_ = ann["sementity"]['type'].split('>')[-1]
            if type_ != 'Top':
                for inst in ann["variant_list"]:
                    occurrences.append({
                        "text":ann['form'],
                        'chars':set([i for i in range(int(inst['inip']),int(inst['endp'])+1)]),
                        "type":ann["sementity"]['type'].split('>')[-1],
                        "relevance":int(ann['relevance'])/100,
                        "uri":np.NAN
                    })
        if len(occurrences) != 0:
            cleaned_annotations = removeDoubleOccurences(occurrences)
            cleaned_annotations = addMissingChars(cleaned_annotations,text)
            if not doubleCheck:
                raise Exception("Double check parse false")
            if not consistencyText(cleaned_annotations,text):
                cleaned_annotations = createConsistencyText(cleaned_annotations,text)
            self.annotations = cleaned_annotations
        else:
            self.annotations = []


    def tokenize(self):
        text = self.text
        annotations = self.annotations
        annotations = addMissingText(annotations,text)
        annotations = fromAnnotationToTokens(annotations)
        self.annotations = annotations


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