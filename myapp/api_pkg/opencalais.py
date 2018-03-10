import requests
from api_pkg.utils.parsing import *
from api_pkg.utils.tokenization import *
from api_pkg.utils import *
import numpy as np

class OPENCALAIS(object):
    def __init__(self, endpoint="api.thomsonreuters.com"):
        self.name = "opencalais"
        self.ontology = "opencalais"
        self.access_token = getCredentials(self.name)
        self.endpoint = endpoint
        self.lang = None
        self.annotations = None
        self.text = None
        self.headers = {'X-AG-Access-Token' : self.access_token, 'Content-Type' : 'text/raw', 'outputformat' : 'application/json'}
        self.disambiguation = False
        self.recognition = True

    def extract(self, text,lang="fr",min_confidence=0.0):
        self.lang=lang
        self.text=text
        files = {'file': text}
        response = requests.post('https://'+self.endpoint+'/permid/calais',files=files,headers=self.headers, timeout = 80)
        obj =  response.json()
        entities = [obj[key] for key in obj if key != 'doc']
        self.annotations = entities

    def parse(self):
        text = self.text
        annotations = [ann for ann in self.annotations if 'instances' in ann]
        opencalais_annotations = list()
        for ann in annotations :
            occurrences = ann['instances']
            for o in occurrences:
                start = o['offset']-100
                end = o['offset']-100+o['length']
                if "relevance" in ann:
                    relevance = ann["relevance"]
                else:
                    relevance = 0
                opencalais_annotations.append({
                    "text":o['exact'],
                    "type":ann["_type"],
                    'chars':set([i for i in range(start,end)]),
                    "relevance":relevance,
                    "uri":np.NAN
                })
        cleaned_annotations = removeDoubleOccurences(opencalais_annotations)
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