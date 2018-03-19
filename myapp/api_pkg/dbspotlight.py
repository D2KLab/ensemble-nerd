import requests
from api_pkg.utils.parsing import *
from api_pkg.utils.tokenization import *
from api_pkg.utils.request import *
from api_pkg.utils import *
import pandas as pd
import numpy as np

class DBSPOTLIGHT(object):
    def __init__(self, endpoint="model.dbpedia-spotlight.org"):
        self.name = "dbspotlight"
        self.ontology = "wikidata"
        self.endpoint = endpoint
        self.lang = None
        self.annotations = None
        self.text = None
        self.disambiguation = True
        self.recognition = False

    def extract(self, text,extractors="entities,topics",lang="fr",min_confidence=0.0):
        self.lang=lang
        self.text=text
        headers = {
            'Accept': 'application/json',
        }
        params = (
            ('text', text),
            ('confidence', '0.0'),
            ('support', '20'),
        )
        response = requests.get('http://'+self.endpoint+'/'+lang+'/annotate',headers=headers, params=params)
        #print('http://'+self.endpoint+'/'+lang+'/annotate')
        try:
            self.annotations = response.json()["Resources"]
        except:
            
            print(response.text)
            raise Exception



    def parse(self):
        text = self.text
        annotations = self.annotations
        db_urls = list()
        occurrences = list()
        for ann in annotations:
            occurrences.append({
                "text":ann['@surfaceForm'],
                'chars':set([i for i in range(int(ann["@offset"]),int(ann["@offset"])+len(ann['@surfaceForm']))]),
                "relevance":float(ann["@similarityScore"]),
                "uri":ann["@URI"],
                'type':np.NAN
            })

        if len(occurrences) != 0:
            if self.lang == 'fr':
                occurrences = setWikidataUrisfromDbpedia_fr(pd.DataFrame(occurrences)).to_dict(orient='records')

            elif self.lang == 'en':
                occurrences = setWikidataUrisfromDbpedia_en(pd.DataFrame(occurrences)).to_dict(orient='records')

            cleaned_annotations = removeDoubleOccurences(occurrences)
            if not doubleCheck:
                raise Exception("Double check parse false")
            if not consistencyText(cleaned_annotations,text):
                raise Exception("The token start end char and the text don't correspond")
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
