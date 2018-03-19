from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 import Features,EntitiesOptions


from api_pkg.utils.parsing import *
from api_pkg.utils.tokenization import *
from api_pkg.utils import *
import re
import numpy as np

class ALCHEMY(object):
    def __init__(self):
        self.name = "alchemy"
        self.ontology = "alchemy"
        credentials_obj = getCredentials(self.name)
        self.credentials = NaturalLanguageUnderstandingV1(
            username=credentials_obj["username"],
            password=credentials_obj["password"],
            version="2017-02-27"
        )
        self.lang = None
        self.text = None
        self.annotations = None
        self.disambiguation = False
        self.recognition = True
        


    def extract(self, text,extractors="entities,topics",lang="fr",min_confidence=0.0):
        natural_language_understanding = self.credentials
        self.lang=lang
        self.text=text
        response = natural_language_understanding.analyze(
          text=text,language=lang,
          features= Features(entities=EntitiesOptions(limit=250))
        )
        annotations = response["entities"]
        self.annotations=annotations

    def parse(self):
        text = self.text
        annotations = self.annotations
        occurrences = list()
        for ann in annotations:
            matched = list(re.finditer(r"\b"+re.escape(ann['text'])+'\W', text))
            for a in matched:
                obj = {
                    'text':ann['text'],
                    'chars':set([i for i in range(a.start(),a.end()-1)]),
                    'type':ann['type'],
                    'relevance':ann['relevance'],
                    'uri':np.NAN}
                occurrences.append(obj)
        if len(occurrences) != 0:
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


    
