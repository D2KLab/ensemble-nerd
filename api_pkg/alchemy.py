from watson_developer_cloud import AlchemyLanguageV1
from watson_developer_cloud import NaturalLanguageUnderstandingV1
import watson_developer_cloud.natural_language_understanding.features.v1 \
  as Features
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
        


    def extract(self, text,extractors="entities,topics",lang="fr",min_confidence=0.0):
        natural_language_understanding = self.credentials
        self.lang=lang
        self.text=text
        response = natural_language_understanding.analyze(
          text=text,language=lang,
          features=[
            Features.Entities(
              limit=250
            )
          ]
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
        cleaned_annotations = removeDoubleOccurences(occurrences)
        if not doubleCheck:
            raise Exception("Double check parse false")
        if not consistencyText(cleaned_annotations,text):
            raise Exception("The token start end char and the text don't correspond")
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


    
