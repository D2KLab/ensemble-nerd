import textrazor
from api_pkg.utils import *
import pandas as pd
import numpy as np

class TEXTRAZOR(object):
    def __init__(self, endpoint="api.textrazor.com"):
        self.name = "textrazor"
        self.ontology_uri = "wikidata"
        self.ontology_type = "dbpedia"
        self.api_key = getCredentials(self.name)
        self.lang = None
        self.text = None
        self.annotations = None

    def extract(self, text,extractors="entities,topics",lang="fr",min_confidence=0.0):
        self.lang = lang
        self.text = text
        lang = lang.replace("fr","fre").replace("en","eng")
        textrazor.api_key = self.api_key
        client = textrazor.TextRazor(extractors=["entities"])
        client.set_language_override(lang)
        response = client.analyze(text)
        entities = [entity.json for entity in response.entities()]
        self.annotations=entities

    def parse(self):
        text = self.text
        annotations = self.annotations
        occurrences = list()
        for ann in annotations:
            if 'wikidataId' in ann:
                uri = ann['wikidataId']
            else:
                uri = np.NAN
            if 'type' in ann:
                type_ = ann['type'][-1]
            else:
                type_ = np.NAN
            if type(type_) == str or type(uri) == str:
                obj = {
                    'text':ann['matchedText'],
                    'chars':set([i for i in range(int(ann['startingPos']),int(ann['endingPos']))]),
                    'type':type_,
                    'confidence':ann["confidenceScore"],
                    'relevance':ann["relevanceScore"],
                    'uri':uri}
                occurrences.append(obj)
        cleaned_annotations = removeDoubleOccurences(occurrences)
        cleaned_annotations = addMissingChars(cleaned_annotations,text)
        if not doubleCheck(cleaned_annotations):
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
        types_ordered_list1 = [a['uri'] for a in annotations]
        type_features1 = getTypeRepresentation(types_ordered_list1,self.ontology_uri)
        types_ordered_list2 = [a['type'] for a in annotations]
        type_features2 = getTypeRepresentation(types_ordered_list2,self.ontology_type)
        type_features = np.append(type_features1, type_features2, axis=1)
        return type_features

    def get_score_features(self):
        annotations = self.annotations
        return np.array([[a['relevance']] for a in annotations])


    def get_uris_list(self):
        annotations = self.annotations
        uris_ordered_list = [a['uri'] for a in annotations]
        return uris_ordered_list

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
