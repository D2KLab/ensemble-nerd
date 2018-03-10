import requests
from api_pkg.utils.parsing import *
from api_pkg.utils.tokenization import *
from api_pkg.utils.request import *
from api_pkg.utils import *
import pandas as pd
import numpy as np

class DANDELION(object):
    def __init__(self, endpoint="api.dandelion.eu"):
        self.name = "dandelion"
        self.ontology = "wikidata"
        self.token = getCredentials(self.name)
        self.endpoint = endpoint
        self.lang = None
        self.annotations = None
        self.text = None
        self.disambiguation = True
        self.recognition = False

    def extract(self, text,extractors="entities,topics",lang="fr",min_confidence=0.0):
        self.lang=lang
        self.text=text
        params = {"lang":lang,"text": text, "token":self.token,"min_confidence":str(min_confidence)}
        response = requests.post('https://'+self.endpoint+'/datatxt/nex/v1', params=params)
        try:
            self.annotations = response.json()["annotations"]
        except:
            if response.json()["code"] == "error.requestURITooLong":
                self.annotations = []
            else:
                print(respose.text)
                raise Exception


    def parse(self):
        text = self.text
        annotations = self.annotations
        wiki_urls = list()
        occurrences = list()
        for ann in annotations:
            occurrences.append({
            	'chars':set([i for i in range(ann["start"],ann["end"])]),
                "text":ann['spot'],
                "confidence":ann["confidence"],
                "uri":ann["uri"],
                'type':np.NAN
            })
            wiki_urls.append(ann["uri"])

        wiki_wd_dict = {}
        res = fromWikipediatoWikidata(wiki_urls)
        if len(res) > 0:
            wiki_wd_dict = res.set_index('wiki_uri')['wd_uri'].to_dict()
            wiki_keys = set(wiki_wd_dict.keys())
        else:
            wiki_keys = set()

        for uri in set(wiki_urls) - wiki_keys:
            df = pd.read_csv(getWikiMissingInfo(uri))
            if len(df) > 0:
                lines_df = df[df['predicate'] == 'http://schema.org/about'][['subject','object']].to_dict(orient='records')
                for l in lines_df:
                    s=l['subject']
                    o=l['object'].split('/')[-1]
                    wiki_wd_dict[s] = o


        annotations = list()
        for occ in occurrences:
            if occ["uri"] in wiki_wd_dict:
                occ["uri"] = wiki_wd_dict[occ["uri"]]
                annotations.append(occ)
        cleaned_annotations = removeDoubleOccurences(annotations)
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



