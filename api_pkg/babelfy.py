import urllib.request, urllib.parse, urllib.error
import json
from api_pkg.utils import *
import pandas as pd
import numpy as np

class BabelfyJSONKeys(object):
    
    
    TOKEN_FRAGMENT = "tokenFragment"
    CHAR_FRAGMENT = "charFragment"
    CHAR_FRAGMENT_START = "start"
    CHAR_FRAGMENT_END = "end"
    BABEL_SYNSET_ID = "babelSynsetID"
    DBPEDIA_URL = "DBpediaURL"
    BABELNET_URL = "BabelNetURL"
    SCORE = "score"
    COHERENCE_SCORE = "coherenceScore" 
    GLOBAL_SCORE ="globalScore"
    SOURCE =  "source"

class AnnTypeValues(object):
    
    ALL = "ALL" #Disambiguates all
    CONCEPTS = "CONCEPTS" #Disambiguates concepts only
    NAMED_ENTITIES = "NAMED_ENTITIES" #Disambiguates named entities only
    
class AnnResValues(object):
    BN = "BN" #Annotate with BabelNet synsets
    WIKI = "WIKI" #Annotate with Wikipedia page titles
    WN = "WN" #Annotate with WordNet synsets

class MatchValues(object):
    EXACT_MATCHING = "EXACT_MATCHING" #Only exact matches are considered for disambiguation
    PARTIAL_MATCHING = "PARTIAL_MATCHING" #Both exact and partial matches (e.g.
    
class MCSValues(object):
    OFF = "OFF" #Do not use Most Common Sense
    ON = "ON" #Use Most Common Sense
    ON_WITH_STOPWORDS = "ON_WITH_STOPWORDS" #Use Most Common Sense even on Stopwords

class CandsValues(object):
    ALL = "ALL" #Return all candidates for a fragment.
    TOP = "TOP" #Return only the top ranked candidate for a fragment.

class PosTagValues(object):
    #Tokenize the input string by splitting all characters as single tokens 
    #(all tagged as nouns, so that we can disambiguate nouns).
    CHAR_BASED_TOKENIZATION_ALL_NOUN = "CHAR_BASED_TOKENIZATION_ALL_NOUN" 
    INPUT_FRAGMENTS_AS_NOUNS = "INPUT_FRAGMENTS_AS_NOUNS" #Interprets input fragment words as nouns.
    NOMINALIZE_ADJECTIVES = "NOMINALIZE_ADJECTIVES"  #Interprets all adjectives as nouns.
    STANDARD = "STANDARD" #Standard PoS tagging process.

class SemanticAnnotation(object):
           
    def __init__(self,babelfy_dict):
        self.babelfy_dict = babelfy_dict
    
    def babelfy_dict(self):
        return self.babelfy_dict    
    
    def token_fragment(self):
        return self.babelfy_dict[BabelfyJSONKeys.TOKEN_FRAGMENT]
    
    def char_fragment(self):
        return self.babelfy_dict[BabelfyJSONKeys.CHAR_FRAGMENT]
    
    def char_fragment_start(self):
        return self.char_fragment()[BabelfyJSONKeys.CHAR_FRAGMENT_START]
    
    def char_fragment_end(self):
        return self.char_fragment()[BabelfyJSONKeys.CHAR_FRAGMENT_END]
    
    def babel_synset_id(self):
        return self.babelfy_dict[BabelfyJSONKeys.BABEL_SYNSET_ID]
    
    def dbpedia_url(self):
        return self.babelfy_dict[BabelfyJSONKeys.DBPEDIA_URL]
    
    def babelnet_url(self):
        return self.babelfy_dict[BabelfyJSONKeys.BABELNET_URL]
    
    def coherence_score(self):
        return self.babelfy_dict[BabelfyJSONKeys.COHERENCE_SCORE]
    
    def global_score(self):
        return self.babelfy_dict[BabelfyJSONKeys.GLOBAL_SCORE]
    
    def source(self):
        return self.babelfy_dict[BabelfyJSONKeys.SOURCE]
    
    def postag(self):
        return self.babel_synset_id()[-1]

    
    def pprint(self):
        print(self.babel_synset_id())
        print(self.babelnet_url())
        print(self.dbpedia_url())
        print(self.source())

class BABELFY(object):
    TEXT = "text"
    LANG = "lang"
    KEY = "key"
    ANNTYPE= "annType"
    ANNRES = "annRes"
    TH = "th"
    MATCH = "match"
    MCS = "MCS"
    DENS = "dens"
    CANDS = "cands"
    POSTAG = "postag"
    EXTAIDA = "extAIDA"
        
    PARAMETERS = [TEXT,LANG,KEY,ANNTYPE,ANNRES,TH,MATCH,MCS,DENS,CANDS,POSTAG,EXTAIDA]
    
    API = "https://babelfy.io/v1/"
    DISAMBIGUATE = "disambiguate?"
    
    def __init__(self):
        self.name = "babelfy"
        self.ontology = "wikidata"
        self.key = getCredentials(self.name)
        self.lang = None
        self.annotations = None
        self.text = None
        
    def extract(self, text,lang="fr",min_confidence=0.0, anntype=None, annres=None, th=None,
                     match=None,mcs=None,dens=None,cands=None,postag=None,
                     extaida=None):
        
        self.lang=lang
        self.text=text
        key = self.key

        values = [text,lang.upper(),key,anntype,annres,th,match,mcs,dens,
                            cands,postag,extaida]
        
        query = urllib.parse.urlencode({param:value for param,value in zip(self.PARAMETERS, values)
                         if value is not None})

        json_string = urllib.request.urlopen(self.API+self.DISAMBIGUATE+query).read().decode("utf-8")
        #print json_string
        babelfy_jsons = json.loads(json_string)
        semantic_annotations = [SemanticAnnotation(babelfy_json) for babelfy_json in babelfy_jsons]
        entities = [s.babelfy_dict for s in semantic_annotations]
        self.annotations = entities


    def parse(self):
        text = self.text
        annotations = self.annotations
        occurrences = list()
        for ann in annotations:
            start = ann['charFragment']['start']
            end = ann['charFragment']['end']
            if ann['DBpediaURL'] != '':
                uri = ann["DBpediaURL"]
                if '`Anizzah' not in uri and '"Fatherland"' not in uri and '"TASS"' not in uri:
                    occurrences.append({
                        "text":text[start:end+1],
                        "chars":set([i for i in range(start,end+1)]),
                        "relevance":ann['score'],
                        "uri":uri,
                        'type':np.NAN
                    })
        occurrences = setWikidataUrisfromDbpedia_en(pd.DataFrame(occurrences)).to_dict(orient='records')

        cleaned_annotations = removeDoubleOccurences(occurrences)
        cleaned_annotations = addMissingChars(cleaned_annotations,text)
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
        types_ordered_list = [a['uri'] for a in annotations]
        type_features = getTypeRepresentation(types_ordered_list,self.ontology)
        return type_features

    def get_score_features(self):
        annotations = self.annotations
        return np.array([[a['relevance']] for a in annotations])


    def get_uris_list(self):
        annotations = self.annotations
        uris_ordered_list = [a['uri'] for a in annotations]
        return uris_ordered_list

    def get_type_list(self):
        return []

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
