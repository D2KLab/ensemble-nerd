import re


#parsing utils

def addMissingChars(anns,text):
    for ann in anns:
        end_char = ann['end']
        next_char = end_char + 1
        if text[end_char].isalpha() or text[end_char].isdigit():
            while len(text)>next_char and (text[next_char].isalpha() or text[next_char].isdigit()):
                ann['end']+=1
                ann['text'] = text[ann['start']:ann['end']+1]
                end_char = ann['end']
                next_char = end_char + 1
        start_char = ann['start']
        prev_char = start_char - 1
        if text[start_char].isalpha() or text[start_char].isdigit():
            while prev_char>=0 and (text[prev_char].isalpha() or text[prev_char].isdigit()):
                ann['start']-=1
                ann['text'] = text[ann['start']:ann['end']+1]
                start_char = ann['start']
                prev_char = start_char - 1
    return anns


def removeDoubleOccurences(occurrences):
    I = list()
    types_to_append = dict()
    for i,o1 in enumerate(occurrences):
        flag = True
        for j,o2 in enumerate(occurrences[i+1:]):
            j += 1
            chars_ind_1 = o1["chars"]
            chars_ind_2 = o2["chars"]
            intersection = chars_ind_1 & chars_ind_2
            if len(intersection) != 0:
                to_remove=None
                if len(chars_ind_1) < len(chars_ind_2):
                    to_remove=i
                elif len(chars_ind_1) > len(chars_ind_2):
                    to_remove=i+j
                else:
                    if "relevance" in o1 and o1["relevance"] != o2["relevance"]:
                        if o1["relevance"] > o2["relevance"]:
                            to_remove=i+j
                        elif o1["relevance"] < o2["relevance"]:
                            to_remove=i
                    elif "confidence" in o1 and o1["confidence"] != o2["confidence"]:
                        if o1["confidence"] > o2["confidence"]:
                            to_remove=i+j
                        elif o1["confidence"] < o2["confidence"]:
                            to_remove=i
                    elif o1["type"] !=  o2["type"]:
                        to_remove=i+j
                        if i in types_to_append:
                            types_to_append[i].append(o2["type"])
                        else:
                            types_to_append[i]=[o2["type"]]
                    else:
                        to_remove=i+j
                I.append(to_remove)
    cleaned_annotations = list()
    for i,o in enumerate(occurrences):
        if i not in I:
            o["start"] = min(o["chars"])
            o["end"] = max(o["chars"])
            del o["chars"]
            if i in types_to_append:
                o['type'] = ','.join([o['type']]+types_to_append[i])
            cleaned_annotations.append(o)
    return cleaned_annotations

def doubleCheck(cleaned_annotations):
    chars_indexes = set()
    for a in cleaned_annotations:
        s = set([i for i in range(a['start'],a['end']+1)])
        if len(s & chars_indexes) >0:
            print(a)
            return False
        else:
            chars_indexes = chars_indexes | s
    return True


def createConsistencyText(cleaned_annotations,text):
    for a in cleaned_annotations:
        if (text[a['start']:a['end']+1]).lower() != a["text"].lower():
            a['text'] = text[a['start']:a['end']+1].lower()
    return cleaned_annotations

def consistencyText(cleaned_annotations,text):
    for a in cleaned_annotations:
        if (text[a['start']:a['end']+1]).lower() != a["text"].lower():
            print('text',text[a['start']:a['end']+1])
            print('surface',a["text"])
            return False
    return True

def splitUpperCaseTypes(anns):
    for ann in anns:
        spl = re.findall('[A-Z][^A-Z]*', ann['type'] )
        if len(spl) > 1:
            ann['type'] = ','.join(spl)
    return anns

#tolenization utils
import string
import numpy as np
def isGoodChar(c):
    return c.isalpha() or c.isdigit()

splitting_chars = list()
for ch in string.printable:
    if not isGoodChar(ch):
        splitting_chars.append(ch)

def formatText(string):
    string = string.lower()
    for ch in splitting_chars:
        if ch != ' ':
            string = string.replace(ch,' '+ch+' ')
    return string.replace('  ',' ')

def cleanTuple(original_tuples):
    final_tuples = list()
    for t in original_tuples:
        string = t[0]
        start = t[1]
        end = t[2]
        regexPattern = '|'.join(map(re.escape, splitting_chars))
        final_strings=[r for r in re.split(regexPattern, string) if bool(r)]
        starts = []
        for s in final_strings: 
            possible_starts = [m.start() for m in re.finditer(s, string)]
            for p in possible_starts:
                if p not in starts:
                    starts.append(p)
                    final_tuples.append(
                        (
                            s,
                            start + p,
                            start + p + len(s) - 1
                        )
                    )
    return final_tuples

'''
def splitInTokens(inputpath):
    try:
        with open(inputpath) as f:
            ret = [[m.group(0), m.start(),m.start()+len(m.group(0))-1] for m in re.finditer(r'\S+', f.read())]
    except:
        ret = [[m.group(0), m.start(),m.start()+len(m.group(0))-1] for m in re.finditer(r'\S+', inputpath)]
    regexPattern1 = '|'.join(map(re.escape, set(splitting_chars)-{' '}))
    ret1 = [(m.group(0), m.start(),m.start()+len(m.group(0))-1) for m in re.finditer(regexPattern1, 'Ciao .o-kl. Come stai')]
    final_tuples = cleanTuple(ret) + ret1
    final_tuples.sort(key=lambda x:x[1])
    return final_tuples

def splitInTokens(string):
    s=formatText(string)
    final_tuples = [(m.group(0), m.start(),m.start()+len(m.group(0))-1) for m in re.finditer(r'\S+', s)]
    final_tuples.sort(key=lambda x:x[1])
    return final_tuples

'''


def splitInTokens(string):
    s=string.lower()
    final_tuples = ([(m.group(0), m.start(),m.start()+len(m.group(0))-1)
                    for m in re.finditer(re.compile('[^'+''.join(splitting_chars)+']+')
                                         , s)]
                    +
                    [(m.group(0), m.start(),m.start()+len(m.group(0))-1)
                    for m in re.finditer(re.compile('['+''.join(splitting_chars[:-6])+']')
                                         , s)]
    )
    final_tuples.sort(key=lambda x:x[1])
    return final_tuples


def addMissingText(annotations,text):
    starts_ends = [(-2,-1)]+[(record['start'],record['end']) for record in annotations]+[(len(text),len(text))]
    starts_ends.sort(key=lambda x:x[0])
    for i in range(len(starts_ends[:-1])):
        start = starts_ends[i][-1]+1
        end = starts_ends[i+1][0]
        if end > start:
            ann = {
                    'text':text[start:end],
                    'start':start,
                    'end':end-1,
                    'type':np.NAN,
                    'uri':np.NAN,
                    'continue':0
                }
            annotations.append(ann)
    annotations.sort(key=lambda x:x['start'])
    return annotations

def fromAnnotationToTokens(annotations):
    annotations_new = list()
    for record in annotations:
        uri = record["uri"]
        type_ = record["type"]
        try:
            relevance = record["relevance"]
        except:
            relevance = 0
        try:
            confidence = record["confidence"]
        except:
            confidence = 0
        text_to_split = record["text"]
        splitted_text = splitInTokens(text_to_split)
        for word_tuple in splitted_text:
            if word_tuple == splitted_text[-1] or 'continue' in record:
                continue_flag = 0
            else:
                continue_flag = 1
            annotations_new.append({
                'text':word_tuple[0],
                'type':type_,
                'uri':uri,
                'relevance':relevance,
                'confidence':confidence,
                'continue':continue_flag
            })
    return annotations_new




#requests utils
import requests
from queue import Queue
from threading import Thread
import threading
import os
import re
import urllib.request, urllib.error, urllib.parse
import pandas as pd
from copy import deepcopy
import json
ex_query='''
select *{
STR_TO_SUB
}
'''

q_db_wd_sameAs = '''
select ?db_uri ?wd_uri
where{
values ?db_uri { <STR_TO_SUB> }
?db_uri owl:sameAs ?wd_uri.
filter(contains(str(?wd_uri), "http://www.wikidata.org"))
}
'''


dbfr_db_sameAS = '''
select ?db_fr_uri ?db_en_uri
where{
values ?db_fr_uri { <STR_TO_SUB> }
?db_fr_uri owl:sameAs ?db_en_uri.
filter(contains(str(?db_en_uri), "http://dbpedia.org"))
}
'''

dbfr_wiki_isPrimaryTopic = '''
select ?db_fr_uri ?wiki_uri
where{
values ?db_fr_uri { <STR_TO_SUB> }
?db_fr_uri foaf:isPrimaryTopicOf ?wiki_uri.
}
'''

dbfr_wiki_isDisambiguation= '''
select ?db_fr_uri ?db_fr_uri_disambiguation
where{
values ?db_fr_uri { <STR_TO_SUB> }
?db_fr_uri dbpedia-owl:wikiPageRedirects ?db_fr_uri_disambiguation.
}
'''
dben_wiki_isDisambiguation= '''
select ?db_en_uri ?db_en_uri_disambiguation
where{
values ?db_en_uri { <STR_TO_SUB> }
?db_en_uri dbo:wikiPageRedirects ?db_en_uri_disambiguation.
}
'''

qd_query = '''
select distinct *
where{
    values ?s { <STR_TO_SUB> }
    ?s ?p ?o
    <FILTER>
}
'''

url_wikidpedia_to_wikidataid = "/w/api.php?action=query&prop=pageprops&format=json&titles=<TITLE>"


def wikiSearchProperty(p,limit=1000000,columns=["?s","?o"]):
    edgelist_pd = pd.DataFrame(data=[],columns=columns)
    offset = -1
    while True:
        offset += limit
        df = wikiQuery(p=p,LIMIT=limit,OFFSET=offset)
        if len(df) > 0:
            edgelist_pd = edgelist_pd.append(df,ignore_index=True)
        else:
            break
    return edgelist_pd


def wikiQuery(s,p,filter_q='',q=qd_query):
    if s[0] == 'Q':
        s = 'http://www.wikidata.org/entity/'+s
    q = q.replace('STR_TO_SUB',s).replace('?p',p).replace('<FILTER>',filter_q)
    URL = getRequestURL(q,endpointURL='https://query.wikidata.org/sparql')
    try:
        text = getRequestResult_and_Read(URL)
    except:
        print(URL)
        text = getRequestResult_and_Read(URL)
    df = fromXMLtoDataFrame(text)
    return df





    
def fromDbpediaToWikidataUri(db_uris,query=q_db_wd_sameAs):
    db_uris = list(db_uris)
    queries = getQueries(db_uris,query,offset=20)
    URLs = [getRequestURL(q,endpointURL='http://dbpedia.org/sparql') for q in queries]
    results = getQueriesResults(URLs)
    mapping_db_en_wd = pd.concat([pd.read_csv(r) for r in results]).reset_index(drop=True)
    return mapping_db_en_wd

def fromFrenchDbpediatoEnglishDbpedia(db_uris,query=dbfr_db_sameAS):
    db_uris = list(db_uris)
    queries = getQueries(db_uris,query,offset=20)
    URLs = [getRequestURL(q) for q in queries]
    results = getQueriesResults(URLs)
    mapping_db_fr_en = pd.concat([pd.read_csv(r) for r in results]).reset_index(drop=True)
    return mapping_db_fr_en

def fromFrenchDbpediatoWikipedia(db_uris,query=dbfr_wiki_isPrimaryTopic):
    db_uris = list(db_uris)
    queries = getQueries(db_uris,query,offset=20)
    URLs = [getRequestURL(q) for q in queries]
    results = getQueriesResults(URLs)
    mapping_db_fr_wiki = pd.concat([pd.read_csv(r) for r in results]).reset_index(drop=True)
    return mapping_db_fr_wiki

def workerRequestWiki(q1,q2,url_wikidpedia_to_wikidataid=url_wikidpedia_to_wikidataid):
    while not q1.empty():
        w = q1.get()
        base,title=w.split("/wiki/")
        URL = base + url_wikidpedia_to_wikidataid.replace('<TITLE>',urllib.parse.quote(title))
        res = getRequestResult(URL)
        wd_uri = ''
        obj = json.loads(res.read())['query']['pages']
        for k in obj:
            try:
                int(k)
                wd_uri=obj[k]['pageprops']['wikibase_item']
            except:
                pass
        if wd_uri != '':
            record = {
                'wiki_uri':w,
                'wd_uri':wd_uri
            }
            q2.put(record)
        q1.task_done()
        
def getQueriesResultsWiki(items,num_threads=4):
    q1 = Queue(maxsize=0)
    q2 = Queue()
    for item in items:
        q1.put(item)
    for i in range(num_threads):
        worker = Thread(target=workerRequestWiki, args=(q1,q2))
        worker.setDaemon(True)
        worker.start()
    q1.join()
    return list(q2.queue)

def fromWikipediatoWikidata(wp_uris,url_wikidpedia_to_wikidataid=url_wikidpedia_to_wikidataid):
    records = getQueriesResultsWiki(wp_uris)
    return pd.DataFrame(records)

def getDisambiguationListTuple(db_uris,endpointURL='http://fr.dbpedia.org/sparql',query=dbfr_wiki_isDisambiguation):
    db_uris = list(db_uris)
    queries = getQueries(db_uris,query,offset=20)
    URLs = [getRequestURL(q,endpointURL) for q in queries]
    results = getQueriesResults(URLs)
    mapping_disambiguation = pd.concat([pd.read_csv(r) for r in results]).reset_index(drop=True)
    return mapping_disambiguation

def getWikiMissingInfo(wikipage,endpointURL='http://dbpedia.org/sparql'):
    title = wikipage.split('/')[-1]
    wikistring = wikipage.replace("http:/","https:/").replace(title,urllib.parse.quote(title))
    base_query = 'define sql:describe-mode "CBD"  DESCRIBE <STR_TO_SUB>'
    query = base_query.replace("STR_TO_SUB",wikistring)
    escapedQuery = urllib.parse.quote(query)
    requestURL = endpointURL + "?query=" + escapedQuery +"&output=text/csv"
    try:
        request = urllib.request.Request(requestURL)
        result = urllib.request.urlopen(request)
        return result
    except:
        raise Exception

def setWikidataUrisfromDbpedia_fr(annotations_pd):
    db_fr_uris = set(annotations_pd['uri'])
    mapping_disambiguation = getDisambiguationListTuple(db_fr_uris).to_dict(orient='records')
    for m in mapping_disambiguation:
        annotations_pd.loc[annotations_pd['uri'] ==  m['db_fr_uri'],'uri']= m['db_fr_uri_disambiguation']
    db_fr_uris = set(annotations_pd['uri'])
    mapping_db_fr_wiki = fromFrenchDbpediatoWikipedia(db_fr_uris)
    matched_wiki = set(mapping_db_fr_wiki["db_fr_uri"])
    wiki_uris = set(mapping_db_fr_wiki["wiki_uri"])
    mapping_wiki_wd = fromWikipediatoWikidata(wiki_uris)
    matched_wd= set(mapping_wiki_wd["wiki_uri"])
    for uri in wiki_uris - matched_wd:
        df = pd.read_csv(getWikiMissingInfo(uri))
        if len(df) > 0:
            lines_df = df[df['predicate'] == 'http://schema.org/about'][['subject','object']]
            lines_df.columns = list(mapping_wiki_wd.columns)
            mapping_wiki_wd = mapping_wiki_wd.append(lines_df,ignore_index=True)

    mapping_db_fr_wd = (
        pd.merge(
        mapping_db_fr_wiki,
        mapping_wiki_wd,
        on='wiki_uri',
        how='inner')
        [['db_fr_uri','wd_uri']]
        )

    def getWikidataAssociatedUri(uri):
        try:
            return list(mapping_db_fr_wd[mapping_db_fr_wd['db_fr_uri']==uri]['wd_uri'])[0].split('/')[-1]
        except:
            return np.NAN
    annotations_pd['uri'] = annotations_pd['uri'].apply(
        lambda uri:getWikidataAssociatedUri(uri)
    )

    return annotations_pd[~annotations_pd['uri'].isnull()]
    

def setWikidataUrisfromDbpedia_en(annotations_pd):
    db_en_uris = set(annotations_pd['uri'])
    mapping_disambiguation = getDisambiguationListTuple(db_en_uris,
                                                        endpointURL='http://dbpedia.org/sparql',
                                                        query=dben_wiki_isDisambiguation
                                                       ).to_dict(orient='records')
    for m in mapping_disambiguation:
        annotations_pd.loc[annotations_pd['uri'] ==  m['db_en_uri'],'uri']= m['db_en_uri_disambiguation']
    db_en_uris = set(annotations_pd['uri'])
    mapping_db_en_wd = fromDbpediaToWikidataUri(db_en_uris)
    def getWikidataAssociatedUri(uri):
        try:
            return list(mapping_db_en_wd[mapping_db_en_wd['db_uri']==uri]['wd_uri'])[0].split('/')[-1]
        except:
            return np.NAN
    annotations_pd['uri'] = annotations_pd['uri'].apply(
        lambda uri:getWikidataAssociatedUri(uri)
    )
    return annotations_pd[~annotations_pd['uri'].isnull()]



def setWikidataUris_Babelfy(annotations_pd):
    db_en_uris = set(annotations_pd['uri'])
    mapping_disambiguation = getDisambiguationListTuple(db_en_uris,
                                                        endpointURL='http://dbpedia.org/sparql',
                                                        query=dben_wiki_isDisambiguation
                                                       ).to_dict(orient='records')
    for m in mapping_disambiguation:
        annotations_pd.loc[annotations_pd['uri'] ==  m['db_fr_uri'],'uri']= m['db_fr_uri_disambiguation']
    db_en_uris = set(annotations_pd['uri'])
    mapping_db_en_wd = fromDbpediaToWikidataUri(db_en_uris)
    matched_wd= set(mapping_wiki_wd["db_uri"])
    recall_wd = (len(matched_wd) / len(db_en_uris))
    print("Recall_wd:",recall_wd*100,'%')
    def getWikidataAssociatedUri(r):
        try:
            wd_uri = list(mapping_db_en_wd[mapping_db_en_wd['db_uri']==r]['wd_uri'])[0]
            identifier = wd_uri.split('/')[-1]
            return identifier
        except:
            return ''
    annotations_pd['wd_uri'] = annotations_pd['uri'].apply(
        lambda r:getWikidataAssociatedUri(r)
    )
    return annotations_pd
    
    
    
    
    
def getRequestURL(query,endpointURL='http://fr.dbpedia.org/sparql',q=False):
    escapedQuery = urllib.parse.quote(query)
    requestURL = endpointURL + "?query=" + escapedQuery +"&output=text/csv&timeout=10000000"
    if q:
        return requestURL,query
    else:
        return requestURL

def getRequestResult(requestURL):
    request = urllib.request.Request(requestURL)
    result = urllib.request.urlopen(request)
    return result

def getRequestResult_and_Read(requestURL):
    result = getRequestResult(requestURL)
    text = result.read().decode("utf-8")
    return text


        
def workerRequest(q1,q2):
    while not q1.empty():
        URL = q1.get()
        res = getRequestResult(URL)
        q2.put(res)
        q1.task_done()

        
def fromXMLtoDataFrame(sstr):
    obj_list = list()
    rex_column_names= re.compile(r"<variable name='(.*?)'")
    column_names = re.findall(rex_column_names, sstr)
    rex = re.compile(r'<result.*?>(.*?)</result>',re.S|re.M)
    results = re.findall(rex, sstr)
    flag = False
    for j,res in enumerate(results):
        obj = {}
        if flag:
            print(j)
            flag = False
        for c in column_names:
            rex = re.compile(r"<binding name='"+c+"'>\n\t\t\t\t<.*?>(.*?)</.*?>\n\t\t\t</binding>",re.S|re.M)
            obj[c]=re.findall(rex, res)[0]
        try:
            obj_list.append(obj)
        except:
            print(results)
            print("No item")
            print(rex_3)
            raise Exception
            flag = True
    if len(obj_list)>0:
        return pd.DataFrame(obj_list)
    else:
        return pd.DataFrame(data=[],columns=column_names)     


def getQueriesResults(URLs,num_threads=4):
    q1 = Queue(maxsize=0)
    q2 = Queue()
    for url in URLs:
        q1.put(url)
    for i in range(num_threads):
        worker = Thread(target=workerRequest, args=(q1,q2))
        worker.setDaemon(True)
        worker.start()
    q1.join()
    return list(q2.queue)
    
def getQueries(strs,query,offset=10,replaceHTTP=False,flag_string=False,flag_print=False):
    queries = []
    linked_strings = []
    for i in range(0,len(strs),offset):
        if flag_print and i % 10000 == 0:
            print(i)
        if replaceHTTP:
            qr = ex_query.replace("STR_TO_SUB",
                            "\nUNION\n".join(['{'+query.replace("STR_TO_SUB",s.replace("http:/","https:/"))+'}' 
                                              for s in strs[i:i+offset]
                                             if type(s)!=float]))
        else:
            qr = ex_query.replace("STR_TO_SUB",
                            "\nUNION\n".join(['{'+query.replace("STR_TO_SUB",s)+'}' for s in strs[i:i+offset]
                                             if type(s)!=float]))
        queries.append(qr)
        linked_strings.append([s for s in strs[i:i+offset]])
    if flag_string:
        return queries,linked_strings
    return queries
