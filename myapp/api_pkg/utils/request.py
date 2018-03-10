

import requests
from queue import Queue
from threading import Thread
import threading
import urllib.request, urllib.error, urllib.parse
import json
import re
import numpy as np

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

q_wd_db_sameAs = '''
select ?db_uri ?wd_uri
where{
values ?wd_uri { <STR_TO_SUB> }
?db_uri owl:sameAs ?wd_uri.
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

import pandas as pd


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

def fromWikidataToDbpediaUri(wd_uris,query=q_wd_db_sameAs):
    db_uris = list(db_uris)
    queries = getQueries(wd_uris,query,offset=20)
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
        content = res.read()
        if type(content) != str:
            content = content.decode('utf8')
        try:
            obj = json.loads(content)['query']['pages']
        except:
            print(content)

            raise Exception
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