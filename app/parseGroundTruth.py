import os

import pandas as pd
from api_pkg.utils.tokenization import fromAnnotationToTokens, , addMissingText

ground_truth = sys.argv[1]
base_path = 'data/training_data/+'+ground_truth+'/'
train_path = base_path+'train/'
test_path = base_path+'test/'
csv_train_path = train_path + 'csv_ground_truth/'
csv_test_path = test_path + 'csv_ground_truth/'
text_train_path = train_path + 'text_files/'
text_test_path = test_path + 'text_files/'

raw_path = base_path+'raw/'

for item in [train_path,test_path,csv_train_path,csv_test_path,text_train_path,text_test_path]:
    try:
        os.makedirs(item)
    except:
        pass

print(raw_path)

if ground_truth == 'neel2015': 
    
    #'''
    def parseNEEL(df_gt,id_text_dict,csv_path,text_path):
        df_gt['id'] = df_gt['id'].astype(str)
        #df_gt['start'] = df_gt['start'].astype(int)
        df_gt['end'] = df_gt['end'].apply(lambda x:x-1)
        ids = set(df_gt['id'])
        
        for id_ in ids:
            #id_ = str(id_)
            print(id_)
            df_id = df_gt[df_gt['id']==id_]
            text = id_text_dict[id_]
            def getCorrespondingText(row,text=text):
                start,end = row['start'],row['end']+1
                return text[start:end]
            df_id['text'] = df_id.apply (lambda row: getCorrespondingText(row,text=text),axis=1)
            db_uris = set([uri for uri in set(df_id['db_uri']) if type(uri)==str])
            df_id = setWikidataUrisfromDbpedia_en(df_id,name_col='db_uri')
            df_id = df_id.sort_values(by=['start'])
            def setChars(annotations):
                for i,row in enumerate(annotations):
                    start,end = row["start"],row["end"]
                    row['chars'] = range(start,end)
                    annotations[i] = row
                return annotations
            annotations = df_id.to_dict(orient='records')
            df_id['chars'] = setChars(annotations)
            annotations = addMissingText(annotations,text)
            annotations = fromAnnotationToTokens(annotations)
            with open(text_path+id_+'.txt', 'w') as outfile:
                outfile.write(text)
            pd.DataFrame(annotations).to_csv(csv_path+id_+'.csv',index=False)
    
    #'''
    

        #[confidence,continue,relevance,text,type,uri,wd_uri,db_uri,db_type]

    train_path_raw = raw_path + 'NEEL2015-training-gold.tsv'
    test_path_raw = raw_path + 'NEEL2015-test-gold.tsv'
    train_path_raw_tweets = raw_path + 'NEEL2015-training-tweets.tsv'
    test_path_raw_tweets = training_path = raw_path + 'NEEL2015-test-tweets.tsv'

    df_train_raw = pd.read_csv(train_path_raw, sep='\t',header=None,names=['id','start','end','db_uri','type'])
    df_test_raw = pd.read_csv(test_path_raw, sep='\t',header=None,names=['id','start','end','db_uri','type'])
    
    def read_tsv_as_dict(path):
        with open(path) as txt_file:
            lines = txt_file.read().splitlines()
        return {l.split('\t')[0]:l.split('\t')[1] for l in lines}
    
    dict_train_raw_tweets = read_tsv_as_dict(train_path_raw_tweets)
    dict_test_raw_tweets = read_tsv_as_dict(test_path_raw_tweets)
    parseNEEL(df_train_raw,dict_train_raw_tweets,csv_train_path,text_train_path)
    parseNEEL(df_test_raw,dict_test_raw_tweets,csv_test_path,text_test_path)


if ground_truth == 'oke2015':
 
