from flask import Flask, request, jsonify, abort, make_response, current_app
import time
from api_pkg import dandelion,dbspotlight,opencalais,babelfy,adel,meaning_cloud,alchemy,textrazor

def getErrors(text,lang=None,model_setting='default'):
    extractors_list = [
        alchemy.ALCHEMY(),
        adel.ADEL(),
        dbspotlight.DBSPOTLIGHT(),
        opencalais.OPENCALAIS(),
        meaning_cloud.MEANINGCLOUD(),
        dandelion.DANDELION(),
        babelfy.BABELFY(),
        textrazor.TEXTRAZOR()
    ]
    
    #print(strftime("%H:%M:%S", gmtime()))
    limit_failures = 3
    waiting_secs = 7

    extractors_errors = {}
    for ext in extractors_list:
        print(ext.name)
        counter_failures = 0
        while counter_failures >= 0 and counter_failures < limit_failures:
            try:
                if ext.name == 'adel':
                    ext.extract(text,lang=lang,setting=model_setting)
                else:
                    ext.extract(text,lang=lang)
                counter_failures = -1
            except:
                print(sys.exc_info()[1])
                counter_failures += 1
                if counter_failures == limit_failures:
                	extractors_errors[ext.name] = str(sys.exc_info()[1])
                else:
                    time.sleep(waiting_secs)

            
    #print(strftime("%H:%M:%S", gmtime()))
    extractors_responses = {ext.name:ext.get_annotations() for ext in extractors_list}
        
    return extractors_responses,extractors_errors




lang = 'en'


text = "In Italy the rector is the head of the university and Rappresentante Legale (Legal representative) of the university. He or she is elected by an electoral body."

extractors_responses,extractors_errors=getErrors(text,lang=lang)

print(extractors_errors)

input()

for ext in extractors_responses:
    print(ext)
    input()
    print(extractors_responses[ext])
    input()