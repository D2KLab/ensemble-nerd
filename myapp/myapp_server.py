from api_pkg.utils.output import *
from api_pkg.utils.representation import *
from flask import Flask, request, jsonify, abort, make_response, current_app



app = Flask(__name__)

@app.route('/')
#@crossdomain(origin='*')
def home_page():
    return 'hello world'


@app.route('/entities', methods = ['POST'])
#@crossdomain(origin='*')
def getEntities():
    # /position?lat=43.697093&lon=7.270747
    # show the coordinates

    lang = request.args.get('lang', default=None, type=str)

    model_recognition = request.args.get('model_recognition', default='oke2016', type=str)
    model_disambiguation = request.args.get('model_disambiguation', default='oke2016', type=str)

    if request.headers['Content-Type'] == 'text/plain':
        text = request.data.decode('utf-8')


    if request.headers['Content-Type'] == 'application/json':
        request_obj = request.json
        if 'text' in request_obj:
            text = request_obj['text']
        else:
            return 'No text passed'

    print(lang,model_disambiguation,model_recognition)


    features_obj = getFeatures(text,lang=lang,model_setting=model_recognition)
    response_obj = getAnnotationsOutput(features_obj,text,model_name_recognition=model_recognition,model_name_disambiguation=model_disambiguation,return_flag=True,normalize_flag=True)
    response_obj["extractors_responses"] = features_obj["extractors_responses"]


    return jsonify(response_obj)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8089)
