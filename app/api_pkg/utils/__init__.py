import json
import os

PATH_CREDENTIALS = os.path.join(os.getcwd(), 'api_pkg', 'credentials.json')
CREDENTIALS_OBJ = json.load(open(PATH_CREDENTIALS))


def get_credentials(extractor_name):
    global CREDENTIALS_OBJ
    try:
        return CREDENTIALS_OBJ[extractor_name]
    except:
        raise Exception('Wrong extractor name')
