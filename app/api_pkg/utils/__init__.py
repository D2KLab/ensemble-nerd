import json
import h5py
import pickle
import string
import numpy as np
import os
import re
import urllib.request, urllib.error, urllib.parse
import pandas as pd
from copy import deepcopy
from fuzzywuzzy import fuzz
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer


PATH_CREDENTIALS = "api_pkg/credentials.json"
CREDENTIALS_OBJ = json.load(open(PATH_CREDENTIALS))

def getCredentials(extractor_name,CREDENTIALS_OBJ=CREDENTIALS_OBJ):
    try:
        return CREDENTIALS_OBJ[extractor_name]
    except:
        raise Exception('Wrong extractor name')

