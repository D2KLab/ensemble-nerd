from fuzzywuzzy import fuzz
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer

def fuzzSimilarity(a, b,w=0.75):
    return (w*fuzz.token_sort_ratio(a, b)+(1-w)*fuzz.partial_ratio(a,b))/100


def oneHotSimilarity(a,b):
    return bool(a==b)



def ffIdfSimilarity(a,b):
    if a == b:
        return 1.0
    tfidf = TfidfVectorizer().fit_transform([a,b])
    pairwise_similarity = tfidf * tfidf.T
    return pairwise_similarity.A[0][1]