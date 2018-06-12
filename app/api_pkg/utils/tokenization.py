import string
import numpy as np
import re


def isGoodChar(c):
    return c.isalpha() or c.isdigit()


splitting_chars = ['\xa0']
for ch in string.printable:
    if not isGoodChar(ch):
        splitting_chars.append(ch)


def formatText(string):
    string = string.lower()
    for ch in splitting_chars:
        if ch != ' ':
            string = string.replace(ch, ' ' + ch + ' ')
    return string.replace('  ', ' ')


def cleanTuple(original_tuples):
    final_tuples = list()
    for t in original_tuples:
        string = t[0]
        start = t[1]
        end = t[2]
        regexPattern = '|'.join(map(re.escape, splitting_chars))
        final_strings = [r for r in re.split(regexPattern, string) if bool(r)]
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


def splitInTokens(string):
    s = string.lower()
    final_tuples = ([(m.group(0), m.start(), m.start() + len(m.group(0)) - 1)
                     for m in re.finditer(re.compile('[^' + ''.join(splitting_chars) + ']+')
                                          , s)]
                    +
                    [(m.group(0), m.start(), m.start() + len(m.group(0)) - 1)
                     for m in re.finditer(re.compile('[' + ''.join(splitting_chars[:-6]) + ']')
                                          , s)]
                    )
    final_tuples.sort(key=lambda x: x[1])
    return final_tuples


def addMissingText(annotations, text):
    starts_ends = [(-2, -1)] + [(record['start'], record['end']) for record in annotations] + [(len(text), len(text))]
    starts_ends.sort(key=lambda x: x[0])
    for i in range(len(starts_ends[:-1])):
        start = starts_ends[i][-1] + 1
        end = starts_ends[i + 1][0]
        if end > start:
            ann = {
                'text': text[start:end],
                'start': start,
                'end': end - 1,
                'type': np.NAN,
                'uri': np.NAN,
                'continue': 0
            }
            annotations.append(ann)
    annotations.sort(key=lambda x: x['start'])
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
                'text': word_tuple[0],
                'type': type_,
                'uri': uri,
                'relevance': relevance,
                'confidence': confidence,
                'continue': continue_flag
            })
    return annotations_new
