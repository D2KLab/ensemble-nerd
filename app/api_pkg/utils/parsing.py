def addMissingChars(anns, text):
    for ann in anns:
        end_char = ann['end']
        next_char = end_char + 1
        if text[end_char].isalpha() or text[end_char].isdigit():
            while len(text) > next_char and (text[next_char].isalpha() or text[next_char].isdigit()):
                ann['end'] += 1
                ann['text'] = text[ann['start']:ann['end'] + 1]
                end_char = ann['end']
                next_char = end_char + 1
        start_char = ann['start']
        prev_char = start_char - 1
        if text[start_char].isalpha() or text[start_char].isdigit():
            while prev_char >= 0 and (text[prev_char].isalpha() or text[prev_char].isdigit()):
                ann['start'] -= 1
                ann['text'] = text[ann['start']:ann['end'] + 1]
                start_char = ann['start']
                prev_char = start_char - 1
    return anns


def removeDoubleOccurences(occurrences):
    I = list()
    types_to_append = dict()
    for i, o1 in enumerate(occurrences):
        flag = True
        for j, o2 in enumerate(occurrences[i + 1:]):
            j += 1
            chars_ind_1 = o1["chars"]
            chars_ind_2 = o2["chars"]
            intersection = chars_ind_1 & chars_ind_2
            if len(intersection) != 0:
                to_remove = None
                if len(chars_ind_1) < len(chars_ind_2):
                    to_remove = i
                elif len(chars_ind_1) > len(chars_ind_2):
                    to_remove = i + j
                else:
                    if "relevance" in o1 and o1["relevance"] != o2["relevance"]:
                        if o1["relevance"] > o2["relevance"]:
                            to_remove = i + j
                        elif o1["relevance"] < o2["relevance"]:
                            to_remove = i
                    elif "confidence" in o1 and o1["confidence"] != o2["confidence"]:
                        if o1["confidence"] > o2["confidence"]:
                            to_remove = i + j
                        elif o1["confidence"] < o2["confidence"]:
                            to_remove = i
                    elif o1["type"] != o2["type"]:
                        to_remove = i + j
                        if i in types_to_append:
                            types_to_append[i].append(o2["type"])
                        else:
                            types_to_append[i] = [o2["type"]]
                    else:
                        to_remove = i + j
                I.append(to_remove)
    cleaned_annotations = list()
    for i, o in enumerate(occurrences):
        if i not in I:
            o["start"] = min(o["chars"])
            o["end"] = max(o["chars"])
            del o["chars"]
            if i in types_to_append:
                to_app = [a for a in [o['type']] + types_to_append[i] if type(a) != float]
                o['type'] = ','.join(to_app)
            cleaned_annotations.append(o)
    return cleaned_annotations


def doubleCheck(cleaned_annotations):
    chars_indexes = set()
    for a in cleaned_annotations:
        s = set([i for i in range(a['start'], a['end'] + 1)])
        if len(s & chars_indexes) > 0:
            print(a)
            return False
        else:
            chars_indexes = chars_indexes | s
    return True


def createConsistencyText(cleaned_annotations, text):
    for a in cleaned_annotations:
        if (text[a['start']:a['end'] + 1]).lower() != a["text"].lower():
            a['text'] = text[a['start']:a['end'] + 1].lower()
    return cleaned_annotations


def consistencyText(cleaned_annotations, text):
    for a in cleaned_annotations:
        if (text[a['start']:a['end'] + 1]).lower() != a["text"].lower():
            # print('text',text[a['start']:a['end']+1])
            # print('surface',a["text"])
            return False
    return True


def splitUpperCaseTypes(anns):
    for ann in anns:
        spl = re.findall('[A-Z][^A-Z]*', ann['type'])
        if len(spl) > 1:
            ann['type'] = ','.join(spl)
    return anns
