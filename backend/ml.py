import re
import time

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from fuzzywuzzy import fuzz
import numpy as np
from bisect import bisect_left
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
import json
names = pd.read_csv('./names.csv', header=None)
names_small = pd.read_csv('./names_mini (1).csv', header=None)
loaded_data = np.load('arrays.npz', allow_pickle=True)
names['vectors'] = loaded_data['Emb']
names['index'] = names[0]
names = names.drop([0], axis=1)
names['text'] = names[1]
names = names.drop([1], axis=1)
names = names.reset_index(drop=True)
loaded_data_small = np.load('arrays_mini.npz', allow_pickle=True)
names_small['vectors'] = loaded_data_small['Emb']
names_small['index'] = names_small[0]
names_small = names_small.drop([0], axis=1)
names_small['text'] = names_small[1]
names_small = names_small.drop([1], axis=1)
names_small = names_small.reset_index(drop=True)
vec_df_small = names_small
vec_df = names
tokenizer = AutoTokenizer.from_pretrained("ai-forever/sage-fredt5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("ai-forever/sage-fredt5-large")

stopwords_ru = './stopwords_csr.csv'
stemmer = SnowballStemmer("russian")
stopwords_ru = pd.read_csv(stopwords_ru, header=None)[0].tolist()
with open('index_map.json', 'r', encoding='utf-8') as f:
    map_ind = json.load(f)

from sentence_transformers import SentenceTransformer

vec_model = SentenceTransformer("Salesforce/SFR-Embedding-Mistral", device='cuda')

def vectorize_text(text):
    return vec_model.encode(text)


def count_min_cosine(embedding, embeddings):
    mn = 1
    mn_pred = 1
    ind = -1
    ind2 = -1
    min_emb = None
    embeddings = embeddings[0]
    for emb in range(len(embeddings)):
        mn = min(mn, (1 - F.cosine_similarity(torch.Tensor([embedding]), torch.Tensor([embeddings[emb]]))).item())
        if mn_pred != mn:
            min_emb = embeddings[emb]
            ind2 = ind
            ind = emb
        mn_pred = mn

    return ind, ind2

def get_top(promt, s, k_ma=200):
    ma = []
    for index, i in enumerate(s):
        rr = fuzz.token_sort_ratio(i, promt)
        ind = bisect_left(ma, (rr, i, index))
        ma.insert(ind, (rr, i, index))
        if len(ma) > k_ma:
            ma.pop(0)
    ans = []
    for i, j, k in ma:
        ans.append(k)
    return ans


import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize


def stem_func(text, stemmer):
    tokens = word_tokenize(text)
    stemmed_words = [stemmer.stem(word) for word in tokens]

    return stemmed_words


def stopwords(text_arr, stopwords_ru):
    trash_indx = []
    new_text = []
    for indx, word in enumerate(text_arr):
        for stopword in stopwords_ru:
            if word in stopword:
                trash_indx.append(indx)

    for indx, word in enumerate(text_arr):
        if indx not in trash_indx:
            new_text.append(word)

    return new_text


def get_tops(s1, texts, vectors, k_ma=400):

    ans = []
    res_text = []
    for i in s1:
        ind = get_top(i, texts, k_ma)
        for s in ind:
            ans.append(vectors[s])
            res_text.append(texts[s])
    print(len(ans))
    return ans, res_text

def data_cleaning(text):
    text = text.lower()
    pattern = r"[:№.,!?/{}()\+=@$%^&*0-9]+"
    pattern2 = r"[\d]+"
    text = re.sub(pattern, "", text)
    text = re.sub(pattern2, "", text)
    text = re.sub(r'\s+', ' ', text).strip().rstrip()
    text = spelling_corr(text)
    #text = stem_func(text, stemmer)
    #text = stopwords(text, stopwords_ru)
    return text
def preproc_for_search(text, stemmer, stopwords_ru):
    text = stem_func(text, stemmer)
    text = stopwords(text, stopwords_ru)
    return list(set(text))

def search_mathing_df(text, df_cols, vectors):
    texts = []
    vecs = []
    for csr_name in range(len(df_cols.values)):
        csr_name_token = list(df_cols.values[csr_name].split())
        for word in text:
            for csr_word in csr_name_token:
                if word in csr_word:
                    texts.append(df_cols.values[csr_name])
                    vecs.append(vectors[csr_name])
    return vecs, texts
def spelling_corr(text):
    inputs = tokenizer(text, max_length=None, padding="longest", truncation=False, return_tensors="pt")
    outputs = model.generate(**inputs.to(model.device), max_length=inputs["input_ids"].size(1) * 1.5)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
def pipeline(array):
    ans = []
    tops_vecs = []
    tops_text = []
    new_vecs = []
    new_texts = []
    words_for_search = []
    for i in array:
        ans.append(data_cleaning(i))
        words_for_search.append(preproc_for_search(i, stemmer, stopwords_ru))
    print(1)
    vectors = vectorize_text(ans)
    print(2)
    res = []
    for i in words_for_search:
        vecs, texts = search_mathing_df(i, vec_df_small["text"], vec_df_small["vectors"])
        new_vecs.append(vecs)
        new_texts.append(texts)
    print(3)

    for i in ans:
        vecs, texts = get_tops(i, new_texts, new_vecs)
        tops_text.append(texts)
        tops_vecs.append(vecs)
    print(4)
    for i in range(len(vectors)):
        ind, ind2 = count_min_cosine(vectors[i], tops_vecs[i])
        print(ind, len(tops_text[i]), len(tops_vecs[i]))
        res.append(tops_text[0][i][ind])

    return res

df = pd.read_csv("col_test_csr.csv")
pipeline(df["name"].values.tolist())
lst = df["name"].values.tolist()
for s in lst:
    print(pipeline([s]))