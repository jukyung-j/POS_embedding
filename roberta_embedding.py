import os

import tokenizers
import torch
from torch import nn

from functools import reduce
import operator

import pandas as pd
import numpy as np
from konlpy.tag import Mecab
from transformers import AutoModel, AutoTokenizer
import nltk
from nltk import FreqDist

# vocab : 85293
# total_len : 150081

files = ['./corpus/every_corpus_preprocessing_dev.txt','./corpus/every_corpus_preprocessing_test.txt','./corpus/every_corpus_preprocessing_train.txt']
# files = ['./corpus/every_corpus_preprocessing_dev.txt']
tagger = [
        "NNG",
        "NNP",
        "NNB",
        "NP",
        "NR",
        "VV",
        "VA",
        "VX",
        "VCP",
        "VCN",
        "MMA",
        "MMD",
        "MMN",
        "MAG",
        "MAJ",
        "JC",
        "IC",
        "JKS",
        "JKC",
        "JKG",
        "JKO",
        "JKB",
        "JKV",
        "JKQ",
        "JX",
        "EP",
        "EF",
        "EC",
        "ETN",
        "ETM",
        "XPN",
        "XSN",
        "XSV",
        "XSA",
        "XR",
        "SF",
        "SP",
        "SS",
        "SE",
        "SO",
        "SL",
        "SH",
        "SW",
        "SN",
        "NA",
        "NF",
        "NV",
    ]

dependent = ['EP', 'EF', 'EC', 'ETN', 'ETM', 'XSN', 'XSV', 'XSA', 'VCP', 'VCN', 'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC', 'NNB']
tag = {}
for i, t in enumerate(tagger):
    tag[t] = i       # CLS:47, SEP:48

document = []
sentences = []
tokens = []
count = 0

for file_path in files:
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "" or line == "\n" or line == "\t":
                continue
            if line.startswith("#"):
                pos_ids = []
                token = []
                parsed = line.strip().split("\t")
                if len(parsed) != 2:  # metadata line about dataset
                    continue
                else:
                    # sent_id += 1
                    text = parsed[1].strip()
                    guid = parsed[0].replace("##", "").strip()
                    document.append(text)
                    sentences.append(tokens)
                    tokens = []
            else:
                token_list = [token.replace("\n", "") for token in line.split("\t")] # lemma : 2, pos : 3
                tokens.append((token_list[2], token_list[3]))

sentences.pop(0)

# sentence preprocessing
MAX_LEN = 512
total_len = len(sentences)
input_ids = np.ones(shape=[total_len, MAX_LEN], dtype="int32")
input_masks = np.zeros(shape=[total_len, MAX_LEN], dtype="int32")
input_pos = np.zeros(shape=[total_len, MAX_LEN], dtype="int32")
print("total_len: ", total_len)

# tokenizer
tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")

for i in range(total_len):
    pos_ids = []
    token_list = []
    sentence = []
    for lemma, pos in sentences[i]:
        lemma_list, pos_list = [], []
        pos_list.append(pos)
        token = tokenizer.tokenize(lemma)
        sentence.append(lemma)
        token_list.append(token)
        lemma_list.append(lemma)
        if '+' in pos:
            pos_list = pos.split('+')
            lemma_list = lemma.split()

        # 길이 맞추기
        lem_c = 0
        temp = ""
        for j in range(len(token)):
            string = temp + token[j].replace('##', '')
            if string == lemma_list[lem_c]:
                pos_ids.append(tag[pos_list[lem_c]])
                lem_c += 1
                temp = ""
                continue
            temp = string
            pos_ids.append(tag[pos_list[lem_c]])

    pos_ids.insert(0, 47) # CLS
    pos_ids.append(48) # SEP
    token_list = list(reduce(operator.add, token_list))

    sentence = ' '.join(sentence)
    token_id = tokenizer(sentence)['input_ids']

    if len(pos_ids) != len(token_id):
        print("길이 안맞음")

    length = len(token_id)
    if length > MAX_LEN:
        length = MAX_LEN

    for j in range(length):
        input_ids[i, j] = token_id[j]
        input_masks[i, j] = 1
        input_pos[i, j] = pos_ids[j]


# npy 저장
if not os.path.isdir('./corpus/roberta_npy'):
    os.mkdir('./corpus/roberta_npy')
np.save('./corpus/roberta_npy/input_ids', input_ids)
np.save('./corpus/roberta_npy/input_mask', input_masks)
np.save('./corpus/roberta_npy/input_pos', input_pos)
