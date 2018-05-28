from __future__ import print_function, division
import json
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import unicodedata

from examples import example_helper
from deepmoji.create_vocab import extend_vocab, VocabBuilder
from deepmoji.sentence_tokenizer import SentenceTokenizer
from deepmoji.model_def import deepmoji_feature_encoding,deepmoji_emojis
from deepmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH

def flatten_cols(df):
    df.columns = [
        '_'.join(tuple(map(str, t))).rstrip('_') 
        for t in df.columns.values
        ]
    return df

def main():
    df = pd.read_csv('../data/interim/sentences.csv')

    maxlen = 30
    batch_size = 32

    print('Tokenizing using dictionary from {}'.format(VOCAB_PATH))
    with open(VOCAB_PATH, 'r') as f:
        vocabulary = json.load(f)

    st = SentenceTokenizer(vocabulary, maxlen)

    sentences = []
    for sent in df.body.tolist():
        sent = unicode(str(sent), "utf-8")
        if sent.strip() == "":
            sent = 'blank'
            sent = unicode(str(sent), "utf-8")
        sentences.append(sent)

    tokenized, _, _ = st.tokenize_sentences(sentences)

    print('Loading model from {}.'.format(PRETRAINED_PATH))
    model = deepmoji_feature_encoding(maxlen, PRETRAINED_PATH)
    model.summary()

    print('Encoding texts..')
    encoding = model.predict(tokenized)

    encodings = pd.DataFrame(encoding)
    encodings = pd.concat([df, encodings], axis=1)

    aggregated = encodings.drop(['sentence_num', 'body'], axis=1).groupby('post_id').agg(['mean', 'max', 'min'])
    final = flatten_cols(aggregated)
    deepmoji_feats = pd.DataFrame(final.to_records())
    deepmoji_feats = deepmoji_feats.add_prefix('deepmoji_')

    deepmoji_feats.to_csv('../data/interim/deepmoji_features.csv', index=None)

if __name__ == "__main__":
    main()