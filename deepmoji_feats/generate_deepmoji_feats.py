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

    # generate full deepmoji features for sentences
    print('Loading model from {}.'.format(PRETRAINED_PATH))
    model = deepmoji_feature_encoding(maxlen, PRETRAINED_PATH)
    model.summary()

    print('Encoding texts with deepmoji features...')
    encoding = model.predict(tokenized)

    deepmoji_encodings = pd.DataFrame(encoding)
    deepmoji_encodings.index = df.post_id
    
    deepmoji_post_scores = deepmoji_encodings.groupby('post_id').agg(['mean', 'max', 'min'])
    deepmoji_post_scores = flatten_cols(deepmoji_post_scores)
    deepmoji_post_scores = deepmoji_post_scores.add_prefix('deepmoji_')
    
    # generate 64 emoji encodings
    print('Loading model from {}.'.format(PRETRAINED_PATH))
    model = deepmoji_emojis(maxlen, PRETRAINED_PATH)
    model.summary()

    print('Running emoji predictions...')
    prob = model.predict(tokenized)
    emoji_scores = pd.DataFrame(prob)
    emoji_scores = emoji_scores.add_prefix('emoji_')
    emoji_scores.index = df.post_id
    
    emoji_post_scores = emoji_scores.groupby('post_id').agg(['mean', 'max', 'min'])
    emoji_post_scores = flatten_cols(emoji_post_scores)
    
    print('deepmoji features shape: {}'.format(deepmoji_post_scores.shape))
    print('emoji features shape: {}'.format(emoji_post_scores.shape))
    total_feats = deepmoji_post_scores.merge(emoji_post_scores, left_index=True, right_index=True)
    print('total features shape: {}'.format(total_feats.shape))
    total_feats.to_csv('../data/interim/all_sent_level_deepmoji.csv')

if __name__ == "__main__":
    main()