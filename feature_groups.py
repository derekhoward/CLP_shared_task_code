# -*- coding: utf-8 -*-

import pandas
import config
import os

#just read the header
#df = pandas.read_csv(os.path.join(config.DATA_DIR,'interim', 'processed_features_plus_doc2vec_plus_openAI.csv'), low_memory=False, nrows=1)
df = pandas.read_csv(os.path.join(config.DATA_DIR,'interim', 'all_features_slim.csv'), low_memory=False, nrows=1)

#print(str(list(df.columns)))

#label columns
LABEL_COLS = {'label', 'granular_label', 'linear_label', 'training'}

#doc2vec column - Doc2Vec
DOC2VEC_COLS = {x for x in list(df.columns) if "Doc2Vec" in x}

DOC2VEC_BODY_COLS_50 = {x for x in list(df.columns) if x.startswith("Doc2VecD50")}

DOC2VEC_BODY_COLS_100 = {x for x in list(df.columns) if x.startswith("Doc2VecD100")}

DOC2VEC_SUBJECT_COLS_100 = {x for x in list(df.columns) if x.startswith("SubjectDoc2VecD100W")}

VADER_COLS = {x for x in list(df.columns) if x.startswith("vader")}

EMPATH_COLS = {x for x in list(df.columns) if x.startswith("empath")}

TF_IDF_2GRAM_COLS = {x for x in list(df.columns) if x.startswith("tf_idf_2gram_")}

TF_IDF_COLS = {x for x in list(df.columns) if x.startswith("tf_idf_")} - TF_IDF_2GRAM_COLS

ALL_TF_IDF_COLS = {x for x in list(df.columns) if x.startswith("tf_idf_")} #bad naming here - done in a rush

LDA_COLS = {x for x in list(df.columns) if x.startswith("LDA")}

BOARD_COLS = {x for x in list(df.columns) if "board_1hot_" in x}
AUTHOR_RANK_COLS = {x for x in list(df.columns) if "author_rank_1hot_" in x}

OPENAI_SENTIMENT_COLS = {x for x in list(df.columns) if x.startswith("openAI_")}

SENTOPENAI_SENTIMENT_COLS = {x for x in list(df.columns) if x.startswith("sent_openAI_")}

DEEPMOJI_COLS = {x for x in list(df.columns) if "deepmoji_" in x}

#sentence level deepmoji encoded as length 64 vector
EMOJI_COLS = {x for x in list(df.columns) if "emoji_" in x}

UNIVERSAL_COLS = {x for x in list(df.columns) if "universal_" in x}

SENTVADER_COLS = {x for x in list(df.columns) if x.startswith("sentvader")}

SENTEMPATH_COLS = {x for x in list(df.columns) if x.startswith("sentempath")}
#Free text columns
NON_NUMERIC_COLS = {'subject', 'cleaned_body', 'images', 'board', 'author_rank', 'post_time'}

#id columns
#'board', 'author_rank' need one-hot conversion
ID_COLS = { 'post_id', 'index', 'author_id', 'board_id', 'edit_author_id', 'parent_post', 'thread', 'following_post_id', 'previous_post_id', 'following_staff_post_id', 'previous_staff_post_id'}

#columns we won't have at test time
#views, kudos
NOT_AVAIL_WHEN_FIRST_POSTED = {'kudos_count', 'last_edit_time', 'views', 'following_post_id', 'following_staff_post_id'}

#ALL_WHEN_POSTED = set(df.columns) - NOT_AVAIL_WHEN_FIRST_POSTED - ID_COLS - LABEL_COLS - NON_NUMERIC_COLS - OPENAI_SENTIMENT_COLS
ALL_WHEN_POSTED = set(df.columns) - NOT_AVAIL_WHEN_FIRST_POSTED - ID_COLS - LABEL_COLS - NON_NUMERIC_COLS - DEEPMOJI_COLS - OPENAI_SENTIMENT_COLS - UNIVERSAL_COLS

#ALL_WHEN_POSTED_INCL_OPENAI = ALL_WHEN_POSTED | OPENAI_SENTIMENT_COLS

#columns we will have at test time - everything else?
print("feature_groups.py set: " + str([x for x in locals().keys() if x.upper() == x]))


