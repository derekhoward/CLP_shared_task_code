import os
import pandas as pd
import spacy
import config

import en_core_web_sm
nlp = en_core_web_sm.load()

df = pd.read_csv(os.path.join(config.DATA_DIR,'interim', 'processed_features.csv'),
                 usecols=['post_id', 'cleaned_body', 'label', 'predict_me'],
                 low_memory=False)

# select rows of interest
df = df[(df.label.notnull()) | (df.predict_me == True)]

df['parsed'] = df['cleaned_body'].astype(str).apply(nlp)


corpus_dict = {}
for doc in df.iterrows():
    document_dict = {}
    post_id = doc[1].post_id
    parsed_doc = doc[1].parsed
    
    for i, sentence in enumerate(parsed_doc.sents):
        document_dict[i] = str(sentence)
        
    corpus_dict[post_id] = document_dict

sentences = pd.DataFrame(corpus_dict).T.reset_index().rename(columns={'index': 'post_id'})
sentences = pd.melt(sentences, id_vars='post_id', var_name='sentence_num', value_name='body')
sentences = sentences.dropna().sort_values(['post_id', 'sentence_num'])

sentences.body = sentences.body.str.replace('1.', '').str.replace('2.', '').str.replace('3.', '')
sentences.body = sentences.body.str.replace('4.', '').str.replace('5.', '').str.replace('6.', '')
sentences.body = sentences.body.str.replace('7.', '').str.replace('8.', '').str.replace('9.', '')
sentences.body = sentences.body.str.replace('-', '')
sentences.body = sentences.body.str.strip()


cleaned_sents = []
for i, row in sentences.iterrows():
    if len(row.body) > 2:
        cleaned_sents.append([row.post_id, row.sentence_num, row.body])

cleaned_sents = pd.DataFrame(cleaned_sents, columns=['post_id', 'sentence_num', 'body'])
cleaned_sents = cleaned_sents[cleaned_sents.body != 'nan']

cleaned_sents.to_csv('./data/interim/sentences.csv', index=None)
