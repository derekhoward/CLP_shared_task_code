import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from dotenv import load_dotenv
from pathlib import Path
import os
from build_features import create_empath_df
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def flatten_cols(df):
    df.columns = [
        '_'.join(tuple(map(str, t))).rstrip('_') 
        for t in df.columns.values
        ]
    return df


def main():
    sentences = pd.read_csv('./data/interim/sentences.csv')
    sentences.body = sentences.body.astype(str)
    
    # vader features
    vader_scores = sentences.drop('sentence_num', axis=1).set_index('post_id').body.astype(str).apply(vader.polarity_scores)
    vader_scores = vader_scores.apply(pd.Series)
    # aggregate scores by post_id
    vader_df = vader_scores.add_prefix('sentvader_').groupby('post_id').agg(['mean', 'max', 'min'])
    vader_df = flatten_cols(vader_df)
    
    # empath features
    empath_df = create_empath_df(sentences.loc[:, ['post_id', 'body']].set_index('post_id').body)
    empath_df = empath_df.groupby('post_id').agg(['mean', 'max', 'min'])
    empath_df = flatten_cols(empath_df)
    empath_df = empath_df.add_prefix('sent')

    
    # Universal sentence encoder features
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/1"
    embed = hub.Module(module_url)
    messages = list(sentences.body.astype(str))
    
    tf.logging.set_verbosity(tf.logging.ERROR)

    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        message_embeddings = session.run(embed(messages))
    
    universal_embeddings = pd.DataFrame(message_embeddings)
    universal_embeddings = universal_embeddings.add_prefix('universal_sent_')
    universal_embeddings.index = sentences.post_id
    universal_embeddings = universal_embeddings.groupby('post_id').agg(['mean', 'max', 'min'])
    universal_embeddings = flatten_cols(universal_embeddings)

    
    # merging all features on post_id
    print(f'vader_df shape: {vader_df.shape}')
    print(f'empath_df shape: {empath_df.shape}')
    print(f'universal_df shape: {universal_embeddings.shape}')
    empath_vader = vader_df.merge(empath_df, left_index=True, right_index=True)
    sentence_level_features = empath_vader.merge(universal_embeddings, left_index=True, right_index=True)
    print(f'shape of final empath,vader, universal sentence level feature df is : {sentence_level_features.shape}')
    
    sentence_level_features.to_csv('./data/interim/sent_lvl_empath_vader_universal.csv')


if __name__ == '__main__':
    vader = SentimentIntensityAnalyzer()
    # to download tfhub model to a specified directory (in .env file)
    env_path = Path('.') / '.env'
    load_dotenv(dotenv_path=env_path)
    os.getenv('TFHUB_CACHE_DIR')
    main()    
    
    