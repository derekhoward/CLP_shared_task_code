import feature_groups
import pandas
import numpy
import config
import os
import gc
import time
from shutil import copyfile
from tpot import TPOTClassifier
from tpot import TPOTRegressor
from sklearn import metrics
from sys import platform
from sklearn import model_selection

#from previous_post_features import add_features

run_sentence_experiment = False #this overrides target label and features and add_previous_post_features
cv_folds = 10
random_state=43
max_time_mins= 3*60  #how long to run tpot
max_eval_time_mins = 5 #5 #how many minutes is a single pipeline allowed
target_label = 'label' #granular_label or label or linear_label
population_size = 200

add_previous_post_features = False
#add_previous_post_features = True

tryCrisisFocus = True

#features_to_use = feature_groups.ALL_WHEN_POSTED - feature_groups.DOC2VEC_BODY_COLS_50
#features_to_use = feature_groups.TF_IDF_2GRAM_COLS
#features_to_use = feature_groups.TF_IDF_COLS
#features_to_use = feature_groups.ALL_WHEN_POSTED_INCL_OPENAI
#features_to_use = feature_groups.ALL_WHEN_POSTED_INCL_TF_IDF
#features_to_use = feature_groups.ALL_WHEN_POSTED_INCL_ALL_TF_IDF_COLS
#features_to_use = feature_groups.ALL_WHEN_POSTED_INCL_TF_IDF_AND_OPENAI
#features_to_use = feature_groups.ALL_WHEN_POSTED_INCL_TF_IDF_2GRAM_AND_OPENAI

#features_to_use = feature_groups.ALL_WHEN_POSTED - feature_groups.DOC2VEC_COLS #features except doc2vec
#features_to_use = feature_groups.DOC2VEC_BODY_COLS_100 | feature_groups.DOC2VEC_SUBJECT_COLS_100 #subject + body 100
#features_to_use = feature_groups.DOC2VEC_BODY_COLS_100
#features_to_use = feature_groups.DOC2VEC_COLS

#features_to_use = feature_groups.OPENAI_SENTIMENT_COLS
#features_to_use = feature_groups.DEEPMOJI_COLS
features_to_use = feature_groups.UNIVERSAL_COLS

######Experimental
if run_sentence_experiment:
    print("Using sentence level experiment settings, overriding some settings")
    features_to_use = feature_groups.OPENAI_SENTIMENT_CORRELATED_SUPERVISED_COLS
    target_label = 'linear_label'
    add_previous_post_features = False
###### END Experimental

start_time = str(round(time.time()))

tpot_base_folder = os.path.join(config.DATA_DIR,'interim', 'best_tpot')
full_tpot_out_filename = os.path.join(tpot_base_folder, 'tpot_exported_pipeline.' + target_label + ".time." + start_time + '.basic_tpot.py') 
print("TPOT code written to:" + full_tpot_out_filename)
#copyfile(os.path.realpath(__file__), full_tpot_out_filename)


if platform == "darwin": #multithreading fails on osx for tpot due to some BLAS thing
    n_jobs = 1
    config_dict = None
else: 
    n_jobs = config.CORES

if len(features_to_use & feature_groups.OPENAI_SENTIMENT_COLS) > 0: #if loading openAI features -use less memory and use light version
    config_dict = 'TPOT light'
elif len(features_to_use & feature_groups.TF_IDF_COLS) > 0: #if loading TF_IDF features -use less memory and use light version
    config_dict = 'TPOT light'
else:
    config_dict = None

if target_label == 'linear_label':
    config_dict = 'TPOT light'

#now using the slimmed file 
#df = pandas.read_csv(os.path.join(config.DATA_DIR,'interim', 'processed_features_plus_doc2vec.plus_tf_idf_openAI.slimmed.csv'), low_memory=False)
df = pandas.read_csv(os.path.join(config.DATA_DIR,'interim', 'all_features_slim.csv'), low_memory=False)

######Experimental
if run_sentence_experiment:
    df = pandas.read_csv(os.path.join(config.DATA_DIR,'interim', 'sentence_level_openAI_features_only.csv'), low_memory=False)
###### END Experimental
    

#### Add features from previous post ####
if add_previous_post_features:
    print("Adding previous features")
    previous_post_df = pandas.read_csv(os.path.join(config.DATA_DIR,'interim', 'processed_features_plus_doc2vec.csv'), low_memory=False)
    previous_features_to_use = features_to_use - feature_groups.MISSING_IN_UNLABELLED_SET
    added_features_to_use, df = add_features(previous_features_to_use, df, previous_post_df=previous_post_df)
    #add in the new features
    features_to_use = features_to_use | added_features_to_use
    print("Done previous features. " + str(len(added_features_to_use)) + " features added.")
############  

print("Using TPOT config: " + str(config_dict))

#labelled data only
df = df[df['label'].notnull()]
print(df.shape)

X_train = numpy.array(df[list(features_to_use)])
X_train = X_train.astype(float) 


#score function - some code duped from test_many_doc2vec_models
def get_macroF1(y_true, y_pred):
    return metrics.f1_score(y_true, y_pred, labels=["red", "amber", "crisis"], average="macro" )


def get_macroF1_crisis(y_true, y_pred):
    return metrics.f1_score(y_true, y_pred, labels=["crisis"], average="macro" )

#convert between granular and macro, could be combined into a single score function
def get_macroF1_from_granular(y_true, y_pred):
    label_dict = {'currentMildDistress' : 'amber', 'followupOk' : 'amber', 'pastDistress' : 'amber', 'underserved' : 'amber', 'crisis' : 'crisis', 'allClear' : 'green', 'followupBye' : 'green', 'supporting' : 'green', 'angryWithForumMember' : 'red', 'angryWithReachout' : 'red', 'currentAcuteDistress' : 'red', 'followupWorse' : 'red'}
    y_true = [ label_dict[x] for x in y_true]
    y_pred = [ label_dict[x] for x in y_pred]
    return metrics.f1_score(y_true, y_pred, labels=["red", "amber", "crisis"], average="macro" )

metrics.SCORERS['macroF1MinusGreen'] = metrics.make_scorer(get_macroF1)
metrics.SCORERS['macroF1MinusCrisis'] = metrics.make_scorer(get_macroF1_crisis)
metrics.SCORERS['macroF1FromGranular'] = metrics.make_scorer(get_macroF1_from_granular)

print(X_train.shape)

#use whatever our target is
y_train = df[target_label]
print(y_train.shape)

#to save space when/if tpot forks the process (not sure if this helps)
df = None
previous_post_df = None
gc.collect()

kf = model_selection.KFold(n_splits=cv_folds, random_state=random_state, shuffle=True)

if target_label == "linear_label": 
    print("Running regression")
    tpot = TPOTRegressor(population_size=population_size, verbosity=2,  
                      random_state = random_state, cv= kf, n_jobs= n_jobs, 
                      max_time_mins=max_time_mins, max_eval_time_mins= max_eval_time_mins,
                      config_dict = config_dict)
    tpot.fit(X_train, y_train )
else: #label or granular label
    score_function = "macroF1MinusGreen" if target_label == "label" else "macroF1FromGranular"
    if tryCrisisFocus:
        score_function = "macroF1MinusCrisis"
    tpot = TPOTClassifier(population_size=population_size, verbosity=2, scoring= score_function, 
                      random_state = random_state, cv= kf, n_jobs= n_jobs, 
                      max_time_mins=max_time_mins, max_eval_time_mins= max_eval_time_mins,
                      config_dict = config_dict)
    tpot.fit(X_train, y_train )
    

score_df = pandas.DataFrame.from_dict(tpot.evaluated_individuals_, orient='index')
score_df.columns = ['generation', 'score']
score_df.sort_values('score', ascending = False)
top_score = score_df.score.max()
top_score_string =str(round(top_score,ndigits=3))
print("Best score = " + top_score_string)
#print(score_df[score_df['score'] == float("-inf")])
print("Runs with neg infinity score (didn't finish?):" + str(score_df[score_df['score'] == float("-inf")].shape[0]))

#write out
if not os.path.exists(tpot_base_folder):
    os.makedirs(tpot_base_folder)
full_tpot_out_filename =os.path.join(tpot_base_folder, 'tpot_exported_pipeline.score.' + top_score_string + ".features." + str(X_train.shape[1]) + "."  + target_label + ".addPreviousFeats." + str(add_previous_post_features) + ".time." + start_time + '.py') 
print("Best pipeline written to:" + full_tpot_out_filename)
tpot.export(full_tpot_out_filename)