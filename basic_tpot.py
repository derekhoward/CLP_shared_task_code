import feature_groups
import pandas
import numpy
import config
import os
import pathlib
import gc
import time
from shutil import copyfile
from tpot import TPOTClassifier
from tpot import TPOTRegressor
from sklearn import metrics
from sys import platform
from sklearn import model_selection

run_sentence_experiment = False #this overrides target label and features and add_previous_post_features
cv_folds = 10
random_state=43
max_time_mins= 44*60  #how long to run tpot
max_eval_time_mins = 5 #5 #how many minutes is a single pipeline allowed
target_label = 'label' #granular_label or label or linear_label
population_size = 200

add_previous_post_features = False
#add_previous_post_features = True

tryCrisisFocus = False

################################################################################
########################## MODIFY FEATURES TO USE ##############################
################################################################################

# make sure to change feats string to update the output properly in tpotname
feats = 'EMOJI_COLS'

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

#sentence level features
#features_to_use = feature_groups.SENTOPENAI_SENTIMENT_COLS 
#features_to_use = feature_groups.DEEPMOJI_COLS
features_to_use = feature_groups.EMOJI_COLS
#features_to_use = feature_groups.SENTVADER_COLS
#features_to_use = feature_groups.SENTEMPATH_COLS
#features_to_use = feature_groups.UNIVERSAL_COLS


################################################################################
################################################################################
################################################################################


start_time = str(round(time.time()))

models_base_folder = pathlib.Path.cwd() / 'models'
tpot_base_folder = models_base_folder / 'best_tpot'
tpot_name = 'tpot_exported_pipeline.' + target_label + '.feats.' + feats + '.time.' + start_time + '.basic_tpot.py'
full_tpot_out_filename = tpot_base_folder / tpot_name


print("TPOT code written to:", full_tpot_out_filename)
copyfile(os.path.realpath(__file__), full_tpot_out_filename)


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
df = pandas.read_csv(os.path.join(config.DATA_DIR,'interim', 'final_features_slim.csv'), low_memory=False)

    

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



#kf = model_selection.KFold(n_splits=cv_folds, random_state=random_state, shuffle=True)

# stratified CV:
#kf = model_selection.StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
kf = model_selection.RepeatedStratifiedKFold(n_splits=cv_folds, repeats=10, random_state=random_state)


#generate a folder to write checkpoint models
checkpoint_folder = models_base_folder / 'checkpoint_models' 
checkpoint_folder.mkdir(exist_ok=True, parents=True)

if target_label == "linear_label": 
    print("Running regression")
    tpot = TPOTRegressor(population_size=population_size, verbosity=2,  
                      random_state = random_state, cv=kf, n_jobs= n_jobs, 
                      max_time_mins=max_time_mins, max_eval_time_mins= max_eval_time_mins,
                      config_dict = config_dict)
    tpot.fit(X_train, y_train)
else: #label or granular label
    score_function = "macroF1MinusGreen" if target_label == "label" else "macroF1FromGranular"
    if tryCrisisFocus:
        score_function = "macroF1MinusCrisis"
    tpot = TPOTClassifier(population_size=population_size, verbosity=2, scoring=score_function, 
                      random_state = random_state, cv=kf, n_jobs=n_jobs, 
                      max_time_mins=max_time_mins, max_eval_time_mins=max_eval_time_mins,
                      config_dict = config_dict, memory='auto', periodic_checkpoint_folder=checkpoint_folder)
    tpot.fit(X_train, y_train )
    

tpot.export(full_tpot_out_filename)
