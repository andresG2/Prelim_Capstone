from sklearn import linear_model
import pandas as pd
import pickle
import tensorflow as tf
#import seaborn as sns


# Create neural net
import numpy as np # linear algebra
import pandas as pd
import io
import pip._vendor.requests
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn import metrics
#import tensorflow as tf
#from tensorflow import keras
#from tensorflow import layers
#from tensorflow import Sequential
#from tensorflow import Dense, Activation
#from tensorflow import EarlyStopping
import numpy as np # linear algebra
import pandas as pd
from tensorflow import get_file
from keras.utils.data_utils import get_file
from scipy.stats import zscore
import matplotlib as mpl

#from symbol import tfpdef

from flask import Flask, request, render_template


# LOAD THE DATA ############################################################
try:
    path = get_file('kddcup.data_10_percent.gz', origin='http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz')
except:
    print('Error downloading')
    raise

# This file is a CSV, just no CSV extension or headers
# Download from: http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
df = pd.read_csv(path, header=None)

#############################################################################
# The CSV file has no column heads, so add them to the dataframe
df.columns = [
    'duration',
    'protocol_type',
    'service',
    'flag',
    'src_bytes',
    'dst_bytes',
    'land',
    'wrong_fragment',
    'urgent',
    'hot',
    'num_failed_logins',
    'logged_in',
    'num_compromised',
    'root_shell',
    'su_attempted',
    'num_root',
    'num_file_creations',
    'num_shells',
    'num_access_files',
    'num_outbound_cmds',
    'is_host_login',
    'is_guest_login',
    'count',
    'srv_count',
    'serror_rate',
    'srv_serror_rate',
    'rerror_rate',
    'srv_rerror_rate',
    'same_srv_rate',
    'diff_srv_rate',
    'srv_diff_host_rate',
    'dst_host_count',
    'dst_host_srv_count',
    'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate',
    'dst_host_srv_serror_rate',
    'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate',
    'outcome'
]


# Data Cleaning #############################################################

# Checking for NULL values
print('Null values in dataset are',len(df[df.isnull().any(1)]))
print('='*40)

# Checking for DUPLICATE values
df.drop_duplicates(keep='first', inplace = True)

# For now, just drop NA's (rows with missing values)
df.dropna(inplace=True,axis=1) 

# stored the data into a pickle file so we can load through
# df.to_pickle('df.pkl')







# Encode a numeric column as zscores
def encode_numeric_zscore(df, name, mean=None, sd=None):
    if mean is None:
        mean = df[name].mean()

    if sd is None:
        sd = df[name].std()

    df[name] = (df[name] - mean) / sd



# Encode the numeric column to zscores #########################################################   
# Encode text values to dummy variables(i.e. [1,0,0],
# [0,1,0],[0,0,1] for red,green,blue)
def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = f" {name} -{x}"
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)


# Encode to Dummy Variables ######################################################################   
# Encode text values to dummy variables(i.e. [1,0,0],[0,1,0],[0,0,1] for red,green,blue)
def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = f"{name}-{x}"
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)



# Now encode the feature vector ################################################################

encode_numeric_zscore(df, 'duration')
encode_text_dummy(df, 'protocol_type')
encode_text_dummy(df, 'service')
encode_text_dummy(df, 'flag')
encode_numeric_zscore(df, 'src_bytes')
encode_numeric_zscore(df, 'dst_bytes')
encode_text_dummy(df, 'land')
encode_numeric_zscore(df, 'wrong_fragment')
encode_numeric_zscore(df, 'urgent')
encode_numeric_zscore(df, 'hot')
encode_numeric_zscore(df, 'num_failed_logins')
encode_text_dummy(df, 'logged_in')
encode_numeric_zscore(df, 'num_compromised')
encode_numeric_zscore(df, 'root_shell')
encode_numeric_zscore(df, 'su_attempted')
encode_numeric_zscore(df, 'num_root')
encode_numeric_zscore(df, 'num_file_creations')
encode_numeric_zscore(df, 'num_shells')
encode_numeric_zscore(df, 'num_access_files')
encode_numeric_zscore(df, 'num_outbound_cmds')
encode_text_dummy(df, 'is_host_login')
encode_text_dummy(df, 'is_guest_login')
encode_numeric_zscore(df, 'count')
encode_numeric_zscore(df, 'srv_count')
encode_numeric_zscore(df, 'serror_rate')
encode_numeric_zscore(df, 'srv_serror_rate')
encode_numeric_zscore(df, 'rerror_rate')
encode_numeric_zscore(df, 'srv_rerror_rate')
encode_numeric_zscore(df, 'same_srv_rate')
encode_numeric_zscore(df, 'diff_srv_rate')
encode_numeric_zscore(df, 'srv_diff_host_rate')
encode_numeric_zscore(df, 'dst_host_count')
encode_numeric_zscore(df, 'dst_host_srv_count')
encode_numeric_zscore(df, 'dst_host_same_srv_rate')
encode_numeric_zscore(df, 'dst_host_diff_srv_rate')
encode_numeric_zscore(df, 'dst_host_same_src_port_rate')
encode_numeric_zscore(df, 'dst_host_srv_diff_host_rate')
encode_numeric_zscore(df, 'dst_host_serror_rate')
encode_numeric_zscore(df, 'dst_host_srv_serror_rate')
encode_numeric_zscore(df, 'dst_host_rerror_rate')
encode_numeric_zscore(df, 'dst_host_srv_rerror_rate')

# display 5 rows

df.dropna(inplace=True,axis=1)
# This is the numeric feature vector, as it goes to the neural net


############################################################################################
# Convert to numpy - Classification
x_columns = df.columns.drop('outcome')
x_columns = ['root_shell','num_access_files','num_failed_logins']
x = df[x_columns].values
dummies = pd.get_dummies(df['outcome']) # Classification
outcomes = dummies.columns
num_classes = len(outcomes)
y = dummies.values

#y.to_pickle('y.pkl')    #to save the dataframe, df to y.pkl

#x.to_pickle('x.pkl')    #to save the dataframe, df to y.pkl


# NEURAL NETWORK MACHINE LEARNING #############################################################

import pandas as pd
import io
import requests
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping
import joblib

import seaborn as sns
from sklearn.metrics import confusion_matrix,  accuracy_score
import matplotlib.pyplot as plt


# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Create neural net
#model = tf.keras.Sequential()
#model.add(Dense(10, input_dim=x.shape[1], kernel_initializer='normal', activation='relu'))
#model.add(Dense(20, input_dim=x.shape[1], kernel_initializer='normal', activation='relu'))
#model.add(Dense(10, input_dim=x.shape[1], kernel_initializer='normal', activation='relu'))
#model.add(Dense(1, kernel_initializer='normal'))
#model.add(Dense(y.shape[1],activation='softmax'))
#model.compile(loss='categorical_crossentropy', optimizer='adam')
#monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
#model.fit(X_train,y_train,validation_data=(X_test,y_test),callbacks=[monitor],verbose=2,epochs=19)
#model.save("model.h5")      

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import FeatureUnion, make_pipeline, Pipeline
from sklearn.tree import DecisionTreeClassifier
from torchvision import transforms
from sklearn.impute import SimpleImputer, MissingIndicator
import streamlit as st


transformer = FeatureUnion(
    transformer_list=[
        ('features', SimpleImputer(strategy='mean')),('indicators', MissingIndicator())])

clf = make_pipeline(transformer, RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1))
clf = clf.fit(X_train, y_train)
#results = clf.predict(X_test)
#score = clf.score(X_test, y_test)
#print("Shape:    ", results.shape)
#print("Score:   ", score)

# Measure accuracy
# This works
#pred = clf.predict(X_test)
#pred = np.argmax(pred,axis=1)
#y_eval = np.argmax(y_test,axis=1)
#score = metrics.accuracy_score(y_eval, pred)
#print("Validation score: {}".format(score))


# Pickel time save and load
pickle.dump(clf, open('myModel.pkl', 'wb'))

# Loading model to compare the results
#pickled_model = pickle.load(open('myModel.pkl', 'rb'))
#pred2 = pickled_model.predict(X_test)
#pred2 = np.argmax(pred2,axis=1)
#y_eval = np.argmax(y_test,axis=1)
#cf_matrix = confusion_matrix(y_eval, pred2)
#score = metrics.accuracy_score(y_eval, pred2)
#print("Accuracy:", score)


#def result():
#    if request.method == 'POST':    
#        return render_template("index.html", prediction = score)


#print(cf_matrix)
#sns.heatmap(cf_matrix, annot=True)

#sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True,  fmt='.2%', cmap='Reds')
# labels = ['True Neg','False Pos','False Neg','True Pos']
# labels = np.asarray(labels).reshape(2,2)
# sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')


    


#filename = 'finalized_model.sav' # for joblib
#save_path = './model.'
#tf.keras.model.save(save_path)
# load tensorflow model (ALWAYS HAS ISSUES)
#myModel = tf.saved_model.load(save_path)
#Model = tf.keras.models.load_model("model.h5")

#joblib.dump(myModel, filename) # SAVE THE MODEL AS PICKLE


# load the model from disk and tests score
#loaded_model = joblib.load(myModel)
#result = loaded_model.score(X_test, y_test)
#print(result)




