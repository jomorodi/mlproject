import pandas as pd
import numpy as np
from pycaret.classification import *
import category_encoders as ce
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pandas_profiling import ProfileReport
import pickle
df = pd.read_csv("./500_Person_Gender_Height_Weight_Index.csv")
def ordinal_encoding(df,col,mapping):
    ordinal_encoder = ce.OrdinalEncoder(cols = [col],return_df = True,mapping = [{'col':col,'mapping':mapping}])
    df_final = ordinal_encoder.fit_transform(df)
    return df_final
mapping = {"Male":0,"Female":1}
df_final = ordinal_encoding(df,"Gender",mapping)
df_final
experiment = setup(df_final,target="Index")
X = df_final.drop("Index",axis = 1)
Y = df_final["Index"]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state=27,test_size = 0.2,stratify = Y)
clf = QuadraticDiscriminantAnalysis()
clf.fit(X_train,Y_train)

