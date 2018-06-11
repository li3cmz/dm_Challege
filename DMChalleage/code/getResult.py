
import numpy as np
import pandas as pd
from sklearn import metrics
import lightgbm as lgb
import time
from multiprocessing import cpu_count
import warnings
from sklearn.cross_validation import train_test_split

############################################### get result ########################
test_predss = pd.read_csv('./test_preds.csv')
test_label = test_predss['label']
test_uid = test_predss['uid']
for index in range(3000):
    if test_label[index] > 0.5:
        test_label[index] = 1
    else:
        test_label[index] = 0
t = pd.concat([test_uid,pd.to_numeric(test_label)],axis=1)
t.to_csv('./test_preds_res.csv',index=False,sep=',',header=None)


############################################### see result ########################
k = pd.read_csv('./test_preds_res.csv',header=None)
print(k.shape)
print(k)