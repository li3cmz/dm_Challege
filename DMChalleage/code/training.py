from __future__ import division
import numpy as np
import pandas as pd
from sklearn import metrics
import lightgbm as lgb
import time
from multiprocessing import cpu_count
import warnings
from sklearn.cross_validation import train_test_split
warnings.filterwarnings('ignore')


# Constants define
ROOT_PATH = './'
ONLINE = 1

target = 'label'
train_len = 4999
threshold = 0.5


########################################### Helper function ###########################################


def log(info):
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' ' + str(info))

def merge_feat_count(df, df_feat, columns_groupby, new_column_name, type='int'):
    df_count = pd.DataFrame(df_feat.groupby(columns_groupby).size()).fillna(0).astype(type).reset_index()
    df_count.columns = columns_groupby + [new_column_name]
    df = df.merge(df_count, on=columns_groupby, how='left')
    return df, [new_column_name]

def merge_feat_onehot_count(df, df_feat, columns_groupby, prefix, type='int'):
    df_count = df_feat.groupby(columns_groupby).size().unstack().fillna(0).astype(type).reset_index()
    df_count.columns = [i if i == columns_groupby[0] else prefix + '_' + str(i) for i in df_count.columns]
    df = df.merge(df_count, on=columns_groupby[0], how='left')
    return df, list(np.delete(df_count.columns.values, 0))

def merge_feat_nunique(df, df_feat, columns_groupby, column, new_column_name, type='int'):
    df_nunique = pd.DataFrame(df_feat.groupby(columns_groupby)[column].nunique()).fillna(0).astype(type).reset_index()
    df_nunique.columns = columns_groupby + [new_column_name]
    df = df.merge(df_nunique, on=columns_groupby, how='left')
    return df, [new_column_name]

def merge_feat_min(df, df_feat, columns_groupby, column, new_column_name, type='float'):
    df_min = pd.DataFrame(df_feat.groupby(columns_groupby)[column].min()).fillna(0).astype(type).reset_index()
    df_min.columns = columns_groupby + [new_column_name]
    df = df.merge(df_min, on=columns_groupby, how='left')
    return df, [new_column_name]

def merge_feat_max(df, df_feat, columns_groupby, column, new_column_name, type='float'):
    df_max = pd.DataFrame(df_feat.groupby(columns_groupby)[column].max()).fillna(0).astype(type).reset_index()
    df_max.columns = columns_groupby + [new_column_name]
    df = df.merge(df_max, on=columns_groupby, how='left')
    return df, [new_column_name]

def merge_feat_mean(df, df_feat, columns_groupby, column, new_column_name, type='float'):
    df_mean = pd.DataFrame(df_feat.groupby(columns_groupby)[column].mean()).fillna(0).astype(type).reset_index()
    df_mean.columns = columns_groupby + [new_column_name]
    df = df.merge(df_mean, on=columns_groupby, how='left')
    return df, [new_column_name]

def merge_feat_std(df, df_feat, columns_groupby, column, new_column_name, type='float'):
    df_std = pd.DataFrame(df_feat.groupby(columns_groupby)[column].std()).fillna(0).astype(type).reset_index()
    df_std.columns = columns_groupby + [new_column_name]
    df = df.merge(df_std, on=columns_groupby, how='left')
    return df, [new_column_name]

def merge_feat_median(df, df_feat, columns_groupby, column, new_column_name, type='float'):
    df_median = pd.DataFrame(df_feat.groupby(columns_groupby)[column].median()).fillna(0).astype(type).reset_index()
    df_median.columns = columns_groupby + [new_column_name]
    df = df.merge(df_median, on=columns_groupby, how='left')
    return df, [new_column_name]

def merge_feat_sum(df, df_feat, columns_groupby, column, new_column_name, type='float'):
    df_sum = pd.DataFrame(df_feat.groupby(columns_groupby)[column].sum()).fillna(0).astype(type).reset_index()
    df_sum.columns = columns_groupby + [new_column_name]
    df = df.merge(df_sum, on=columns_groupby, how='left')
    return df, [new_column_name]



def eval_auc_f1(preds, dtrain):
    df = pd.DataFrame({'y_true': dtrain.get_label(), 'y_score': preds})
    df['y_pred'] = df['y_score'].apply(lambda x: 1 if x >= threshold else 0)
    auc = metrics.roc_auc_score(df.y_true, df.y_score)
    f1 = metrics.f1_score(df.y_true, df.y_pred)
    return 'feval', (auc * 0.6 + f1 * 0.4), True

def lgb_cv(train_x, train_y, params, rounds, folds):
    start = time.clock()
    log(str(train_x.columns))
    dtrain = lgb.Dataset(train_x, label=train_y)
    log('run cv: ' + 'round: ' + str(rounds))
    res = lgb.cv(params, dtrain, rounds, nfold=folds, 
                 metrics=['eval_auc_f1', 'auc'], feval=eval_auc_f1, 
                 early_stopping_rounds=200, verbose_eval=5)
    elapsed = (time.clock() - start)
    log('Time used:' + str(elapsed) + 's')
    return len(res['feval-mean']), res['feval-mean'][len(res['feval-mean']) - 1], res['auc-mean'][len(res['auc-mean']) - 1]

def lgb_train_predict(train_x, train_y, test_x, params, rounds):
    start = time.clock()
    log(str(train_x.columns))
    dtrain = lgb.Dataset(train_x, label=train_y)
    valid_sets = [dtrain]
    model = lgb.train(params, dtrain, rounds, valid_sets, feval=eval_auc_f1, verbose_eval=5)
    pred = model.predict(test_x)
    elapsed = (time.clock() - start)
    log('Time used:' + str(elapsed) + 's')
    return model, pred

def store_result(test_index, pred, threshold, name):
    result = pd.DataFrame({'uid': test_index, 'prob': pred})
    result = result.sort_values('prob', ascending=False)
    result['label'] = 0
    result.loc[result.prob > threshold, 'label'] = 1
    result.to_csv('./0606V1/' + name + '.csv', index=0, header=0, columns=['uid', 'label'])
    return result


########################################### Read data ###########################################


train = pd.read_csv(ROOT_PATH + '/uid_train.txt', header=None, sep='\t')
train.columns = ['uid', 'label']
train_voice = pd.read_csv(ROOT_PATH + '/voice_train.txt', header=None, sep='\t')
train_voice.columns = ['uid', 'opp_num', 'opp_head', 'opp_len', 'start_time', 'end_time', 'call_type', 'in_out']
train_sms = pd.read_csv(ROOT_PATH + '/sms_train.txt', header=None, sep='\t')
train_sms.columns = ['uid', 'opp_num', 'opp_head', 'opp_len', 'start_time', 'in_out']
train_wa = pd.read_csv(ROOT_PATH + '/wa_train.txt', header=None, sep='\t')
train_wa.columns = ['uid', 'wa_name', 'visit_cnt', 'visit_dura', 'up_flow', 'down_flow', 'wa_type', 'date']

test = pd.DataFrame({'uid': ['u' + str(i) for i in range(7000, 10000)]})
test_voice = pd.read_csv(ROOT_PATH + '/voice_test_b.txt', header=None, sep='\t')
test_voice.columns = ['uid', 'opp_num', 'opp_head', 'opp_len', 'start_time', 'end_time', 'call_type', 'in_out']
test_sms = pd.read_csv(ROOT_PATH + '/sms_test_b.txt', header=None, sep='\t')
test_sms.columns = ['uid', 'opp_num', 'opp_head', 'opp_len', 'start_time', 'in_out']
test_wa = pd.read_csv(ROOT_PATH + '/wa_test_b.txt', header=None, sep='\t')
test_wa.columns = ['uid', 'wa_name', 'visit_cnt', 'visit_dura', 'up_flow', 'down_flow', 'wa_type', 'date']


############################## Next I will sperate the start date of voice (date and hour) ################################

time_feature = test_voice['start_time']
time_feature = time_feature.as_matrix()
voice_start_date = []
voice_start_hour = []
print(np.size(time_feature,0))
for i in range(np.size(time_feature,0)):
    tmp = str(time_feature[i])

    voice_start_date.append(tmp[0:2])
    voice_start_hour.append(tmp[2:4])
voice_start_date = pd.DataFrame(voice_start_date)
voice_start_hour = pd.DataFrame(voice_start_hour)
voice_start_date.columns = ['voice_start_date']
voice_start_hour.columns = ['voice_start_hour']
test_voice['voice_start_date'] = voice_start_date
test_voice['voice_start_hour'] = voice_start_hour


time_feature = train_voice['start_time']
time_feature = time_feature.as_matrix()
voice_start_date = []
voice_start_hour = []
print(np.size(time_feature,0))
for i in range(np.size(time_feature,0)):
    tmp = str(time_feature[i])

    voice_start_date.append(tmp[0:2])
    voice_start_hour.append(tmp[2:4])
voice_start_date = pd.DataFrame(voice_start_date)
voice_start_hour = pd.DataFrame(voice_start_hour)
voice_start_date.columns = ['voice_start_date']
voice_start_hour.columns = ['voice_start_hour']
train_voice['voice_start_date'] = voice_start_date
train_voice['voice_start_hour'] = voice_start_hour



############################## Next I will sperate the start date of sms (date and hour) ################################

time_feature = test_sms['start_time']
time_feature = time_feature.as_matrix()
sms_start_date = []
sms_start_hour = []
print(np.size(time_feature,0))
for i in range(np.size(time_feature,0)):
    tmp = str(time_feature[i])
    #print(tmp)
    sms_start_date.append(tmp[0:2])
    sms_start_hour.append(tmp[2:4])
sms_start_date = pd.DataFrame(sms_start_date)
sms_start_hour = pd.DataFrame(sms_start_hour)
sms_start_date.columns = ['sms_start_date']
sms_start_hour.columns = ['sms_start_hour']
test_sms['sms_start_date'] = sms_start_date
test_sms['sms_start_hour'] = sms_start_hour

time_feature = train_sms['start_time']
time_feature = time_feature.as_matrix()
sms_start_date = []
sms_start_hour = []
print(np.size(time_feature,0))
for i in range(np.size(time_feature,0)):
    tmp = str(time_feature[i])
    #print(tmp)
    sms_start_date.append(tmp[0:2])
    sms_start_hour.append(tmp[2:4])
sms_start_date = pd.DataFrame(sms_start_date)
sms_start_hour = pd.DataFrame(sms_start_hour)
sms_start_date.columns = ['sms_start_date']
sms_start_hour.columns = ['sms_start_hour']
train_sms['sms_start_date'] = sms_start_date
train_sms['sms_start_hour'] = sms_start_hour


############################## Next I will sperate the end date of voice(date and hour) ################################

time_feature = test_voice['end_time']
time_feature = time_feature.as_matrix()
voice_end_date = []
voice_end_hour = []
print(np.size(time_feature,0))
for i in range(np.size(time_feature,0)):
    tmp = str(time_feature[i])
    #print(tmp)
    voice_end_date.append(tmp[0:2])
    voice_end_hour.append(tmp[2:4])
voice_end_date = pd.DataFrame(voice_end_date)
voice_end_hour = pd.DataFrame(voice_end_hour)
voice_end_date.columns = ['voice_end_date']
voice_end_hour.columns = ['voice_end_hour']
test_voice['voice_end_date'] = voice_end_date
test_voice['voice_end_hour'] = voice_end_hour

time_feature = train_voice['end_time']
time_feature = time_feature.as_matrix()
voice_end_date = []
voice_end_hour = []
print(np.size(time_feature,0))
for i in range(np.size(time_feature,0)):
    tmp = str(time_feature[i])
    #print(tmp)
    voice_end_date.append(tmp[0:2])
    voice_end_hour.append(tmp[2:4])
voice_end_date = pd.DataFrame(voice_end_date)
voice_end_hour = pd.DataFrame(voice_end_hour)
voice_end_date.columns = ['voice_end_date']
voice_end_hour.columns = ['voice_end_hour']
train_voice['voice_end_date'] = voice_end_date
train_voice['voice_end_hour'] = voice_end_hour



############################## Next I will sperate the duration of train_voice (date and hour) ################################

duraVStart = train_voice['start_time']
duraVStart = duraVStart.as_matrix()
print(type(duraVStart))
duraSec = np.zeros(np.size(duraVStart,0))

for index in range(np.size(duraVStart,0)):
    duraVStartt = str(duraVStart[index])
    if len(duraVStartt) == 8:
        ddSec =  int(duraVStartt[:2])*86400
        hhSec =  int(duraVStartt[2:4])*3600
        mmSec = int(duraVStartt[4:6])*60
        ssSec = int(duraVStartt[6:8])
    elif len(duraVStartt) == 7:
        duraVStartt = '0'+ duraVStartt
        ddSec =  int(duraVStartt[:2])*86400
        hhSec =  int(duraVStartt[2:4])*3600
        mmSec = int(duraVStartt[4:6])*60
        ssSec = int(duraVStartt[6:8])
    else:
        duraVStartt = '00'+ duraVStartt
        ddSec =  int(duraVStartt[:2])*86400
        hhSec =  int(duraVStartt[2:4])*3600
        mmSec = int(duraVStartt[4:6])*60
        ssSec = int(duraVStartt[6:8])
    duraSec[index] = ddSec+hhSec+mmSec+ssSec
print(duraSec)


duraVEnd = train_voice['end_time']
duraVEnd = duraVEnd.as_matrix()
print(type(duraVEnd))
duraSec1 = np.zeros(np.size(duraVEnd,0))

for index in range(np.size(duraVEnd,0)):
    duraVEndd = str(duraVEnd[index])
   
    if len(duraVEndd) == 8:
        duraVEndd = duraVEndd
        ddSec =  int(duraVEndd[:2])*86400
        hhSec =  int(duraVEndd[2:4])*3600
        mmSec = int(duraVEndd[4:6])*60
        ssSec = int(duraVEndd[6:8])
    elif len(duraVEndd) == 7:
        duraVEndd = '0' + duraVEndd
        ddSec =  int(duraVEndd[:2])*86400
        hhSec =  int(duraVEndd[2:4])*3600
        mmSec = int(duraVEndd[4:6])*60
        ssSec = int(duraVEndd[6:8])
    else:
        duraVEndd = '00' + duraVEndd
        ddSec =  int(duraVEndd[:2])*86400
        hhSec =  int(duraVEndd[2:4])*3600
        mmSec = int(duraVEndd[4:6])*60
        ssSec = int(duraVEndd[6:8])
    duraSec1[index] = ddSec+hhSec+mmSec+ssSec
print(duraSec1)

duration = np.zeros(np.size(duraSec1,0))
for i in range(np.size(duraSec1,0)):
    if duraSec1[i] >= duraSec[i]:
        duration[i] = duraSec1[i] - duraSec[i]
    else:
        print('1..',duraSec1[i],'2..',duraSec[i])
        duration[i] = duraSec1[i]+86400-duraSec[i]


maxx = duration.max()
minn = duration.min()
meann = duration.mean()
print('maxx..',maxx,'minn..',minn, 'mean..',meann)
duration = (duration-minn)/(maxx-minn)
print(duration)
duration = pd.DataFrame(duration)
train_voice['duration'] = duration



############################## Next I will sperate the duration of test_voice (date and hour) ################################

duraVStart = test_voice['start_time']
duraVStart = duraVStart.as_matrix()
print(type(duraVStart))
duraSec = np.zeros(np.size(duraVStart,0))

for index in range(np.size(duraVStart,0)):
    duraVStartt = str(duraVStart[index])
    if len(duraVStartt) == 8:
        ddSec =  int(duraVStartt[:2])*86400
        hhSec =  int(duraVStartt[2:4])*3600
        mmSec = int(duraVStartt[4:6])*60
        ssSec = int(duraVStartt[6:8])
    elif len(duraVStartt) == 7:
        duraVStartt = '0'+ duraVStartt
        ddSec =  int(duraVStartt[:2])*86400
        hhSec =  int(duraVStartt[2:4])*3600
        mmSec = int(duraVStartt[4:6])*60
        ssSec = int(duraVStartt[6:8])
    else:
        duraVStartt = '00'+ duraVStartt
        ddSec =  int(duraVStartt[:2])*86400
        hhSec =  int(duraVStartt[2:4])*3600
        mmSec = int(duraVStartt[4:6])*60
        ssSec = int(duraVStartt[6:8])
    duraSec[index] = ddSec+hhSec+mmSec+ssSec
print(duraSec)


duraVEnd = test_voice['end_time']
duraVEnd = duraVEnd.as_matrix()
print(type(duraVEnd))
duraSec1 = np.zeros(np.size(duraVEnd,0))

for index in range(np.size(duraVEnd,0)):
    duraVEndd = str(duraVEnd[index])
   
    if len(duraVEndd) == 8:
        duraVEndd = duraVEndd
        ddSec =  int(duraVEndd[:2])*86400
        hhSec =  int(duraVEndd[2:4])*3600
        mmSec = int(duraVEndd[4:6])*60
        ssSec = int(duraVEndd[6:8])
    elif len(duraVEndd) == 7:
        duraVEndd = '0' + duraVEndd
        ddSec =  int(duraVEndd[:2])*86400
        hhSec =  int(duraVEndd[2:4])*3600
        mmSec = int(duraVEndd[4:6])*60
        ssSec = int(duraVEndd[6:8])
    else:
        duraVEndd = '00' + duraVEndd
        ddSec =  int(duraVEndd[:2])*86400
        hhSec =  int(duraVEndd[2:4])*3600
        mmSec = int(duraVEndd[4:6])*60
        ssSec = int(duraVEndd[6:8])
    duraSec1[index] = ddSec+hhSec+mmSec+ssSec
print(duraSec1)

duration = np.zeros(np.size(duraSec1,0))
for i in range(np.size(duraSec1,0)):
    if duraSec1[i] >= duraSec[i]:
        duration[i] = duraSec1[i] - duraSec[i]
    else:
        print('1..',duraSec1[i],'2..',duraSec[i])
        duration[i] = duraSec1[i]+86400-duraSec[i]


maxx = duration.max()
minn = duration.min()
meann = duration.mean()
print('maxx..',maxx,'minn..',minn, 'mean..',meann)
duration = (duration-minn)/(maxx-minn)
print(duration)

duration = pd.DataFrame(duration)
test_voice['duration'] = duration


############################## Normalize visist_cnt and visit_dura ################################

test_visit_cnt_normal = test_wa['visit_cnt']
maxx = test_visit_cnt_normal.max()
minn = test_visit_cnt_normal.min()
meann = test_visit_cnt_normal.mean()

print('maxx..',maxx,'minn..',minn, 'mean..',meann)
test_visit_cnt_normal = (test_visit_cnt_normal-minn)/(maxx-minn)
test_visit_cnt_normal = test_visit_cnt_normal*10000
test_wa['visit_cnt'] = test_visit_cnt_normal

train_visit_cnt_normal = train_wa['visit_cnt']
maxx = train_visit_cnt_normal.max()
minn = train_visit_cnt_normal.min()
meann = train_visit_cnt_normal.mean()
print('maxx..',maxx,'minn..',minn, 'mean..',meann)

train_visit_cnt_normal = (train_visit_cnt_normal-minn)/(maxx-minn)
trainvisit_cnt_normal = train_visit_cnt_normal*10000
train_wa['visit_cnt'] = train_visit_cnt_normal

train_visit_dura_normal = train_wa['visit_dura']
maxx = train_visit_dura_normal.max()
minn = train_visit_dura_normal.min()
meann = train_visit_dura_normal.mean()
print('maxx..',maxx,'minn..',minn, 'mean..',meann)
train_visit_dura_normal = (train_visit_dura_normal-minn)/(maxx-minn)
train_visit_dura_normal = train_visit_dura_normal*10000
train_wa['visit_dura'] = train_visit_dura_normal

test_visit_dura_normal = test_wa['visit_dura']
maxx = test_visit_dura_normal.max()
minn = test_visit_dura_normal.min()
meann = test_visit_dura_normal.mean()
print('maxx..',maxx,'minn..',minn, 'mean..',meann)
test_visit_dura_normal = (test_visit_dura_normal-minn)/(maxx-minn)
test_visit_dura_normal = test_visit_dura_normal*10000
test_wa['visit_dura'] = test_visit_dura_normal


#################################################### Feature engineer ###########################################

df = pd.concat([train, test]).reset_index(drop=True)
df_voice = pd.concat([train_voice, test_voice]).reset_index(drop=True)
df_sms = pd.concat([train_sms, test_sms]).reset_index(drop=True)
df_wa = pd.concat([train_wa, test_wa]).reset_index(drop=True)
predictors = []

#-------------------------------------------- To get wa_name_number freature ---------------------------------------------
wa_name = df_wa['wa_name'].fillna(0)
wa_name = np.array(wa_name)
Size = np.size(wa_name,0)
wa_name_set = pd.read_csv('./wa_name_set_strr.csv', header=None, sep='\t')
wa_name_set = np.array(wa_name_set)
wa_name_set = np.reshape(wa_name_set,(15160,))
print(wa_name_set[0])

wa_name_set = np.array(wa_name_set).tolist()
#print(wa_name_set.shape)

wa_name_number = np.zeros(Size)
for i in range(Size):
    cur_wa_name = wa_name[i]
    print(cur_wa_name)
    if(cur_wa_name!=0 and cur_wa_name!='00934'):
        wa_name_number[i] = wa_name_set.index(cur_wa_name)
    elif cur_wa_name == '00934':
        wa_name_number[i] = wa_name_set.index('00934')
    else:
        wa_name_number[i] = 15161

df_wa['wa_name_number'] = wa_name_number

#------------------------------------------------------------- To get uid --------------------------------------------

df, predictors_tmp = merge_feat_count(df, df_voice, ['uid'], 'count_gb_uid_in_voice'); predictors += predictors_tmp
df, predictors_tmp = merge_feat_count(df, df_sms, ['uid'], 'count_gb_uid_in_sms'); predictors += predictors_tmp
df, predictors_tmp = merge_feat_count(df, df_wa, ['uid'], 'count_gb_uid_in_wa'); predictors += predictors_tmp

#---------------------------------------------------- To get one-hot feature ------------------------------------------

df, predictors_tmp = merge_feat_onehot_count(df, df_voice, ['uid', 'opp_len'], 'voice_opp_len'); predictors += predictors_tmp
df, predictors_tmp = merge_feat_onehot_count(df, df_voice, ['uid', 'call_type'], 'voice_call_type'); predictors += predictors_tmp
df, predictors_tmp = merge_feat_onehot_count(df, df_voice, ['uid', 'in_out'], 'voice_in_out_'); predictors += predictors_tmp
df, predictors_tmp = merge_feat_onehot_count(df, df_sms, ['uid', 'opp_len'], 'sms_opp_len'); predictors += predictors_tmp
df, predictors_tmp = merge_feat_onehot_count(df, df_sms, ['uid', 'in_out'], 'sms_in_out'); predictors += predictors_tmp
df, predictors_tmp = merge_feat_onehot_count(df, df_wa, ['uid', 'wa_type'], 'wa_type'); predictors += predictors_tmp
df, predictors_tmp = merge_feat_onehot_count(df, df_wa, ['uid', 'date'], 'wa_date'); predictors += predictors_tmp
df, predictors_tmp = merge_feat_onehot_count(df, df_voice, ['uid', 'voice_start_date'], 'voice_start_date'); predictors += predictors_tmp
df, predictors_tmp = merge_feat_onehot_count(df, df_voice, ['uid', 'voice_start_hour'], 'voice_start_hour'); predictors += predictors_tmp
df, predictors_tmp = merge_feat_onehot_count(df, df_sms, ['uid', 'sms_start_date'], 'sms_start_date'); predictors += predictors_tmp
df, predictors_tmp = merge_feat_onehot_count(df, df_sms, ['uid', 'sms_start_hour'], 'sms_start_hour'); predictors += predictors_tmp
df, predictors_tmp = merge_feat_onehot_count(df, df_voice, ['uid', 'opp_head'], 'voice_opp_head_onehot'); predictors += predictors_tmp
df, predictors_tmp = merge_feat_onehot_count(df, df_sms, ['uid', 'opp_head'], 'sms_opp_head_onehot'); predictors += predictors_tmp
df, predictors_tmp = merge_feat_onehot_count(df, df_wa, ['uid', 'wa_name_number'], 'wa_name_onehot'); predictors += predictors_tmp

#---------------------------------------------------- To count nunique feature -----------------------------------------

df, predictors_tmp = merge_feat_nunique(df, df_voice, ['uid'], 'opp_num', 'nunique_oppNum_gb_uid_in_voice'); predictors += predictors_tmp
df, predictors_tmp = merge_feat_nunique(df, df_voice, ['uid'], 'opp_head', 'nunique_oppHead_gb_uid_in_voice'); predictors += predictors_tmp
df, predictors_tmp = merge_feat_nunique(df, df_sms, ['uid'], 'opp_num', 'nunique_oppNum_gb_uid_in_sms'); predictors += predictors_tmp
df, predictors_tmp = merge_feat_nunique(df, df_sms, ['uid'], 'opp_head', 'nunique_oppHead_gb_uid_in_sms'); predictors += predictors_tmp
df, predictors_tmp = merge_feat_nunique(df, df_wa, ['uid'], 'wa_name', 'nunique_waName_gb_uid_in_wa'); predictors += predictors_tmp


#---------------------------------------------------- To do mean\max\min\median\sum\std ---------------------------------

col_list = ['visit_cnt', 'visit_dura', 'up_flow', 'down_flow']
for i in col_list:
    df, predictors_tmp = merge_feat_min(df, df_wa, ['uid'], i, 'min_%s_gb_uid_in_wa' % i); predictors += predictors_tmp
    df, predictors_tmp = merge_feat_max(df, df_wa, ['uid'], i, 'max_%s_gb_uid_in_wa' % i); predictors += predictors_tmp
    df, predictors_tmp = merge_feat_mean(df, df_wa, ['uid'], i, 'mean_%s_gb_uid_in_wa' % i); predictors += predictors_tmp
    df, predictors_tmp = merge_feat_median(df, df_wa, ['uid'], i, 'median_%s_gb_uid_in_wa' % i); predictors += predictors_tmp
    df, predictors_tmp = merge_feat_std(df, df_wa, ['uid'], i, 'std_%s_gb_uid_in_wa' % i); predictors += predictors_tmp
    df, predictors_tmp = merge_feat_sum(df, df_wa, ['uid'], i, 'sum_%s_gb_uid_in_wa' % i); predictors += predictors_tmp


col_list1 = ['opp_len']
for i in col_list1:
    df, predictors_tmp = merge_feat_min(df, df_voice, ['uid'], i, 'min_%s_gb_uid_in_voice' % i); predictors += predictors_tmp
    df, predictors_tmp = merge_feat_max(df, df_voice, ['uid'], i, 'max_%s_gb_uid_in_voice' % i); predictors += predictors_tmp
    df, predictors_tmp = merge_feat_mean(df, df_voice, ['uid'], i, 'mean_%s_gb_uid_in_voice' % i); predictors += predictors_tmp
    df, predictors_tmp = merge_feat_median(df, df_voice, ['uid'], i, 'median_%s_gb_uid_in_voice' % i); predictors += predictors_tmp
    df, predictors_tmp = merge_feat_std(df, df_voice, ['uid'], i, 'std_%s_gb_uid_in_voice' % i); predictors += predictors_tmp
    df, predictors_tmp = merge_feat_sum(df, df_voice, ['uid'], i, 'sum_%s_gb_uid_in_voice' % i); predictors += predictors_tmp
    
col_list2 = ['opp_len']
for i in col_list2:
    df, predictors_tmp = merge_feat_min(df, df_sms, ['uid'], i, 'min_%s_gb_uid_in_sms' % i); predictors += predictors_tmp
    df, predictors_tmp = merge_feat_max(df, df_sms, ['uid'], i, 'max_%s_gb_uid_in_sms' % i); predictors += predictors_tmp
    df, predictors_tmp = merge_feat_mean(df, df_sms, ['uid'], i, 'mean_%s_gb_uid_in_sms' % i); predictors += predictors_tmp
    df, predictors_tmp = merge_feat_median(df, df_sms, ['uid'], i, 'median_%s_gb_uid_in_sms' % i); predictors += predictors_tmp
    df, predictors_tmp = merge_feat_std(df, df_sms, ['uid'], i, 'std_%s_gb_uid_in_sms' % i); predictors += predictors_tmp
    df, predictors_tmp = merge_feat_sum(df, df_sms, ['uid'], i, 'sum_%s_gb_uid_in_sms' % i); predictors += predictors_tmp
    
col_list3 = ['duration']
for i in col_list3:
    df, predictors_tmp = merge_feat_min(df, df_voice, ['uid'], i, 'min_%s_gb_uid_in_voice1' % i); predictors += predictors_tmp
    df, predictors_tmp = merge_feat_max(df, df_voice, ['uid'], i, 'max_%s_gb_uid_in_voice1' % i); predictors += predictors_tmp
    df, predictors_tmp = merge_feat_mean(df, df_voice, ['uid'], i, 'mean_%s_gb_uid_in_voice1' % i); predictors += predictors_tmp
    df, predictors_tmp = merge_feat_median(df, df_voice, ['uid'], i, 'median_%s_gb_uid_in_voice1' % i); predictors += predictors_tmp
    df, predictors_tmp = merge_feat_std(df, df_voice, ['uid'], i, 'std_%s_gb_uid_in_voice1' % i); predictors += predictors_tmp
    df, predictors_tmp = merge_feat_sum(df, df_voice, ['uid'], i, 'sum_%s_gb_uid_in_voice1' % i); predictors += predictors_tmp



df_voice['nunique_oppNum_gb_uid_in_voice'] = df['nunique_oppNum_gb_uid_in_voice']
col_list4 = ['nunique_oppNum_gb_uid_in_voice']
for i in col_list4:
    df, predictors_tmp = merge_feat_min(df, df_voice, ['uid'], i, 'min_%s_gb_uid_in_voice3' % i); predictors += predictors_tmp
    df, predictors_tmp = merge_feat_max(df, df_voice, ['uid'], i, 'max_%s_gb_uid_in_voice3' % i); predictors += predictors_tmp
    df, predictors_tmp = merge_feat_mean(df, df_voice, ['uid'], i, 'mean_%s_gb_uid_in_voice3' % i); predictors += predictors_tmp
    df, predictors_tmp = merge_feat_median(df, df_voice, ['uid'], i, 'median_%s_gb_uid_in_voice3' % i); predictors += predictors_tmp
    df, predictors_tmp = merge_feat_std(df, df_voice, ['uid'], i, 'std_%s_gb_uid_in_voice3' % i); predictors += predictors_tmp
    df, predictors_tmp = merge_feat_sum(df, df_voice, ['uid'], i, 'sum_%s_gb_uid_in_voice3' % i); predictors += predictors_tmp

df_sms['nunique_oppNum_gb_uid_in_sms'] = df['nunique_oppNum_gb_uid_in_sms']
col_list5 = ['nunique_oppNum_gb_uid_in_sms']
for i in col_list5:
    df, predictors_tmp = merge_feat_min(df, df_sms, ['uid'], i, 'min_%s_gb_uid_in_sms4' % i); predictors += predictors_tmp
    df, predictors_tmp = merge_feat_max(df, df_sms, ['uid'], i, 'max_%s_gb_uid_in_sms4' % i); predictors += predictors_tmp
    df, predictors_tmp = merge_feat_mean(df, df_sms, ['uid'], i, 'mean_%s_gb_uid_in_sms4' % i); predictors += predictors_tmp
    df, predictors_tmp = merge_feat_median(df, df_sms, ['uid'], i, 'median_%s_gb_uid_in_sms4' % i); predictors += predictors_tmp
    df, predictors_tmp = merge_feat_std(df, df_sms, ['uid'], i, 'std_%s_gb_uid_in_sms4' % i); predictors += predictors_tmp
    df, predictors_tmp = merge_feat_sum(df, df_sms, ['uid'], i, 'sum_%s_gb_uid_in_sms4' % i); predictors += predictors_tmp

############################################## Separate Data ############################################

X = df.loc[:(train_len - 1), predictors]
y = df.loc[:(train_len - 1), target]
X_test = df.loc[train_len:, predictors]


############################################## To split val and train ###################################
seed = 7
test_size = 0.1
X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=test_size, random_state=seed)


########################################### LightGBM Model #############################################
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss', 'auc'},
    'num_leaves': 5,
    'max_depth': 8,
    'min_data_in_leaf': 255,
    'learning_rate': 0.1,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.95,
    'bagging_freq': 9,
    'lambda_l1': 1,  
    'lambda_l2': 0.00001,  # 
    'min_gain_to_split': 0.3,
    'verbose': 6,
    'is_unbalance': True
}
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val,reference=train_data)

########################################### training #############################################
print('Start training...')
gbm = lgb.train(params,
                train_data,
                num_boost_round=4000,
                valid_sets=val_data,
                early_stopping_rounds=3000)


########################################### predicting #############################################
print ("Begin predicting...")
test_preds = gbm.predict(X_test, num_iteration=gbm.best_iteration)  # 
val_preds = gbm.predict(X_val,  num_iteration=gbm.best_iteration)
pd.DataFrame(test_preds).to_csv('./test_preds.csv',header=False,index=False)


########################################### AUC #############################################
auc = metrics.roc_auc_score(y_val,val_preds)
print("AUC: ",auc)
for index in range(len(val_preds)):
    if val_preds[index] > 0.5:
        val_preds[index] = 1
    else:
        val_preds[index] = 0

########################################### F-Score #############################################
F1= metrics.f1_score(y_val, val_preds)
print("F1..",F1)

########################################### evaluate #############################################
score = 0.6*auc + 0.4*F1
print(score)

########################################### Data Dim #############################################
print('df.shape####################',df.shape)