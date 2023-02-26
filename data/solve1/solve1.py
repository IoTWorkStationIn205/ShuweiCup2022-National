#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import klib as kl
import os
import warnings

os.environ['KERAS_BACKEND']='tensorflow'
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Kaiti']
plt.rcParams['axes.unicode_minus'] = False
PIC_PATH = "../../models/image/image1"
DATA_PATH = '../../data'
RESULT_PATH = '../../data/summary'
MODEL_PATH = '../../models/model1'


# In[2]:


import pathlib2 as pl2

def creat_dir():
    pic_path = pl2.Path(PIC_PATH)
    if not os.path.exists(PIC_PATH):
        pic_path.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(RESULT_PATH):
        os.mkdir(RESULT_PATH)

creat_dir()


# In[3]:


figure_count = 0

def create_figure(figure_name):
    global figure_count
    figure_count += 1
    return PIC_PATH + f'/figure{figure_count}_{figure_name}.png'


# In[4]:


data = pd.read_csv(DATA_PATH + '/ADNIMERGE_New.csv')
data


# In[5]:


dtypes = data.dtypes
object_type = dtypes[data.dtypes=='object'].index
data[object_type]


# In[6]:


import re

# 数据格式校验

def pid_ack(pt_id):
    re_str = r'\d{3}_[A-Za-z]{1}_\d{4}'
    res = re.findall(re_str, pt_id)
    if len(res) > 0:
        return res[0]
    return np.nan

def time_ack(time):
    re_str = r'\d{4}.\d{1,2}.\d{1,2}'
    time = str(time)
    res = re.findall(re_str, time)
    if len(res) > 0:
        return res[0]
    return np.nan

def num_ack(num):
    num = str(num)
    re_str = '[-]?\d+[\.]?[\d+]?$'
    res = re.findall(re_str, num)
    if len(res) > 0:
        return float(res[0])
    return np.nan

def object_act(value):
    if value == '<NA>' or value == 'NA':
        return np.nan
    return value

data['PTID'] = data['PTID'].apply(pid_ack)
data['EXAMDATE'] = pd.to_datetime(data['EXAMDATE'].apply(time_ack), infer_datetime_format=True)
data[['ABETA', 'TAU', 'PTAU', 'ABETA_bl', 'TAU_bl', 'PTAU_bl']] = data[[
                                          'ABETA', 'TAU', 'PTAU', 'ABETA_bl', 'TAU_bl', 'PTAU_bl'
                                        ]].applymap(num_ack).astype(np.float64)

data = data.applymap(object_act)
data


# In[7]:


data['VISCODE'] = data['VISCODE'].str.replace('bl', '0')
data['VISCODE'] = data['VISCODE'].str.replace('m', '')
data['VISCODE'] = data['VISCODE'].astype('int')


# In[8]:


print(data.columns.tolist())


# In[9]:


import toad

description = toad.detect(data)
description.T.to_excel(RESULT_PATH + '/solve1_description.xlsx')
description


# In[10]:


kl.missingval_plot(data, figsize=(18,18))
plt.savefig(create_figure('missing_plot'), dpi=1200)


# In[11]:


missing_rate = 1 - (data.count() / data.shape[0])
missing_rate = missing_rate[missing_rate > 0]
missing_rate


# In[12]:


missing_rate.sort_values()


# In[13]:


import missingno as mns

mns.heatmap(data, labels=False)
plt.savefig(create_figure('missing_heatmap'), dpi=1200)


# In[14]:


for illness in data['DX'].unique():
    _ = data[data['DX'] == illness]
    print(illness)
    print(_.isna().any())
    kl.missingval_plot(_)
    plt.savefig(create_figure(f'missing_value_{illness}'), dpi=1200)


# In[15]:


mns.dendrogram(data)
plt.savefig(create_figure('missing_cluster'), dpi=1200)


# In[16]:


description


# In[17]:


data.hist(figsize=(24, 24))
plt.savefig(create_figure('data_hist'))


# In[18]:


data.drop(missing_rate[missing_rate>.7].index, inplace=True, axis=1)
data


# In[19]:


plt.figure(figsize=(16, 16))
plt.pie(data['COLPROT'].value_counts(), labels=data['COLPROT'].value_counts().index, autopct='%.2f%%',
        shadow=True)
plt.savefig(create_figure('colprot_distribu'))


# In[20]:


missing_group = pd.DataFrame()

for weekly in data['COLPROT'].unique():
    _ = data[data['COLPROT'] == weekly]
    miss = 1 - (_.count() / _.shape[0])
    missing_group[weekly] = miss

missing_group


# In[21]:


missing_group = missing_group[missing_group > 0].dropna(axis=0, how='all')
missing_group.fillna(0., inplace=True)
missing_group


# In[22]:


missing_group.to_excel(RESULT_PATH + '/solve1_missing_group.xlsx')


# In[23]:


missing_group2 = pd.DataFrame()

for weekly in data['ORIGPROT'].unique():
    _ = data[data['ORIGPROT'] == weekly]
    miss = 1 - (_.count() / _.shape[0])
    missing_group2[weekly] = miss

missing_group2 = missing_group2[missing_group2 > 0].dropna(axis=0, how='all')
missing_group2.fillna(0., inplace=True)
missing_group2


# In[24]:


differ = missing_group.T.diff().abs()
differ[differ>.2].dropna(axis=1, how='all', inplace=True)
differ


# In[25]:


has_ones = missing_group[missing_group==1].dropna(axis=0, how='all')
has_ones


# In[26]:


differ_column = [col for col in has_ones.index if col in differ.columns]
differ_column


# In[27]:


missing_group_differ = missing_group.T[differ_column].T
missing_group_differ


# In[28]:


missing_rate = 1 - (data.count() / data.shape[0])
missing_rate = missing_rate[missing_rate > 0]
missing_rate


# In[29]:


missing_rate[missing_rate<.01]


# In[30]:


missing_data = data.copy()
for i in missing_rate[missing_rate<.01].index:
    missing_data = missing_data[missing_data[i].notna()]

missing_data


# In[31]:


not_nan_col = [col for col in data.columns if col not in missing_rate.index]
missing_data[not_nan_col]


# In[32]:


numeric = missing_data[missing_data.dtypes[missing_data.dtypes == 'float64'].index]
numeric


# In[33]:


kl.corr_plot(numeric, annot=False, figsize=(18, 18))
plt.savefig(create_figure('numeric_pearson'), dpi=1200)


# In[34]:


corr = numeric.corr()

plt.figure(figsize=(16, 16))
mask = np.zeros_like(corr[corr>=.7],dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.set_style(style="white")
sns.heatmap(corr[corr>=.7],annot=False,mask=mask,cbar=True, linewidths=.5)
plt.title('Strong Correlation Feature')
plt.savefig(create_figure('Strong Correlation'), dpi=1200, bbox_inches = 'tight')


# In[35]:


target_na = missing_data[missing_data['DX'].isna()]
target_na


# In[36]:


plt.figure(figsize=(12, 12))
plt.pie(target_na['COLPROT'].value_counts(), autopct='%.2f%%', shadow=True,
        labels=target_na['COLPROT'].value_counts().index)
plt.savefig(create_figure('target_nan_distribute'))


# In[37]:


target_na[target_na['DX_bl'].notna()]


# In[38]:


target_na['DX'] = target_na['DX_bl']
target_na['DX'] = target_na['DX'].str.replace('LMCI', 'MCI')
target_na['DX'] = target_na['DX'].str.replace('EMCI', 'MCI')
target_na['DX'] = target_na['DX'].str.replace('SMC', 'MCI')
target_na['DX'] = target_na['DX'].str.replace('AD', 'Dementia')
counterpart = missing_data.copy()
counterpart.loc[target_na.index, 'DX'] = target_na['DX']
counterpart


# In[39]:


print(counterpart['DX'].isna().sum())
print(counterpart['DX'].unique())


# In[40]:


categorical = counterpart.dtypes[missing_data.dtypes == 'object'].index.tolist()
categorical.extend(counterpart.dtypes[missing_data.dtypes == 'int64'].index.tolist())
categorical = counterpart[categorical]
categorical


# In[41]:


science_dict = {
    'ADNI1': 1,
    'ADNI2': 2,
    'ADNI3': 4,
    'ADNIGO': 3
}

counterpart['PTGENDER'] = counterpart['PTGENDER'].map(
    {
        'Male': 0,
        'Female': 1
    }
)
counterpart['PTETHCAT'] = counterpart['PTETHCAT'].map(
    {
        'Not Hisp/Latino': 0,
        'Hisp/Latino': 1
    }
)

def get_number(value):
    value = str(value)
    re_str = '[0-9]+[\.]?[\d+]*'
    res = re.findall(re_str, value)
    if len(res) > 0:
        return float(res[0])
    return np.nan

counterpart['FLDSTRENG'] = counterpart['FLDSTRENG'].apply(get_number).astype('float')
counterpart['FLDSTRENG_bl'] = counterpart['FLDSTRENG_bl'].apply(get_number).astype('float')
counterpart['FSVERSION'] = counterpart['FSVERSION'].apply(get_number).astype('float')
counterpart['FSVERSION_bl'] = counterpart['FSVERSION_bl'].apply(get_number).astype('float')
counterpart['COLPROT'] = counterpart['COLPROT'].map(science_dict).astype('float')
counterpart['ORIGPROT'] = counterpart['ORIGPROT'].map(science_dict).astype('float')


# In[42]:


counterpart.dtypes


# In[43]:


categorical2 = counterpart.dtypes[counterpart.dtypes == 'object'].index.tolist()
counterpart[categorical2]


# In[44]:


counterpart['EXAMDATE_bl'] = pd.to_datetime(counterpart['EXAMDATE_bl'], infer_datetime_format=True)
categorical2 = counterpart.dtypes[counterpart.dtypes == 'object'].index.tolist()
counterpart[categorical2]


# In[45]:


counterpart['update_stamp'] = counterpart['update_stamp'].str.split(':')
counterpart['update_stamp'] = [np.float(i[0]) * 60 + np.float(i[1])
                               for i in counterpart['update_stamp'].tolist()]
counterpart[categorical2]


# In[46]:


counterpart['DX'].isna().sum()


# 

# In[47]:


missing_rate = 1 - (counterpart.count() / counterpart.shape[0])
missing_rate = missing_rate[missing_rate > 0]
missing_rate


# In[48]:


counterpart = counterpart[counterpart['PTETHCAT'].notna()]
counterpart


# In[49]:


counterpart['RAVLT_forgetting'] = np.where(counterpart['RAVLT_perc_forgetting_bl']<0, 0,
                                               counterpart['RAVLT_perc_forgetting_bl'])
counterpart['RAVLT_perc_forgetting'] = np.where(counterpart['RAVLT_perc_forgetting']<0, 0,
                                                    counterpart['RAVLT_perc_forgetting'])
counterpart


# In[50]:


missing_rate = 1 - (counterpart.count() / counterpart.shape[0])
missing_rate = missing_rate[missing_rate > 0].sort_values()
missing_rate


# In[51]:


from sklearn.preprocessing import LabelEncoder
from joblib import dump



def create_model(model, model_name):
    if not os.path.exists(MODEL_PATH):
        os.mkdir(MODEL_PATH)
    dump(model, MODEL_PATH + model_name)

def encode(column):
    lb = LabelEncoder()
    code = lb.fit_transform(counterpart_2[column].dropna())
    counterpart_2.loc[counterpart_2[counterpart_2[column].notna()].index, column] = code
    create_model(lb, f'/{column}_lb.model')

counterpart_2 = counterpart.copy()
categorical2.remove('PTID')
categorical2.remove('update_stamp')

for col in categorical2:
    encode(col)


# In[52]:


counterpart_2['DX'].unique()


# In[53]:


counterpart['DX'].unique()


# In[54]:


data[categorical2].isna().any()


# In[55]:


from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import train_test_split


corr_missing = [col for col in missing_rate.index if col in corr.index]
feature = [i for i in data.columns if i not in missing_rate.index]
feature.remove('PTID')
feature.remove('EXAMDATE')
feature.remove('EXAMDATE_bl')
acc = pd.DataFrame(index=corr_missing, columns=['accuracy', 'score'])

score = []
f1 = []

for col in corr_missing:
    lr = LinearRegression()
    if counterpart_2[col].nunique() < 100:
        dt = LogisticRegression()
    base = corr[corr>.7][col].dropna()
    temp = counterpart_2[counterpart_2[col].notna()]
    temp = temp[base.index].dropna()
    base = base.index.tolist()
    base.remove(col)
    if len(base) == 0:
        print(col)
        score.append(0)
        f1.append(0)
        continue
    X = temp[base]
    y = temp[col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)
    try:
        lr.fit(X_train, y_train)
    except ValueError:
        y = y.astype(np.str)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)
    lr.fit(X_train, y_train)
    score.append(lr.score(X_test, y_test))
    predict = lr.predict(X_test)
    if type(lr) is LinearRegression:
        f1.append(r2_score(y_test, predict))
        continue
    f1.append(f1_score(y_test, predict, average='weighted'))

acc['accuracy'] = score
acc['score'] = f1
acc


# In[56]:


pre_pre = acc[acc > .7].dropna().sort_values(ascending=False, by='accuracy')
pre_pre


# In[57]:


cols = []
pre_acc = pd.DataFrame(index=corr_missing, columns=['accuracy', 'score'])

score = []
f1 = []
for col in pre_pre.index:

    lr = LinearRegression()

    if counterpart_2[col].nunique() < 100:
        lr = LogisticRegression()

    base = corr[corr>.7][col].dropna()
    temp = counterpart_2[counterpart_2[col].notna()]
    nan  = counterpart_2[counterpart_2[col].isna()]
    temp = temp[base.index].dropna()
    base = base.index.tolist()
    base.remove(col)

    X = temp[base]
    y = temp[col]
    X_ = nan[base]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)
    try:
        lr.fit(X_train, y_train)
    except ValueError:
        y_train = y_train.astype(np.str)
        lr.fit(X_train, y_train)
    try:
        y_pre = lr.predict(X_)
        counterpart.loc[nan.index, col] = y_pre
        counterpart[col] = counterpart[col].astype('float')
    except ValueError as e:
        print(e)


# In[58]:


missing_rate_ = 1 - (counterpart_2.count() / counterpart_2.shape[0])
missing_rate_ = missing_rate_[missing_rate_ > 0].sort_values()
missing_rate_


# In[58]:





# In[59]:


from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import train_test_split

feature = [i for i in data.columns if i not in missing_rate.index]
feature.remove('PTID')
feature.remove('EXAMDATE')
feature.remove('EXAMDATE_bl')
acc = pd.DataFrame(index=missing_rate.index, columns=['accuracy', 'score'])

score = []
f1 = []

for col in missing_rate.index:
    dt = DecisionTreeRegressor(random_state=42)
    if counterpart_2[col].nunique() < 100:
        dt = DecisionTreeClassifier(random_state=42)
    temp = counterpart_2[counterpart_2[col].notna()]
    X = temp[[i for i in feature if i != col]]
    y = temp[col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)
    try:
        dt.fit(X_train, y_train)
    except ValueError:
        y = y.astype(np.str)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)
    dt.fit(X_train, y_train)
    score.append(dt.score(X_test, y_test))
    predict = dt.predict(X_test)
    if type(dt) is DecisionTreeRegressor:
        f1.append(r2_score(y_test, predict))
        continue
    f1.append(f1_score(y_test, predict, average='weighted'))

acc['accuracy'] = score
acc['score'] = f1
acc


# 

# In[60]:


acc_pre = acc[acc > .7].dropna().sort_values(by='accuracy', ascending=False)
acc_pre


# In[61]:


cols = []


for col in acc_pre.index:
    missing = (counterpart_2.shape[0] - counterpart_2.count()) / counterpart_2.shape[0]
    missing = missing[missing > 0.]
    dt = DecisionTreeClassifier(random_state=42)

    if counterpart_2[col].nunique() > 100:
        dt = DecisionTreeRegressor(random_state=42)

    temp = counterpart_2[counterpart_2[col].notna()]
    nan = counterpart_2[counterpart_2[col].isna()]

    X = temp[[i for i in temp.columns if i not in missing.index and i not in [col, 'PTID', 'EXAMDATE',
                                                                              'EXAMDATE_bl']]]
    y = temp[col]
    X_ = nan[[i for i in temp.columns if i not in missing.index and i not in [col, 'PTID', 'EXAMDATE',
                                                                              'EXAMDATE_bl']]]
    cols.append(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

    try:
        dt.fit(X_train, y_train)
    except ValueError:
        y_train = y_train.astype(np.str)
        y_test  = y_test.astype(np.str)
        dt.fit(X_train, y_train)

    print(col, end='\t')
    print(dt.score(X_test, y_test))
    try:
        y_pre = dt.predict(X_)
        print(counterpart[col].isnull().sum(), end='\t')
        counterpart.loc[X_.index, col] = y_pre
        counterpart_2.loc[X_.index, col] = y_pre
        print(counterpart[col].isnull().sum())
    except ValueError as e:
        print(e)


# In[62]:


missing_rate = 1 - (counterpart_2.count() / counterpart_2.shape[0])
missing_rate = missing_rate[missing_rate > 0]
missing_rate


# In[63]:


acc.sort_values(by='accuracy', ascending=True)


# In[64]:


bl_col = [col for col in counterpart_2.columns if '_bl' in col and counterpart_2[col].dtypes
          in ['int64', 'float64']]

depart = counterpart_2.copy()
_df = pd.DataFrame()
cols = []
for col in bl_col:
    try:
        _update = col[:-3]
        _col = _update
        depart[col].fillna(0, inplace=True)
        depart[_update].fillna(0, inplace=True)
        _df[col] = depart[_update] - depart[col]
        cols.append(col)
        # cols.append(_update)
    except KeyError:
        continue

_df


# In[65]:


counterpart_2.drop(cols, axis=1, inplace=True)
counterpart_2[cols] = _df
counterpart_2


# In[66]:


from joblib import load

counterpart_end = counterpart_2.copy()
for col in categorical2:
    model: LabelEncoder = load(MODEL_PATH + f'/{col}_lb.model')
    # try:
    #     counterpart_2[col] = counterpart_2[col].astype('float')
    # except ValueError:
    #     pass
    print(col)
    counterpart_2[col] = counterpart_2[col].astype(int)
    counterpart_2[col] = model.inverse_transform(counterpart_2[col].to_numpy())

counterpart_2[categorical2]


# In[66]:





# In[67]:


transfer = counterpart_end.dtypes[counterpart_end.dtypes == 'object'].index.tolist()
transfer.remove('PTID')
# transfer.remove('DX')

for col in transfer:
    counterpart_end[col] = counterpart_end[col].astype(float)


# In[68]:


counterpart_end.dtypes


# In[69]:


counterpart_end['DX'] = counterpart_end['DX'].astype(int)
iv = toad.quality(counterpart_end.drop(['RID', 'COLPROT', 'ORIGPROT', 'PTID', 'EXAMDATE',
                                      'EXAMDATE_bl', 'DX_bl'], axis=1), target='DX')
iv


# In[70]:


kl.corr_plot(counterpart_2, method='spearman', annot=False, figsize=(20, 20))


# In[70]:





# In[71]:


counterpart_2['DX'].unique()


# In[72]:


from scipy.stats import *


idx = counterpart_end.columns.tolist()[4:]
stat = pd.DataFrame(index=idx).drop('DX')
stat


# In[73]:


group = counterpart_end.groupby('DX')


# In[74]:


lb: LabelEncoder = load(MODEL_PATH + '/DX_lb.model')
lb.inverse_transform([0, 1, 2])


# In[75]:


Mann_CN_Dem_p = []
Mann_CN_Dem_stat = []

for col in stat.index:
    d1 = group.get_group(0)
    d2 = group.get_group(1)
    sta, p = mannwhitneyu(d1[col].dropna(), d2[col].dropna())
    Mann_CN_Dem_p.append(p)
    Mann_CN_Dem_stat.append(sta)


Mann_Dem_Mci_p = []
Mann_Dem_Mci_stat = []
for col in stat.index:
    d1 = group.get_group(1)
    d2 = group.get_group(2)
    sta, p = mannwhitneyu(d1[col].dropna(), d2[col].dropna())
    Mann_Dem_Mci_p.append(p)
    Mann_Dem_Mci_stat.append(sta)

Mann_CN_Mci_p = []
Mann_CN_Mci_stat = []
for col in stat.index:
    d1 = group.get_group(0)
    d2 = group.get_group(2)
    sta, p = mannwhitneyu(d1[col].dropna(), d2[col].dropna())
    Mann_CN_Mci_p.append(p)
    Mann_CN_Mci_stat.append(sta)

stat['Mann_CN_Dem'] = Mann_CN_Dem_p
stat['Mann_CN_Dem_Stats'] = Mann_CN_Dem_stat
stat['Mann_Dem_Mci'] = Mann_Dem_Mci_p
stat['Mann_Dem_Mci_Stats'] = Mann_Dem_Mci_stat
stat['Mann_CN_Mci'] = Mann_CN_Mci_p
stat['Mann_CN_Mci_Stats'] = Mann_CN_Mci_stat

stat


# In[76]:


_ = None
chi = []
for col in stat.index:
    cross_table = pd.crosstab(counterpart_2[col], counterpart_2['DX'])
    chi.append(
        chi2_contingency(cross_table)[1]
    )

chi


# In[77]:


_df


# In[78]:


cols


# In[79]:


counterpart_2.loc[:, cols] = _df
counterpart_2


# In[80]:


counterpart_2


# In[81]:


counterpart_2.to_excel(RESULT_PATH + '/evolve_data.xlsx')
counterpart_end.to_excel(RESULT_PATH + '/encode_data.xlsx')


# In[82]:


statsm = pd.concat([iv, stat], axis=1)
statsm['chi2'] = chi
statsm


# In[83]:


statsm.to_csv(RESULT_PATH + '/stats.csv')


# In[84]:


kl.corr_plot(counterpart_end, target='DX', method='spearman', annot=False)
plt.savefig(create_figure('target_corr'), dpi=800)


# In[85]:


(counterpart_end['VISCODE'] == counterpart['M']).all()


# In[85]:




