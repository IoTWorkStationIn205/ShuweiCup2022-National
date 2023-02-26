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
PIC_PATH = "../../models/image/image2"
DATA_PATH = '../../data'
RESULT_PATH = '../../data/summary'
MODEL_PATH = '../../models/model2'


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


data = pd.read_excel(RESULT_PATH + '/evolve_data.xlsx', index_col=0)
data


# In[5]:


features = ['RID', 'IMAGEUID', 'Ventricles_bl', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform',
            'MidTemp', 'ICV',
            'CDRSB', 'ADAS11', 'ADAS13', 'ADASQ4', 'MMSE', 'RAVLT_immediate',
            'RAVLT_learning', 'RAVLT_forgetting', 'RAVLT_perc_forgetting',
            'DX']
counterpart = data[features]
counterpart


# In[6]:


kl.missingval_plot(counterpart)
plt.tight_layout()
plt.savefig(create_figure('missing_plot'))


# In[7]:


from dataprep.eda import create_report

create_report(counterpart).show_browser()


# In[8]:


import missingno as mns

mns.heatmap(counterpart)
plt.tight_layout()
plt.savefig(create_figure('missing_plot_heatmap'), dpi=800)


# In[9]:


from missingpy import MissForest

forest = MissForest()
missing_eva = counterpart.loc[:, 'CDRSB': 'RAVLT_perc_forgetting']
missing_eva


# In[10]:


impute = forest.fit_transform(missing_eva)
impute


# In[11]:


counterpart.loc[:, 'CDRSB': 'RAVLT_perc_forgetting'] = impute
counterpart.isna().sum()


# In[12]:


counterpart.loc[:, ['Entorhinal', 'ICV']] = forest.fit_transform(
    counterpart.loc[:, ['Entorhinal', 'ICV']]
)
counterpart


# In[13]:


kl.corr_plot(counterpart, method='spearman', annot=False, figsize=(16, 16))
plt.tight_layout()
plt.savefig(create_figure('spearman_corr'), dpi=800)


# In[14]:


corr = counterpart.corr()

plt.figure(figsize=(16, 16))
mask = np.zeros_like(corr[corr>=.7],dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.set_style(style="white")
sns.heatmap(corr[corr>=.7],annot=False,mask=mask,cbar=True, linewidths=.5)
plt.tight_layout()
plt.savefig(create_figure('Strong Correlation'), dpi=1200, bbox_inches = 'tight')


# In[15]:


brain = counterpart.loc[:, 'IMAGEUID': 'ICV']
plt.figure(figsize=(24, 24))
sns.pairplot(brain, kind='scatter')
plt.tight_layout()
plt.savefig(create_figure('brain_pairplot'), dpi=1200)


# In[16]:


behaviour = counterpart.loc[:, 'CDRSB': 'RAVLT_perc_forgetting']
plt.figure(figsize=(24, 24))
sns.pairplot(behaviour, kind='reg')
plt.tight_layout()
plt.savefig(create_figure('behaviour_pairplot'), dpi=1200)


# In[17]:


corr[corr > .7].dropna(how='all')


# In[18]:


counterpart['RAVLT_perc_forgetting'] = np.where(
    counterpart['RAVLT_perc_forgetting'] < 0, 0, counterpart['RAVLT_perc_forgetting']
)


# In[19]:


behaviour


# In[20]:


from scipy import stats

i = 1
plt.figure(figsize=(16, 16))
for col in behaviour.columns:
    plt.subplot(3, 3, i)
    i += 1
    _sort = np.sort(counterpart[col])
    _yval = stats.norm.cdf(_sort, 0, 1)
    plt.xlabel(col)
    sns.lineplot(_sort, _yval)

plt.savefig(create_figure('behaviour_cdf'))


# In[21]:


i = 1
plt.figure(figsize=(16, 16))
for col in behaviour.columns:
    plt.subplot(3, 3, i)
    i += 1
    _sort = np.sort(counterpart[col])
    _yval = stats.norm.cdf(_sort, 0, 1)
    _label = stats.norm.ppf(_yval)
    plt.xlabel(col)
    sns.scatterplot(_label, _sort)

plt.tight_layout()
plt.savefig(create_figure('behaviour_pp_plot'))


# In[22]:


def drop_3_sigma(df: pd.DataFrame):
    _ = df.copy()
    for cols in df.columns:
        if _[cols].dtype == 'float64':
            rule = (_[cols].mean() - 3 * _[cols].std() > _[cols]) |                   (_[cols].mean() + 3 * _[cols].std() < _[cols])
            index = np.arange(_[cols].shape[0])[rule]
            _.iloc[index] = np.nan
    return _


# In[22]:





# In[23]:


data_clean = drop_3_sigma(behaviour)
data_clean.dropna(axis=0, inplace=True)
data_clean


# In[24]:


plt.figure(figsize=(20, 10))
plt.subplot(231)
sns.regplot(x='ADAS11', y='ADAS13', data=counterpart)
plt.subplot(232)
sns.regplot(x='CDRSB', y='ADAS11', data=counterpart)
plt.subplot(233)
sns.regplot(x='ADAS13', y='ADASQ4', data=counterpart)
plt.subplot(234)
sns.regplot(x='RAVLT_forgetting', y='RAVLT_perc_forgetting', data=counterpart)
plt.subplot(235)
sns.regplot(x='ADAS11', y='ADASQ4', data=counterpart)
plt.subplot(236)
sns.regplot(x='WholeBrain', y='MidTemp', data=counterpart)


# In[25]:


import toad

counterpart_2 = counterpart.copy()
counterpart_2.loc[:, 'CDRSB': 'RAVLT_perc_forgetting'] = data_clean
counterpart_2.dropna(axis=0, inplace=True)
counterpart_2.to_excel(RESULT_PATH + '/data_clean.xlsx')
counterpart_2['DX'] = counterpart_2['DX'].map(
    {
        'CN': 0, 'MCI': 1, 'Dementia': 2
    }
)
toad.quality(counterpart_2, target='DX')


# In[26]:


stat = pd.read_csv(RESULT_PATH + '/stats.csv', index_col=0)
stat


# In[27]:


index = [idx for idx in stat.index if idx in counterpart_2.columns]
stat = stat.loc[index]
stat


# In[28]:


counterpart_3 = counterpart_2.copy()
counterpart_3.drop(['ICV', 'WholeBrain'], axis=1)
X = counterpart_3.drop('DX', axis=1)
y = counterpart_3['DX']


# In[29]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)


# In[30]:


from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

def return_score(model):
    score = []
    model.fit(X_train, y_train)
    score.append(model.score(X_test, y_test))
    pred = model.predict(X_test)
    score.append(f1_score(y_test, pred, average='weighted'))
    # score.append(roc_auc_score(y_test, pred, multi_class='ovr'))
    score.append(precision_score(y_test, pred, average='weighted'))
    score.append(recall_score(y_test, pred, average='weighted'))

    return score


# In[31]:


from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier

X_train = pd.DataFrame(X_train, columns=counterpart_3.drop('DX', axis=1).columns)

score_account = pd.DataFrame(index=['accuracy', 'f1', 'precision', 'recall'])
score_account['Bayes'] = return_score(GaussianNB())
score_account['SVM'] = return_score(SVC())
score_account['DT'] = return_score(DecisionTreeClassifier())
score_account['RF'] = return_score(RandomForestClassifier())
score_account['GBDT'] = return_score(GradientBoostingClassifier())
score_account['XGB'] = return_score(XGBClassifier())

score_account


# In[32]:


X


# In[33]:


from keras.layers import Dense, BatchNormalization, Dropout
from keras.models import Sequential
from keras.callbacks import TensorBoard
from keras.utils.np_utils import to_categorical

model = Sequential()
model.add(Dense(input_dim=18, units=32, activation='relu'))
model.add(Dropout(.3))
# model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(Dropout(.3))
model.add(Dense(3, activation='softmax'))

y_nn = to_categorical(y, 3)
x_train_, x_test_, y_train_, y_test_ = train_test_split(X, y_nn, test_size=.3, random_state=42)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train_, y_train_, epochs=32, validation_split=.2, callbacks=[
    TensorBoard(log_dir=RESULT_PATH + '/log')
])


# In[34]:


model.evaluate(x_test_, y_test_)


# In[35]:


best_model = XGBClassifier(gpu_id=0)

best_model.fit(X_train, y_train)


# In[36]:


# from sklearn.metrics import make_scorer
# from sklearn.model_selection import cross_val_score
# from skopt.space import Real, Integer
# from skopt.utils import use_named_args
# from skopt import gp_minimize
# from sklearn.metrics import accuracy_score
#
# space = [
#     Integer(100, 1000, name='n_estimators'),
#     Integer(3, 10, name='max_depth'),
#     Real(1e-3, 1e-1, name='learning_rate'),
#     Integer(1, 10, name='min_child_weight')
# ]
#
# @use_named_args(space)
# def objective(**params):
#     best_model.set_params(**params)
#     return 1 - np.mean(cross_val_score(
#         best_model, X, y, cv=3, n_jobs=-1, scoring=make_scorer(accuracy_score)
#     ))
#
# res_gp = gp_minimize(objective, space, random_state=42, n_calls=50)


# In[37]:


# res_gp.x


# In[38]:


# best_model = XGBClassifier(n_estimators=res_gp.x[0], max_depth=res_gp.x[1],
#                            learning_rate=res_gp.x[2], min_child_weitht=res_gp.x[3])
# return_score(best_model)


# In[39]:


X = counterpart_3.drop('DX', axis=1)
y = counterpart_3['DX']

xgb = XGBClassifier(gpu_id=0)
xgb.fit(X, y)


# In[40]:


import shap

explain = shap.TreeExplainer(xgb, X)
shap_value = explain.shap_values(X)
shap_value2 = explain(X)


# In[41]:


shap.summary_plot(shap_value, X, show=False)
plt.xlabel('SHAP Value')
plt.legend(['CN', 'Dementia', 'Mci', ])
plt.tight_layout()
plt.savefig(create_figure('feather impact'), dpi=800)


# In[42]:


shap.dependence_plot('CDRSB', shap_value[0], X, show=False)
plt.savefig(create_figure('CDRSB_CN_Dependence'), dpi=800)


# In[43]:


shap.dependence_plot('CDRSB', shap_value[1], X, show=False)
plt.savefig(create_figure('CDRSB_Dem_Dependence'), dpi=800)


# In[44]:


shap.dependence_plot('CDRSB', shap_value[2], X, show=False)
plt.savefig(create_figure('CDRSB_Mci_Dependence'), dpi=800)


# In[44]:




