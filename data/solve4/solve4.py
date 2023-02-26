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
PIC_PATH = "../../models/image/image4"
DATA_PATH = '../../data'
RESULT_PATH = '../../data/summary'
MODEL_PATH = '../../models/model4'


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


features = ['RID', 'CDRSB_bl', 'AGE',
 'ADAS11_bl',
 'ADAS13_bl',
 'ADASQ4_bl',
 'MMSE_bl',
 'RAVLT_immediate_bl',
 'RAVLT_learning_bl',
 'RAVLT_forgetting_bl',
 'RAVLT_perc_forgetting_bl',
 'TRABSCOR_bl',
 'FAQ_bl',
'mPACCdigit_bl',
 'mPACCtrailsB_bl',
 'IMAGEUID_bl',
 'Ventricles_bl',
 'Hippocampus_bl',
 'WholeBrain_bl',
 'Entorhinal_bl',
 'Fusiform_bl',
 'MidTemp_bl',
 'ICV_bl',
 'MOCA_bl',
'EcogPtTotal_bl',
 'EcogSPTotal_bl',
 'Month_bl', 'DX_bl', 'DX', 'EXAMDATE_bl', 'EXAMDATE']
data_new = data[features]
data_new


# In[6]:


data_new['timestamp'] = data_new['EXAMDATE'] - data_new['EXAMDATE_bl']
data_new['timestamp'] = data_new['timestamp'].astype(np.str)
data_new['timestamp'] = [int(i[0]) for i in data_new['timestamp'].str.split(' ')]
data_new.drop('EXAMDATE', axis=1, inplace=True)
data_new


# In[7]:


data_new = data_new[data_new['DX'] != data_new['DX_bl']]
data_new = data_new.sort_values(by='RID', ascending=False)
data_new


# In[8]:


data_new['target'] = data_new['DX_bl'] + '_' + data_new['DX']
data_new['target'].unique()


# In[9]:


data_new.drop(['DX', 'DX_bl'], axis=1, inplace=True)
group = data_new.groupby('target')
target = data_new['target'].unique()


# In[10]:


kl.corr_plot(data_new, annot=False, method='spearman', figsize=(16, 14))
plt.savefig(create_figure('spearman'), dpi=800)


# In[11]:


kl.corr_plot(data_new, annot=False, method='spearman', figsize=(16, 14), target='timestamp')
plt.savefig(create_figure('spearman_time_diff'), dpi=800)


# In[12]:


import toad

detect = pd.DataFrame()

for i in target:
    group_i = group.get_group(i)
    _detect = toad.detect(group_i)
    _detect['group'] = i
    detect = pd.concat([detect, _detect], axis=0)

detect.to_excel(RESULT_PATH + '/solve4_group.xlsx')


# In[13]:


be_bad = ['LMCI_Dementia', 'EMCI_Dementia', 'CN_MCI', 'SMC_Dementia', 'CN_Dementia']
be_better = ['SMC_CN', 'AD_MCI', 'EMCI_CN', 'LMCI_CN']


# In[14]:


from statsmodels.stats.anova import anova_lm
from statsmodels.sandbox.stats.multicomp import MultiComparison
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import MultiComparison

def oneWayAnova(df,cata_name,num_name,alpha_anova=0.05,alpha_tukey=0.05):
    info = ''
    df[cata_name]=df[cata_name].astype('str')

    s1=df[cata_name]
    s2=df[num_name]

    fml=num_name+'~C('+cata_name+')'

    model = ols(fml,data=df).fit()
    anova_table_1 = anova_lm(model, typ = 2).reset_index()
    p1=anova_table_1.loc[0,'PR(>F)']

    if p1>alpha_anova:
        print(num_name + '组间【无】显著差异')
        info = info + '\n' + str(num_name) + '组间【无】显著差异\n'
    else:
        print(num_name + '组间【有】显著差异')
        info = info + '\n' + str(num_name) + '组间【有】显著差异\n'

    df_p1=df.groupby([cata_name])[num_name].describe()

    mc = MultiComparison(df[num_name],df[cata_name])
    df_smry = mc.tukeyhsd(alpha=alpha_tukey).summary()
    m = np.array(df_smry.data)
    df_p2 =pd.DataFrame(m[1:],columns=m[0])

    df_p1_sub=df_p1[['mean']].copy()
    df_p1_sub.sort_values(by='mean',inplace=True)

    output_list=[]

    for x in range(1,len(df_p1_sub.index)):
        if (df_p2.loc[((df_p2.group1==df_p1_sub.index[x-1])&(df_p2.group2==df_p1_sub.index[x]))|
                      ((df_p2.group1==df_p1_sub.index[x])&(df_p2.group2==df_p1_sub.index[x-1])),
                      'reject'].iloc[0])=="True":
            smb='<'
        else:
            smb='<='
        if x==1:
            output_list.append(df_p1_sub.index[x-1])
            output_list.append(smb)
            output_list.append(df_p1_sub.index[x])
        else:
            output_list.append(smb)
            output_list.append(df_p1_sub.index[x])
    out_sentence=' '.join(output_list)
    print(out_sentence)
    info += out_sentence
    info += '\n\n'

    return df_p1,df_p2, info


# In[15]:


better = pd.DataFrame()

for i in be_better:
    better = pd.concat([better, group.get_group(i)], axis=0)

better


# In[16]:


better_excel = pd.DataFrame()

info = ''
columns = better.drop(['target', 'EXAMDATE_bl', 'RID'], axis=1).columns
for col in columns:
    d1, d2, infos = oneWayAnova(better, cata_name='target', num_name=col)
    d2['target'] = col
    better_excel = pd.concat([better_excel, d2], axis=0)
    print(d2)
    info = info + str(d2) + infos
better_excel.to_excel(RESULT_PATH + '/solve_5_better_ANOVA.xlsx')
with open(RESULT_PATH + '/solve5_info_better.txt', 'w') as f:
    f.write(info)


# In[17]:


bader = pd.DataFrame()

for i in be_bad:
    bader = pd.concat([bader, group.get_group(i)], axis=0)

bader


# In[18]:


info = ''
bader_excel = pd.DataFrame()

columns = better.drop(['target', 'EXAMDATE_bl', 'RID'], axis=1).columns
for col in columns:
    d1, d2, infos = oneWayAnova(bader, cata_name='target', num_name=col)
    d2['target'] = col
    bader_excel = pd.concat([bader_excel, d2], axis=0)
    print(d2)
    info = info + str(d2) + infos
better_excel.to_excel(RESULT_PATH + '/solve_5_bader_ANOVA.xlsx')
with open(RESULT_PATH + '/solve5_info_bader.txt', 'w') as f:
    f.write(info)


# In[19]:


better['target'] = 'better'
bader['target'] = 'bader'

all_ = pd.concat([better, bader], axis=0)
info = ''
result_excel = pd.DataFrame()

for col in columns:
    d1, d2, infos = oneWayAnova(all_, cata_name='target', num_name=col)
    d2['target'] = col
    result_excel = pd.concat([result_excel, d2], axis=0)
    print(d2)
    info = info + str(d2) + infos
result_excel.to_excel(RESULT_PATH + '/solve_5_result_ANOVA.xlsx')
with open(RESULT_PATH + '/solve5_info_result.txt', 'w') as f:
    f.write(info)


# In[20]:


g_detect = toad.detect(better)
b_detect = toad.detect(bader)

g_detect['target'] = 'better'
b_detect['target'] = 'worsen'

res_detect = pd.concat([g_detect, b_detect], axis=0)
res_detect


# In[21]:


columns_none = ['EcogSPTotal_bl', 'Month_bl', 'IMAGEUID_bl']

for col in columns_none:
    plt.figure(figsize=(8, 5))
    df = pd.DataFrame(
        {
            col: ['mean', 'mean', 'std', 'std'],
            'value': res_detect.loc[col, 'mean_or_top1'].tolist() +
                     res_detect.loc[col, 'std_or_top2'].tolist(),
            'hue': ['better', 'worsen', 'better', 'worsen']
        }
    )
    sns.barplot(x=col, y='value', hue='hue', data=df)
    plt.savefig(create_figure(f'{col}_description'), dpi=800)


# In[22]:


for col in columns_none:
    plt.figure(figsize=(8, 5))
    sns.distplot(better[col], label='better')
    sns.distplot(bader[col], label='worsen')
    plt.legend()
    plt.savefig(create_figure(f'{col}_distplot'), dpi=800)


# In[23]:


plt.figure(figsize=(8, 5))
sns.distplot(better['timestamp'], label='better')
sns.distplot(bader['timestamp'], label='worsen')
plt.legend()
plt.savefig(create_figure('timestamp_distplot'), dpi=800)


# In[24]:


plt.figure(figsize=(8, 5))
sns.distplot(better['AGE'], label='better')
sns.distplot(bader['AGE'], label='worsen')
plt.legend()
plt.savefig(create_figure('AGE_distplot'), dpi=800)


# In[ ]:




