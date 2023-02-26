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
PIC_PATH = "../../models/image/image3"
DATA_PATH = '../../data'
RESULT_PATH = '../../data/summary'
MODEL_PATH = '../../models/model3'


# In[2]:


figure_count = 0

def create_figure(figure_name):
    global figure_count
    figure_count += 1
    return PIC_PATH + f'/figure{figure_count}_{figure_name}.png'


# In[3]:


data = pd.read_excel(RESULT_PATH + '/data_clean.xlsx', index_col=0)
data


# In[4]:


data_whole = pd.read_excel(RESULT_PATH + '/evolve_data.xlsx', index_col=0)
data_whole


# In[5]:


features = ['IMAGEUID_bl', 'Ventricles_bl', 'Hippocampus_bl', 'WholeBrain_bl',
            'Entorhinal_bl', 'Fusiform_bl',
            'MidTemp_bl', 'ICV_bl',
            'CDRSB_bl', 'ADAS11_bl', 'ADAS13_bl', 'ADASQ4_bl', 'MMSE_bl', 'RAVLT_immediate_bl',
            'RAVLT_learning_bl', 'RAVLT_forgetting_bl', 'RAVLT_perc_forgetting_bl',
            'DX_bl', 'IMAGEUID', 'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal',
       'Fusiform', 'MidTemp', 'ICV']

fea = [col for col in data_whole.columns if col in features]
whole = data_whole[fea]
whole


# In[6]:


import toad

target = whole['DX_bl']
department = whole.copy()

department['DX_bl'] = department['DX_bl'].str.replace('EMCI', 'MCI')
department['DX_bl'] = department['DX_bl'].str.replace('LMCI', 'MCI')
department['DX_bl'] = department['DX_bl'].str.replace('SMC', 'MCI')

department['DX_bl'] = department['DX_bl'].map(
    {
        'CN': 0, 'MCI': 1, 'AD': 2
    }
)

toad.quality(department, 'DX_bl')


# In[7]:


select, drop = toad.select(department, target='DX_bl', return_drop=True)
select = select[select.columns[:8]]
select.dropna(axis=1, inplace=True)
select


# In[8]:


kl.corr_plot(select, annot=False, method='spearman')
plt.savefig(create_figure('select-corr'), dpi=800)


# In[9]:


from sklearn.preprocessing import StandardScaler


x = StandardScaler().fit_transform(select)


# In[10]:


department['DX_bl'].value_counts()


# In[11]:


from sklearn.metrics import accuracy_score
from sklearn.cluster import MiniBatchKMeans, KMeans
from kmodes.kmodes import KModes
from sklearn.metrics import classification_report

true = department['DX_bl']

model = KMeans(n_clusters=3)
pred = model.fit_predict(x, true)
print(classification_report(true, pred))


# In[12]:


from sklearn.metrics import silhouette_score, davies_bouldin_score


# In[13]:


display(
    silhouette_score(x, pred),
    davies_bouldin_score(x, pred)
)


# In[14]:


model = KModes(n_clusters=3)
pred = model.fit_predict(x, true)
print(classification_report(true, pred))


# In[15]:


pd.Series(pred).value_counts()


# In[16]:


from sklearn.metrics import silhouette_score, davies_bouldin_score

display(
    silhouette_score(x, pred),
    davies_bouldin_score(x, pred)
)


# In[17]:


import math
from minisom import MiniSom

N = x.shape[0]
M = x.shape[1]

size = math.ceil(np.sqrt(5 * np.sqrt(N)))

max_iter = 300

som = MiniSom(size, size, M, sigma=10, learning_rate=0.1,
              neighborhood_function='gaussian')


# In[18]:


som.pca_weights_init(x)
som.train_batch(x, max_iter, verbose=False)
winmap = som.labels_map(x, true)


# In[19]:


def classify(som,data,winmap):
    from numpy import sum as npsum
    default_class = npsum(list(winmap.values())).most_common()[0][0]
    result = []
    for d in data:
        win_position = som.winner(d)
        if win_position in winmap:
            result.append(winmap[win_position].most_common()[0][0])
        else:
            result.append(default_class)
    return result


# In[20]:


y_pred = classify(som, x, winmap)
print(classification_report(true, np.array(y_pred)))


# In[21]:


display(
    silhouette_score(x, y_pred),
    davies_bouldin_score(x, y_pred)
)


# In[22]:


heatmap = som.distance_map()
plt.figure(figsize=(9, 9))
plt.imshow(heatmap)
plt.colorbar()
plt.savefig(create_figure('U-matrix'), dpi=800)


# In[23]:


from sklearn.metrics import confusion_matrix

plt.figure(figsize=(12, 9))
matrix = confusion_matrix(true, y_pred)
sns.heatmap(matrix, annot=False)
plt.savefig(create_figure('confusion-matrix'), dpi=800)


# In[24]:


from skfuzzy.cluster import cmeans

center, u, u0, d, jm, p, fpc = cmeans(x.T, m=2, c=3, error=5e-3, maxiter=1000)
label = u.T.argmax(1)


# In[25]:


display(
    silhouette_score(x, label),
    davies_bouldin_score(x, label)
)


# In[26]:


pd.Series(label).value_counts()


# In[27]:


label = pd.Series(label).map(
    {1: 2, 0: 0, 2: 1}
)

print(classification_report(true, label))


# In[28]:


department = data_whole[select.columns]
smc = department[department['DX_bl'] == 'SMC']
emci = department[department['DX_bl'] == 'EMCI']
lmci = department[department['DX_bl'] == 'LMCI']

department = pd.concat([smc, emci, lmci], axis=0)
department


# In[29]:


sns.countplot(department['DX_bl'])
plt.xlabel('MCI')
plt.savefig(create_figure('mci_count'), dpi=800)


# In[30]:


target = department['DX_bl']
select = department.drop('DX_bl', axis=1)
select


# In[31]:


from sklearn.preprocessing import MinMaxScaler


# In[32]:


x = StandardScaler().fit_transform(select)

N = x.shape[0]
M = x.shape[1]

size = math.ceil(np.sqrt(5 * np.sqrt(N)))

max_iter = 1000

som2 = MiniSom(size, size, M, sigma=5, learning_rate=0.1,
              neighborhood_function='bubble', activation_distance='euclidean')
som2.pca_weights_init(x)
som2.train_batch(x, max_iter, verbose=True)
winmap = som2.labels_map(x, target)

y_pred = classify(som2, x, winmap)
print(classification_report(target, np.array(y_pred)))


# In[32]:





# In[33]:


heatmap = som2.distance_map()
plt.figure(figsize=(9, 9))
plt.imshow(heatmap)
plt.colorbar()
plt.savefig(create_figure('U-matrix-2'), dpi=800)


# In[34]:


display(
    silhouette_score(x, y_pred),
    davies_bouldin_score(x, y_pred)
)


# In[35]:


w = som2.get_weights()
w


# In[36]:


color_map = {
    'SMC': 1,
    'LMCI': 2,
    'EMCI': 3
}

wmap = {}
im = 0

plt.figure(figsize=(9, 9))

test = pd.DataFrame(x)
test['target'] = target
test = test.sample(3000).dropna(axis=0, how='any')
x_ = test.drop('target', axis=1).to_numpy()
target_ = test['target'].to_numpy()

for temp1, temp2 in zip(x_, target_):
    w = som2.winner(temp1)
    # wmap[temp1] = im
    plt.text(w[0] + .5, w[1] + .5, str(temp2), color=plt.cm.Dark2(color_map[temp2]),
             fontdict={'weight': 'bold', 'size': 11})
    # im = im + 1
plt.axis([0, som2.get_weights().shape[0], 0,  som2.get_weights().shape[1]])
plt.savefig(create_figure('word_dis'), dpi=800)


# In[37]:


from matplotlib.gridspec import GridSpec

class_names = color_map.keys()

plt.figure(figsize=(12, 12))
the_grid = GridSpec(size, size)
for position in winmap.keys():
    label_fracs = [winmap[position][label] for label in [0,1,2]]
    plt.subplot(the_grid[position[1], position[0]], aspect=1)
    patches, texts = plt.pie(label_fracs)
    plt.text(position[0]/100, position[1]/100,  str(len(list(winmap[position].elements()))),
              color='black', fontdict={'weight': 'bold',  'size': 15},
              va='center',ha='center')
plt.tight_layout()
# plt.legend(patches, class_names, loc='upper right', bbox_to_anchor=(-1,0), ncol=3)
# plt.savefig()
plt.savefig(create_figure('num_dis'), dpi=800)


# In[38]:


plt.figure(figsize=(12, 12))

corr = pd.DataFrame(index=range(23), columns=range(23))
for position in winmap.keys():
    num = len(list(winmap[position].elements()))
    corr.iloc[position[0], position[1]] = num
corr.fillna(0, inplace=True)

plt.figure(figsize=(12, 12))
sns.heatmap(corr, cmap='coolwarm', center=150)
plt.savefig(create_figure('neural network'), dpi=800)


# In[40]:


W = som2.get_weights()
plt.figure(figsize=(18, 18))
for i, f in enumerate(select.columns[:5]):
    plt.subplot(2, 3, i+1)
    plt.title(f, fontdict={'size': 25})
    plt.imshow(W[:,:,i], cmap='coolwarm')
    # plt.colorbar()
    plt.xticks(np.arange(size))
    plt.yticks(np.arange(size))
plt.tight_layout()
plt.savefig(create_figure('feather_w1'), dpi=1200)


# In[39]:




