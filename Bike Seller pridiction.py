
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

import os 
os.getcwd()
os.chdir('D:\study\ml\others')
os.getcwd()

dataset = pd.read_csv('bikebuyer1 (1).csv')


# <h1><b> RECTIFYNG THE VALUES IN THE DATASET</b></h1>

# In[2]:


dataset


# In[3]:


dataset.isnull().sum()


# <h1><b> CHECKING FOR OUTLIERS AND PLOTING </b></h1>

# In[4]:


import seaborn as sns
sns.boxplot(x=dataset['Yearly Income'])


# In[5]:


dataset['Yearly Income'].value_counts()


# In[6]:


dataset['Yearly Income'].median()


# In[7]:


dataset['Yearly Income'].replace([150000, 160000, 170000],[dataset['Yearly Income'].median(), dataset['Yearly Income'].median(), dataset['Yearly Income'].median()], inplace = True)


# In[8]:


dataset['Yearly Income'].value_counts()


# In[9]:


sns.boxplot(x=dataset['Yearly Income'])


# In[10]:


sns.boxplot(x=dataset['Cars'])


# In[11]:


dataset['Cars'].value_counts()


# In[12]:


dataset['Cars'].median()


# In[13]:


dataset['Cars'].replace([4],[dataset['Cars'].median()],inplace = True)


# In[14]:


dataset['Cars'].value_counts()


# In[15]:


sns.boxplot(x=dataset['Cars'])


# In[16]:


sns.boxplot(x=dataset['Commute Distance'])


# In[17]:


dataset['Commute Distance'].value_counts()


# In[18]:


sns.boxplot(x=dataset['Age'])


# In[19]:


dataset['Age'].value_counts()


# In[20]:


dataset['Age'].replace([77,78,80,84,94,86,82,79, 83, 93,89, 96,85,90,87,95], [42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42], inplace=True)


# In[21]:


sns.boxplot(x=dataset['Age'])


# <h1><b> REMOVING THE MISSING VALUES  </b></h1>

# In[22]:


dataset.isnull().sum()


# In[23]:


dataset.apply(lambda x: sum(x.isnull()))


# In[24]:


dataset['Marital Status'].value_counts()


# In[25]:


dataset['Marital Status'].fillna('Married', inplace=True)


# In[26]:


dataset.isnull().sum()


# In[27]:


dataset['Gender'].value_counts()


# In[28]:


dataset['Gender'].fillna('Male', inplace=True)


# In[29]:


dataset.isnull().sum()


# In[30]:


dataset['Children'].median()


# In[31]:


dataset['Children'].value_counts()


# In[32]:


dataset['Children'].fillna('0.0', inplace=True)


# In[33]:


dataset.isnull().sum()


# In[34]:


dataset['Commute Distance'].value_counts()


# In[35]:


dataset['Commute Distance'].median()


# In[36]:


dataset['Commute Distance'].fillna('4.0', inplace=True)


# In[37]:


dataset.isnull().sum()


# <h1><b> DROPPING THE UNNECCESSARY COLUMNS </b></h1>

# In[38]:


dataset=dataset.drop('ID',axis=1)


# In[39]:


dataset


# In[40]:


dataset.isnull().sum()


# <h1><b> CATEGORISING THE DATASET ACCORDINGLY WITH THE FEATURES NECCESSARY </h1></b>

# In[41]:


dataset[dataset['Gender'].str.contains('Male') & dataset['Region'].str.contains('Europe') & dataset['Bike Buyer'].str.contains('Yes')]


# In[42]:


dataset[dataset['Marital Status'].str.contains('Single') & (dataset['Children'] == 1)]


# In[43]:


dataset.dtypes


# In[44]:


dataset['Children'] = dataset['Children'].astype(np.float64)


# In[45]:


dataset.dtypes


# In[46]:


dataset[dataset['Marital Status'].str.contains('Single') & (dataset['Children'] >= 1)]


# In[47]:


dataset['Children'] = dataset['Children'].astype(np.object)


# In[48]:


dataset['Children'] = dataset['Children'].astype(np.float64)


# In[49]:


dataset.dtypes


# In[50]:


dataset


# In[51]:


X = dataset.iloc[:,0:11].values
Y = dataset.iloc[:, 11].values


# In[52]:


from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform((X[:,0]))
X[:,1] = labelencoder_X.fit_transform((X[:,1]))
X[:,4] = labelencoder_X.fit_transform((X[:,4]))
X[:,5] = labelencoder_X.fit_transform((X[:,5]))
X[:,6] = labelencoder_X.fit_transform((X[:,6]))
X[:,9] = labelencoder_X.fit_transform((X[:,9]))


# In[53]:


onehotencoder = OneHotEncoder(categorical_features = [4])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [5])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [9])
X = onehotencoder.fit_transform(X).toarray()


# In[54]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)


# In[55]:


from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.20, random_state = 0)


# <h1><b> APPLYING LOGISTIC REGRESSION </b></h1>

# In[56]:


# logistic regression 
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)


# In[57]:


Y_pred_logic = classifier.predict(X_test)


# In[58]:


from sklearn.metrics import confusion_matrix
cm_logic = confusion_matrix(Y_test,Y_pred_logic)
cm_logic


# In[59]:


acc_logic = (1203/(1203+197))*100
acc_logic


# <h1><b> APPLYING KNN </b></h1>

# In[60]:


# KNN
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, Y_train)


# In[61]:


Y_pred_knn = classifier.predict(X_test)


# In[62]:


from sklearn.metrics import confusion_matrix
cm_knn = confusion_matrix(Y_test,Y_pred_knn)
cm_knn


# In[63]:


acc_knn = ((1167+25)/(1167+37+171+25))*100
acc_knn


# <h1><b> APPLYING SVC_Linear </b></h1>

# In[64]:


# SVC_linear
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0 )
classifier.fit(X_train, Y_train)


# In[65]:


Y_pred_svc1 = classifier.predict(X_test)


# In[66]:


from sklearn.metrics import confusion_matrix
cm_svc1 = confusion_matrix(Y_test,Y_pred_svc1)
cm_svc1


# In[67]:


acc_svc1 = ((1204)/(1204+196))*100
acc_svc1


# <h1><b> APPLYING SVC_rbf </b></h1>

# In[68]:


# SVC_rbf
from sklearn.svm import SVC 
classifier = SVC(kernel = 'rbf',random_state = 0 )
classifier.fit(X_train , Y_train)


# In[69]:


Y_pred_svc2 = classifier.predict(X_test)


# In[70]:


from sklearn.metrics import confusion_matrix
cm_svc2 = confusion_matrix(Y_test,Y_pred_svc2)
cm_svc2


# In[71]:


acc_svc2 = ((1204)/(1204+196))*100
acc_svc2


# <h1><b> APPLYING NAIVE BAYES </b></h1>

# In[72]:


# naive bayes 
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train , Y_train)


# In[73]:


Y_pred_nb = classifier.predict(X_test)


# In[74]:


from sklearn.metrics import confusion_matrix
cm_nb = confusion_matrix(Y_test,Y_pred_nb)
cm_nb


# In[75]:


acc_svc2 = ((1049+38)/(1049+155+158+38))*100
acc_svc2


# <h1><b> APPLYING DECISION TREE </b></h1>

# In[76]:


# decision tree
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train , Y_train)


# In[77]:


Y_pred_dt = classifier.predict(X_test)


# In[78]:


from sklearn.metrics import confusion_matrix
cm_dt = confusion_matrix(Y_test,Y_pred_dt)
cm_dt


# In[79]:


acc_svc2 = ((1062+76)/(1062+142+120+76))*100
acc_svc2


# <h1><b> APPLYING RANDOM FOREST</b></h1>

# In[80]:


# randomforest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy',random_state = 0)
classifier.fit(X_train , Y_train)


# In[81]:


Y_pred_rf = classifier.predict(X_test)


# In[82]:


from sklearn.metrics import confusion_matrix
cm_rf = confusion_matrix(Y_test,Y_pred_rf)
cm_rf


# In[83]:


acc_svc2 = ((1120+52)/(1120+84+144+52))*100
acc_svc2


# <h1><b> APPLYING PCA </b></h1>

# In[84]:


from sklearn.decomposition import PCA
pca = PCA()
pca.fit(X_train)
cs = np.cumsum(pca.explained_variance_ratio_)


# In[85]:


d = np.argmax(cs>=0.95) + 1


# In[86]:


"""from sklearn.decomposition import PCA
pca = PCA(n_components = d)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_"""


# In[87]:


"""explained_variance.sum()"""


# In[88]:


pca = PCA(n_components=d, svd_solver='randomized',
          whiten=True).fit(X_train)
X_reduced=pca.fit_transform(X_train)


# In[89]:


plt.hist(pca.explained_variance_ratio_, bins=16, log=True)
pca.explained_variance_ratio_.sum()


# In[90]:


# SVC_rbf_2\
from sklearn.svm import SVC 
classifier = SVC(kernel = 'rbf',random_state = 0 )
classifier.fit(X_train , Y_train)


# In[91]:


Y_pred_svc2_2 = classifier.predict(X_test)


# In[92]:


from sklearn.metrics import confusion_matrix
cm_svc2_2 = confusion_matrix(Y_test,Y_pred_svc2_2)
cm_svc2_2

