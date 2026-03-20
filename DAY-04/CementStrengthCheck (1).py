
# In[1]:
#today i learned power tranfer so now i am going to implent this 

# In[2]:
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

# In[3]:
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# In[4]:
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PowerTransformer

# In[5]:
df=pd.read_csv(r"C:\Users\hp\Downloads\concrete_data.csv")

# In[6]:
df.head()

# In[7]:
df.shape

# In[8]:
df.isnull().sum()

# In[9]:
df.describe()

# In[10]:
X=df.drop(columns=["Strength"])
y=df.iloc[:,-1]

# In[11]:
X_train,X_test,y_train,y_test=train_test_split=train_test_split(X,y,test_size=0.2,random_state=42)

# In[12]:
X_train.shape

# In[13]:
#first applying linearRegression without PowerTransformer
lr=LinearRegression()
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)
r2_score(y_test,y_pred)

# In[14]:
#cross val score 
lr=LinearRegression()
np.mean(cross_val_score(lr,X,y,scoring='r2'))

# In[15]:
#ploting distplot without PPowerTransformer
for col in X_train.columns:
    plt.figure(figsize=(14,4))
    plt.subplot(121)
    sns.distplot(X_train[col])
    plt.title(col)
    plt.subplot(122)
    stats.probplot(X_train[col],dist="norm",plot=plt)
    plt.show()

# In[16]:
# we can see there are normal and  skewed data

# In[17]:
# since yeo Jhonson superior for higly skewed and psotive and negative data but box cox best for simpler and postive data

# In[18]:
pt=PowerTransformer()
X_train_transformed=pt.fit_transform(X_train)
X_test_transformed=pt.transform(X_test)

model=LinearRegression()

# In[19]:
model.fit(X_train_transformed,y_train)

# In[20]:
y_pred=model.predict(X_test_transformed)
print(r2_score(y_test,y_pred))

# In[21]:
# as you can see how tranformation improves accuracy of model 

# In[22]:
#applying Cross val score 

# In[23]:
pt=PowerTransformer()
X_train_transformed=pt.fit_transform(X)
lr=LinearRegression()
np.mean(cross_val_score(lr,X_train_transformed,y,scoring='r2'))

# In[24]:
import pickle

# In[25]:
pickle.dump(model,open("model.pkl","wb"))

# In[26]:
