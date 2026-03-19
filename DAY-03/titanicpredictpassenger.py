
# In[1]:
import pandas as pd
import numpy as np


# In[2]:
df=pd.read_csv(r"C:\Users\hp\Downloads\train.csv",usecols=["Survived","Pclass","Age","Fare"])

# In[3]:
df.head()

# In[4]:
df.isnull().sum()



# In[6]:
df.isnull().sum()

# In[7]:
df.sample(5)

# In[8]:
df["Age"].fillna(df["Age"].mean(),inplace=True)

# In[9]:
df.isnull().sum()

# In[10]:
df.sample(5)

# In[11]:
df.shape

# In[12]:
from sklearn.model_selection import train_test_split

# In[13]:
X_train,X_test,y_train,y_test=train_test_split(df.drop("Survived",axis=1),df["Survived"],test_size=0.2,random_state=42)

# In[14]:
X_train

# In[15]:
y_test

# In[16]:
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.ensemble import RandomForestClassifier

# In[17]:
model=Pipeline([
    ('imputer',SimpleImputer()),
    ('scaler',MinMaxScaler()),
    ('classifier',RandomForestClassifier())
])

# In[18]:
model.fit(X_train,y_train)   # i use model beause i have to not need to edit my app.py file again

# In[19]:
y_pred=model.predict(X_test)

# In[20]:
print(y_pred)

# In[21]:
from sklearn.metrics import accuracy_score

# In[22]:
accuracy_score(y_test,y_pred)

# In[23]:
#crosvalidation using pipeline
from sklearn.model_selection import cross_val_score

# In[24]:
cross_val_score(model,X_train,y_train,cv=5,scoring='accuracy').mean()

# In[25]:
test_input=np.array([1,20,120],dtype=object).reshape(1,3)

# In[26]:
pipe.predict(test_input)

# In[27]:
import pickle

# In[28]:
pickle.dump(model,open("model.pkl","wb"))

# In[29]:
