
# In[1]:
import numpy as np
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier

# In[2]:
df=pd.read_csv(r"C:\Users\hp\Downloads\Crop_recommendation.csv")

# In[3]:
df.head()

# In[4]:
df.isnull().sum()  # no null data

# In[5]:
df.shape

# In[6]:
# now we seprate input output 
X=df.iloc[:,:-1]
y=df.iloc[:,-1]

# In[7]:
from sklearn.model_selection import train_test_split

# In[8]:
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=40)

# In[9]:
X_train

# In[10]:
y_train

# In[11]:
model=RandomForestClassifier()

# In[12]:
model.fit(X_train,y_train)

# In[13]:
prediction=model.predict(X_test)

# In[14]:


# In[15]:
acuracy=model.score(X_test,y_test)

# In[16]:
print(acuracy)

# In[17]:
newfeature=[[36,58,25,28.16,59.31,8.39,36.9]]

# In[18]:
predicted_crop=model.predict(newfeature)

# In[19]:
print(predicted_crop)

# In[20]:
import pickle

# In[21]:
pickle.dump(model,open("model.pkl","wb"))

# In[22]:
