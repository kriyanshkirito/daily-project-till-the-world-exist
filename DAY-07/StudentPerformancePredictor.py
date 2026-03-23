
# In[1]:
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# In[2]:
df=pd.read_csv(r"C:\Users\hp\Downloads\StudentPerformance.csv")

# In[3]:
df.head()

# In[4]:
df.shape

# In[5]:
df.info()

# In[6]:
df.isnull().sum(axis=1)  # no null object so we can perform test

# In[7]:
df.describe()

# In[8]:
df["Extracurricular Activities"].describe() # from here about 50% student take part in extracurricular activites 

# In[9]:
#check distribution skew or not
sns.histplot(df['Performance Index'],bins=30,kde=True) # as we can data is not skewed so becused bell shaped curved most value in middle 
print(df["Performance Index"].skew()) # if -0.5 to 0.5 no skew ,-1 to -0.5 or 0.5 to 1 moderetly skew ,less than -1 or greater than 1 moderetly skew
# if 0 perfect bell curve if greater than 0 then right skew
plt.figure() 


# check imbalance 
sns.countplot(x="Extracurricular Activities",data=df)
   # compare no of yes and no in our plot to bars our almost equal this means our data is well balnced if yes was 9000 and  no 1000 then then our model will be simply biased towards yes 


# In[10]:
sns.histplot(df['Hours Studied'],bins=30,kde=True)
print(df["Performance Index"].skew())      # So we can se that not skew

# In[11]:
plt.figure(figsize=(8,6))
sns.heatmap(df.iloc[:,[0,1,3,5]].corr(),annot=True,cmap="coolwarm")

# In[12]:
df["Extracurricular Activities"]=df["Extracurricular Activities"].replace({"Yes":1,"No":0})   # so our whole data should be number you can aslo use ohe but this easy

# In[13]:
df.head()

# In[14]:
sns.heatmap(data=df.corr(),annot=True,cmap="coolwarm")

# In[15]:
#does check sleep hour affectt performance or not if median are same confirm this sleep hour  not affect
sns.boxenplot(x="Sleep Hours",y="Performance Index",data=df)

# In[16]:
# since our dataset is best without any skewness and imbalances so we are 

# In[17]:
X=df.drop(columns=['Performance Index'])

# In[18]:
y=df['Performance Index']

# In[19]:
X

# In[20]:
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# In[21]:
X_train

# In[22]:
from sklearn.ensemble import RandomForestRegressor

# In[23]:
rfg=RandomForestRegressor()

# In[24]:
rfg.fit(X_train,y_train)

# In[25]:
y_pred=rfg.predict(X_test)

# In[26]:
from sklearn.metrics import r2_score

# In[27]:
r2_score(y_test,y_pred)

# In[28]:
from sklearn.tree import DecisionTreeClassifier

# In[29]:
clf=DecisionTreeClassifier()

# In[30]:
clf.fit(X_train,y_train)

# In[31]:
y_pred=clf.predict(X_test)

# In[32]:
r2_score(y_test,y_pred)

# In[33]:
# from both same so we now made in pickle

# In[34]:
import pickle

# In[35]:
pickle.dump(rfg,open("rfg.pkl","wb"))

# In[36]:
