# %%
import pandas as pd
import numpy as np

# %%
df=pd.read_csv(r"C:\Users\hp\Downloads\data_science_job (1).csv")

# %%
df.head()

# %%
df.shape

# %%
df.info()

# %%
df.isnull().mean()*100  # gives percent of data missing

# %%
# we are going to clean the datset

# %%


# %%
cols = [var for var in df.columns if df[var].isnull().mean() < 0.05 and df[var].isnull().mean() > 0]  # select those whose missing data less than 5 %
cols

# %%
df[cols].sample(5)

# %%
len(df[cols].dropna())/len(df)  # gives remaing ratio of data ( in 100 data 20 null then => 80/100=0.8 is the nof data reamain)

# %%
new_df=df[cols].dropna()

# %%
df.shape,new_df.shape  # ok, but now main test come we have to check previous v/s now distribution if they overlap almost then we can dropt if not overalap then we will not drop

# %%
import matplotlib.pyplot as plt

# %%
new_df.hist(bins=50, density=True, figsize=(12, 12))
plt.show()

# %%
fig = plt.figure()
ax = fig.add_subplot(111)

# original data
df['training_hours'].hist(bins=50, ax=ax, density=True, color='red')

# data after cca, the argument alpha makes the color transparent, so we can
# see the overlay of the 2 distributions
new_df['training_hours'].hist(bins=50, ax=ax, color='green', density=True, alpha=0.8)

# %%
fig = plt.figure()
ax = fig.add_subplot(111)

# original data
df['training_hours'].plot.density(color='red')

# data after cca
new_df['training_hours'].plot.density(color='green')


# %%
fig = plt.figure()
ax = fig.add_subplot(111)

# original data
df['city_development_index'].hist(bins=50, ax=ax, density=True, color='red')

# data after cca, the argument alpha makes the color transparent, so we can
# see the overlay of the 2 distributions
new_df['city_development_index'].hist(bins=50, ax=ax, color='green', density=True, alpha=0.8)

# %%
fig = plt.figure()
ax = fig.add_subplot(111)

# original data
df['city_development_index'].plot.density(color='red')

# data after cca
new_df['city_development_index'].plot.density(color='green')

# %%
fig = plt.figure()
ax = fig.add_subplot(111)

# original data
df['experience'].hist(bins=50, ax=ax, density=True, color='red')

# data after cca, the argument alpha makes the color transparent, so we can
# see the overlay of the 2 distributions
new_df['experience'].hist(bins=50, ax=ax, color='green', density=True, alpha=0.8)

# %%
fig = plt.figure()
ax = fig.add_subplot(111)

# original data
df['experience'].plot.density(color='red')

# data after cca
new_df['experience'].plot.density(color='green')

# %%
temp = pd.concat([
            # percentage of observations per category, original data
            df['enrolled_university'].value_counts() / len(df),

            # percentage of observations per category, cca data
            new_df['enrolled_university'].value_counts() / len(new_df)
        ],
        axis=1)

# add column names
temp.columns = ['original', 'cca']

temp

# %%
temp = pd.concat([
            # percentage of observations per category, original data
            df['education_level'].value_counts() / len(df),

            # percentage of observations per category, cca data
            new_df['education_level'].value_counts() / len(new_df)
        ],
        axis=1)

# add column names
temp.columns = ['original', 'cca']

temp

# %%
new_df.head()    # so after looking all graph we finally conclude that we are after droping and before droping no distribution change so we can drop dataset

# %% [markdown]
# <h3 style="color:red;">so after looking all graph we finally conclude that we are after droping and before droping no distribution change so we can drop dataset</h3>

# %%
# now we concat df[target] with new_df
df1=pd.concat([df['target'],new_df.iloc[:,:]],axis=1)

# %%
df1.sample(5) # by running this cells many time we understand that when there is nan then in that row all atrributes are nan so we can drop nan also

# %%
df1.isna().mean()*100

# %%
df1.shape  # we understands why nan are coming because when we concat, df target has more rows than new_df rows so on that place the so the place where target has value is filled automatically by nan when we concat so we can easily drp all values 

# %%
dff=df1.dropna()

# %%
dff.isnull().mean()*100

# %%
dff.shape # now all row removed where there is missing values

# %%
dff.head()

# %%
print(dff['education_level'].unique()) # here natural rank order(phd>master>graduate>high school) so we will use ordianl encoding here 

# %%
print(dff['enrolled_university'].unique()) # we use one hot here beacuse if we use ordinal then our model would have been think that full time is much better than half time but this not true
# so one hot give equl importance

# %%
# in Feature Column we check distribution and skewness and imbalces in target beacuse imbalces directly dictates how model think
# let our 80 % of target is 0 and 20 % 1 so our model will be biased towards zero and predict zero mostly so our model became in accurate
import matplotlib.pyplot as plt 
import seaborn as sns

# %%
sns.countplot(x='target',data=dff)
plt.show()
print(dff['target'].value_counts(normalize=True))

# %%
# as we say 0 is 75% and 1 is 25 percent so our model will predict inaccurate result 
# to fix this we detail some data from majority

# %%
# 1. Separate the classes
df_minority = dff[dff['target'] == 1.0]
df_majority = dff[dff['target'] == 0.0]

# 2. Downsample the majority class (0) to match the number of minority samples (1)
df_majority_downsampled = df_majority.sample(n=len(df_minority), random_state=42)

# 3. Combine them back
dff_balanced = pd.concat([df_majority_downsampled, df_minority])

# 4. Shuffle the data so the model doesn't see all 0s followed by all 1s
dff_balanced = dff_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Verify the results
print(dff_balanced['target'].value_counts())
sns.countplot(x='target', data=dff_balanced)

# %%
dff_balanced.shape

# %%
dff_balanced.head()

# %%
# now are target column is balced now check skewness of feature


# %%
df=dff_balanced

# %%
df.shape

# %%
import seaborn as sns

# %%
sns.histplot(df['training_hours'],bins=30,kde=True,color='red')  # right skewd so we use log tranformation

# %%
sns.histplot(df['city_development_index'],bins=30,kde=True,color='red')
df['city_development_index'].skew() # this is not much skewd  when (lie between -0.5 to 0.5 then normal)

# %%
import scipy.stats as stats

# %%
stats.probplot(df['city_development_index'],dist="norm",plot=plt)  # so this is near normal so we need not do apply anything 

# %%
stats.probplot(df['experience'],dist="norm",plot=plt)

# %%
df['experience'].skew()  # since value come to be (-0.5 to 0.5 so  normal)

# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# %%
X=df.iloc[:,1:6]

# %%
y=df.iloc[:,0]

# %%
y

# %%
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# %%
X_train

# %%
X_train.shape,y_train.shape

# %%
X_train.iloc[:,0].unique()

# %%
#columns 
log_col=['training_hours']
ordinal_col=['education_level']
onehot_col=['enrolled_university']

# defined this columns names

# %%


# %%





# %%
# ordinal 
education_order=[['Graduate' 'Masters' 'High School' 'Phd' 'Primary School']]


# %%
#one hot 
one_hot_transformer=OneHotEncoder(handle_unknown='ignore')


# %%
def safe_log1p(x):
    return np.log1p(x.astype(float))

# %%
trf1=ColumnTransformer(transformers=[
    ('log',FunctionTransformer(safe_log1p),[4]),
    ('ord',OrdinalEncoder(categories=[['Graduate' ,'Masters', 'High School', 'Phd', 'Primary School']]),[2]),
    ('ohe',OneHotEncoder(handle_unknown='ignore',sparse_output=False),[1])
],remainder='passthrough')

# %%
from sklearn.linear_model import LogisticRegression  # we choose logistic regression becuase here target is distributed in category of 0 and 1 so it's classification problem

# %%
trf2=LogisticRegression()

# %%
pipe=Pipeline([
 
    ('trf1',trf1),
    ('trf2',trf2),
 
    
])

# %%


# %%
pipe.fit(X_train,y_train)

# %%
y_pred=pipe.predict(X_test)

# %%


# %%
print(y_pred)
y_pred.shape

# %%
test_input=np.array([0.8,"no_enrollment","Graduate",8.0,50],dtype=object).reshape(1,-1)  #since in our column tranformer we have use names thats why not wroking like this

# %%
print(pipe.predict(test_input))

# %%


# %%


# %%
# Transform X_train using only the preprocessing part of the pipe
X_train_transformed = pipe.named_steps['trf1'].transform(X_train)

# Convert to DataFrame to see the distribution
# Note: This assumes 'remainder=passthrough' was used
df_check = pd.DataFrame(X_train_transformed)

# To see the count of 1s in your One-Hot columns:
# If 'enrolled_university' was at index [1], its OHE version usually starts there
print(df_check.head())

# %%
import pickle

# %%
pickle.dump(pipe,open("pipe.pkl","wb"))

# %%



