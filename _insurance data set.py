#!/usr/bin/env python
# coding: utf-8

# # Problem Statement 

# .The goal of the project is to give people an estimate of how much health risk they have based on their indivisual health
# sitution.
# 
# .After that customers can work with any health insurance carrier and its plans and perks while keeping the projected cost 
# from our model in mind.
# 
# .this can assist a person in concentrating on the health side of an insurance policy rather than the in effective part.

# # Overview about the project

# Week 1 - Database(MongoDB),problem Statement and EDA
# 
# Week 2 - Feature Engineering and introduction to Git and Github
# 
# Week 3 - OOPS
# 
# Week 4 - small OOPS project and introduction to POC(Proof of Concept)
# 
# Week 5 - 8 diffrent phases of training Pipline of project
# 
# 
#         - Data Ingestion
#         - Data Validation
#         - Data Transformation
#         - Model Training(Storing the best model in AWS S3 Bucket)

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


dataframe = pd.read_csv("insurance.csv")


# In[3]:


print(dataframe)


# In[4]:


dataframe.tail()


# In[5]:


dataframe.columns


# In[6]:


dataframe.shape


# In[7]:


dataframe.isnull().sum().sum()


# In[8]:


dataframe.duplicated().sum()


# In[9]:


dataframe = dataframe.drop_duplicates()
print(dataframe)


# In[10]:


dataframe.shape


# In[11]:


dataframe.duplicated().sum()


# # EDA(Exploratory Data Analysis)

# In[12]:


for feature in dataframe.columns:
#     print(feature)
    if dataframe[feature].dtype !='O':
        print(feature)


# In[13]:


numerical_features = [feature for feature in dataframe.columns if dataframe[feature].dtype !='O']
print(f"{numerical_features}")


# In[14]:


for feature in dataframe.columns:
#     print(feature)
    if dataframe[feature].dtype =='O':
        print(feature)


# In[15]:


categorical_features = [feature for feature in dataframe.columns if dataframe[feature].dtype =='O']
print(f"{categorical_features}")


# In[16]:


dataframe.info()


# In[17]:


# proportion of count data on categorical_features # in percentage
for col in categorical_features:
    print(dataframe[col].value_counts(normalize=True)*100)
    print('--'*50)


# In[18]:


# # proportion of count data on categorical_features # without percentage
# for col in categorical_features:
#     print(dataframe[col].value_counts())
#     print('--'*50)


# # Univariate Analysis

# Numerical feature

# In[19]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize = (10,10))
plt.suptitle("Unuvariate Analysis of Numerical features", fontsize=20,fontweight="bold")

for i in range(0,len(numerical_features)):
    plt.subplot(2,2,i+1)
    sns.kdeplot(x=dataframe[numerical_features[i]],shade=True,color='b')
    plt.xlabel(numerical_features)
    plt.tight_layout()
# saving the image
# plt.savefig("Univariate_Num.png")


# # checking multicollinearity in numerical feature

# In[20]:


dataframe[list(dataframe.columns)[1:]].corr()


# In[21]:


# dataframe.corr()


# In[22]:


sns.heatmap(dataframe.corr(),cmap="CMRmap_r",annot=True,fmt='.0%')
plt.show()


# In[23]:


sns.heatmap(dataframe.corr(),cmap="CMRmap_r",annot=True)
plt.show()


# # Outliers and histplot

# In[24]:


clr1 = ['#1E90FF','#DC143C']
fig,ax = plt.subplots(3,2,figsize=(10,16))
fig.suptitle('Distribution of Numerical Feature ',color = '#3C3744',fontsize=20,fontweight='bold')

for i,col in enumerate(numerical_features):
    sns.boxplot(data=dataframe,palette=clr1,ax=ax[i,0])
    ax[i,0].set_title(f'Boxplot of {col}',fontsize=12)
    sns.histplot(data=dataframe,x=col,bins=50,kde=True,multiple='stack',palette=clr1,ax=ax[i,1])
    ax[i,1].set_title(f"Histogram of {col}",fontsize=14)
    fig.tight_layout()


# In[ ]:


# !pip install dtale


# In[25]:


import dtale 


# In[26]:


# d = dtale.show(dataframe)
# d.open_browser()


# In[27]:


dataframe.head()


# # Feature Engineering

# In[28]:


# # check dupications
dataframe.duplicated().sum()


# In[29]:


dataframe.drop_duplicates(inplace=True)


# In[30]:


dataframe.duplicated().sum()


# # variance Inflation factor(VIF)

# In[31]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[32]:


def compute_vif(features,df): # amde fucntion 
    X = df[features]
    # calculation of variance  inflation requires a constant
    X['intercept'] = 1
    
    # create a dataframe to store the VIF values
    vif = pd.DataFrame()
    vif['variable'] = X.columns
    vif['vif'] = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
    vif = vif[vif['variable'] !='intercept']
    return vif
   


# In[33]:


import warnings
warnings.filterwarnings('ignore')


# In[34]:


compute_vif(numerical_features,dataframe)


# Note -- As the VIF score is less then the range of 5 to 10, it indicates there is no multi-collinearity in this data 

# # outlier treatment and Capping it

# In[35]:


dataframe.shape


# In[36]:


def detect_outliers(col):
    # finding the IQR
    percentile75 = dataframe[col].quantile(0.75)
    percentile25 = dataframe[col].quantile(0.25)
    print('\n ###',col,"###")
    print('percentile75= ',percentile75)
    print('percentile25= ',percentile25)
    IQR = percentile75 - percentile25
    ## finding the upper limit and lower limit
    upper_limit = percentile75 + 1.5*IQR
    lower_limit = percentile25 - 1.5*IQR
    print('upper_limit=',upper_limit)
    print('lower_limit=',lower_limit)
    
    
    dataframe.loc[(dataframe[col]>upper_limit),col] = upper_limit
    dataframe.loc[(dataframe[col]<lower_limit),col] = lower_limit
    
    return dataframe


# In[37]:


for col in numerical_features:
    detect_outliers(col)


# In[38]:


dataframe.describe()


# In[39]:


dataframe.columns


# In[40]:


import seaborn as sns
sns.pairplot(data=dataframe,hue='smoker')


# # split x and y

# In[41]:


from sklearn.model_selection import train_test_split
import pandas as pd


# In[42]:


X = dataframe.drop(['expenses'],axis=1)
y = dataframe['expenses']


# In[43]:


# create Column Transformer

num_features = X.select_dtypes(exclude='object').columns 
cat_features = X.select_dtypes(include='object').columns


# In[44]:


num_features


# In[45]:


cat_features


# In[46]:


from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
numeric_transformer = StandardScaler()
cat_transformer = OneHotEncoder()

preprocessor = ColumnTransformer(
[
    ('Treatment_of_cat_features',cat_transformer,cat_features),
    ('Trestment_of_numeriacal_features',numeric_transformer,num_features)
])


# In[47]:


X = preprocessor.fit_transform(X)


# In[48]:


from sklearn.metrics import mean_squared_error,r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


# In[49]:


models = {
    "Linear Regression ": LinearRegression(),
    "Lasso ": Lasso(),
    "Redge ": Ridge(),
    "K-Neighbors Regressor ": KNeighborsRegressor(),
    "Decision Tree ": DecisionTreeRegressor(),
    "Random Forest Regressor ": RandomForestRegressor(),
    "XGBRegressor ": XGBRegressor(),
    "CatBoost Regressor ": CatBoostRegressor(verbose=False),
    "AdaBoost Regressor ": AdaBoostRegressor()
}


# # creat a fucntion  to evaluate the model

# In[50]:


import numpy as np
def evaluate_model(true,predicted):
    mae = mean_absolute_error(true,predicted)
    mse = mean_squared_error(true,predicted)
    rmse = np.sqrt(mean_squared_error(true,predicted))
    r2_square = r2_score(true,predicted)
    
#     adjucted_r2 = <your code>
    return mae,mse,r2_square


# In[51]:


# seperating the X and Y into train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=41)
X_train.shape,X_test.shape


# In[52]:


model_list =[]
r2 = []

for i in range(len(list(models))):
    model = list(models.values())[i]
   # print(model)

    model.fit(X_train,y_train)
    # making predictins
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Evaluate the model predictions
    
    model_train_mae,model_train_mse,model_train_r2 = evaluate_model(y_train,y_train_pred)
    model_test_mae,model_test_mse,model_test_r2 = evaluate_model(y_test,y_test_pred)
    
    print(list(models.keys())[i])
    model_list.append(list(models.keys())[i])
    
    
    print('Model Performance for training set ')
    print(f"Mean_Absolute_Error = {model_train_mae}")
    print(f"Mean_squared_Error = {model_train_mse}")
    print(f"R2_score = {model_train_r2}")
    
    print('------------------------------------------')
    
    print('Model Performance for test set ')
    print(f"Mean_Absolute_Error = {model_test_mae}")
    print(f"Mean_squared_Error = {model_test_mse}")
    print(f"R2_score = {model_test_r2}")
    
    print('###################################################')
    
    r2.append(model_test_r2)


# # Result of All Models

# In[53]:


pd.DataFrame(list(zip(model_list,r2)),columns=['Model Name',"R2_score"]).sort_values(by=['R2_score'],ascending = False)


# # Assignment-2

# ## do the hyperparameters for top 3

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




