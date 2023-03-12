#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Problem Statement -----> Customer Personality Analysis is a detailed analysis of a company's ideal customers. it helps a
# business to better understand its customer and makes it easier for then to modify product according to the specific needs, 
# behaviors and concerns of different types of customers.

# customer personality analysis helps a business to modify its prodcut based on its target customers from different types of 
# customer segmetns. for examples, instead of spending money to market a new product every customer in the company's databse,
# a company can analysis which customer segment is most likely to buy the product and then market the prodcut only on that 
# particular segment
# In[2]:


df = pd.read_csv("marketing_campaign(1).csv")


# In[3]:


df.shape


# In[4]:


df


# In[5]:


df.isnull().sum()


# In[6]:


data=df.drop(['Dt_Customer'],axis=1)


# In[7]:


data


# In[8]:


data.info()


# In[9]:


# def missing_vals(df): 
#     for i in df : 
#         print(f"{i}:{df[i].isnull().sum()} out of {len(df[i])}")
        
# missing_vals(df)


# In[10]:


data['Income'].describe().T


# In[11]:


from sklearn.impute import SimpleImputer
mean_imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')

data.Income = mean_imputer.fit_transform(data[["Income"]])


# In[12]:


df.isnull().sum()


# In[13]:


data.hist()


# In[14]:


# import statsmodels.api
# from statsmodels.stats.outliers_influence import variance_inflation_factor
  
# # VIF dataframe
# vif_df = pd.DataFrame()
# vif_df["feature"] = x.columns
  
# # calculating VIF for each feature
# vif_df["VIF"] = [variance_inflation_factor(x.values, k) 
#                            for k in range(len(x.columns))]
  
# print(vif_df)


# In[15]:


# sns.pairplot(data,hue='Response')


# In[16]:


data["Education"].value_counts() # catgorical values


# In[17]:


data['Education'].value_counts().plot(kind = 'pie' , autopct = '%1.1f%%' , shadow = True , explode = [0.1,0,0,0,0])
plt.title('Education level.')
fig = plt.gcf()
fig.set_size_inches(7,7)
plt.show()


# In[18]:


data["Marital_Status"].value_counts()


# In[19]:


per_life_data = data[['ID','Marital_Status','Kidhome','Teenhome']]


# In[20]:


data['Marital_Status'].value_counts().plot(kind = 'pie' , autopct = '%1.1f%%' , shadow = True  , explode = [0.1,0,0,0,0,0,0,0])
plt.title('Marital Status')
fig = plt.gcf()
fig.set_size_inches(7,7)
plt.show()


# In[21]:


from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()


# In[22]:


data['Education'] = LE.fit_transform(data['Education'])
data['Education'].value_counts()


# In[23]:


data['Marital_Status'] = LE.fit_transform(data['Marital_Status'])
data['Marital_Status'].value_counts()


# In[24]:


# Before Removeing outlier box plot
#Box plot 
figure=plt.figure(figsize=(20,20))
sns.boxplot(data=data,linewidth=1)
plt.xticks(rotation='vertical')
plt.show()


# In[25]:


#identifying outliers

def outliers(data,ft):
    Q1=data[ft].quantile(0.25)
    Q3=data[ft].quantile(0.75)
    
    IQR=Q3-Q1
    
    lower_bound=Q1-1.5*IQR
    upper_bound=Q3+1.5*IQR
    
    LS=df.index[(data[ft]<lower_bound)|(data[ft]>upper_bound)]
    
    return LS


# In[26]:


#removing outliers

def remove(data,LS):
    LS=sorted(set(LS))
    df1=data.drop(LS)
    
    return df1

print("old data",data.shape)


# In[27]:


index_list=[]
for feature in [
        'ID', 'Year_Birth', 'Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']:
               index_list.extend(outliers(data, feature))

df2=remove(data, index_list)
print("new data shape",df2.shape)


# In[28]:


# after removing outlier box plot
figure=plt.figure(figsize=(20,20))
sns.boxplot(data=df2,linewidth=1)
plt.xticks(rotation='vertical')
plt.show()


# In[29]:


x = df2.drop("Response",axis = 1)
y = df2[['Response']]


# In[30]:


x


# In[31]:


y


# In[32]:


# Fill diagonal and upper half with NaNs
corr = x.corr()
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
corr[mask] = np.nan
(corr
 .style
 .background_gradient(cmap='coolwarm', axis=None, vmin=-1, vmax=1)
 .highlight_null(null_color='#f1f1f1')  # Color NaNs grey
 .set_precision(2))


# In[33]:


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()


# In[34]:


x_scale = ss.fit_transform(x)


# In[35]:


x_scale


# In[36]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_scale, y, test_size = 0.20, random_state = 42)


# In[37]:


x_train.shape


# In[38]:


x_test.shape


# In[39]:


y_train.shape


# In[40]:


y_test.shape


# # Logistics Regression

# In[41]:


# Logistics Regression
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()


# In[42]:


LR.fit(x_train,y_train)


# In[43]:


LR.intercept_


# In[44]:


LR.coef_


# In[45]:


y_pred_train_LR = LR.predict(x_train)


# In[46]:


y_pred_train_LR


# In[47]:


y_pred_test_LR = LR.predict(x_test)


# In[48]:


y_pred_test_LR


# In[49]:


# confusion metrics fo train data for LR
from sklearn import metrics
print(metrics.confusion_matrix(y_train,y_pred_train_LR))
print(metrics.classification_report(y_train,y_pred_train_LR,digits=2))


# In[50]:


# confusion metrics fo test data for LR
from sklearn import metrics
print(metrics.confusion_matrix(y_test,y_pred_test_LR))
print(metrics.classification_report(y_test,y_pred_test_LR,digits=2))


# In[51]:


# Checking accuracy from Logistics Regression
from sklearn.metrics import accuracy_score
ac_train = accuracy_score(y_train,y_pred_train_LR)
print("log reg training accuracy :" ,ac_train)

ac_test = accuracy_score(y_test,y_pred_test_LR)
print("log reg test accuracy :" ,ac_test)


# # Decision tree 

# In[52]:


# Decision tree 
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion='gini', max_depth=17)


# In[53]:


dt.fit(x_train,y_train)
dt.tree_.node_count
dt.tree_.max_depth


# In[54]:


y_pred_train_DT = LR.predict(x_train)
y_pred_train_DT


# In[55]:


y_pred_test_DT = LR.predict(x_test)
y_pred_test_DT


# In[56]:


# confusion metrics of train data for decision tree
from sklearn import metrics
print(metrics.confusion_matrix(y_train,y_pred_train_DT))
print(metrics.classification_report(y_train,y_pred_train_DT,digits=2))


# In[57]:


# confusion metrics of test data for decision tree
from sklearn import metrics
print(metrics.confusion_matrix(y_test,y_pred_test_DT))
print(metrics.classification_report(y_test,y_pred_test_DT,digits=2))


# In[58]:


# Checking accuracy from Decision tree
from sklearn.metrics import accuracy_score
ac1_train = accuracy_score(y_train,y_pred_train_DT)
print("DT training accuracy :" ,ac1_train)

ac1_test = accuracy_score(y_test,y_pred_test_DT)
print("DT test accuracy :" ,ac1_test)


# # Random Forest 

# In[59]:


# Random Forest 
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier()
RF.fit(x_train,y_train)


# In[60]:


y_pred_train_RF = RF.predict(x_train)
y_pred_test_RF = RF.predict(x_test)


# In[61]:


# confusion metrics of train data for RF
print(metrics.confusion_matrix(y_train,y_pred_train_RF))
print(metrics.classification_report(y_train,y_pred_train_RF,digits=2))


# In[62]:


# confusion metrics of test data for RF
print(metrics.confusion_matrix(y_test,y_pred_test_RF))
print(metrics.classification_report(y_test,y_pred_test_RF,digits=2))


# In[63]:


# cecking accuracy for Randome Forest
from sklearn.metrics import accuracy_score
ac2_train = accuracy_score(y_train,y_pred_train_RF)
print("Random forest training accuracy :" ,ac2_train)

ac2_test = accuracy_score(y_test,y_pred_test_RF)
print("Random forest test accuracy :" ,ac2_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




