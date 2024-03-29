#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


data = pd.read_csv('D:/Ds Intenship/credit card fraud detection/archive (5)/creditcard.csv')


# In[4]:


pd.options.display.max_columns = None


# # 1. Display Top 5 Rows of The Dataset

# In[5]:


data.head()


# # 2. Check Last 5 Rows of The Dataset

# In[6]:


data.tail()


# # 3. Find Shape of Our Dataset (Number of Rows And Number of Columns)

# In[7]:


data.shape


# In[8]:


print("Number of Rows",data.shape[0])
print("Number of Columns",data.shape[1])


# # 4. Get Information About Our Dataset Like Total Number Rows, Total Number of Columns, Datatypes of Each Column And Memory Requirement

# In[9]:


data.info()


# # 5. Check Null Values In The Dataset

# In[10]:


data.isnull().sum()


# # Feature Scaling

# In[11]:


from sklearn.preprocessing import StandardScaler


# In[12]:


sc = StandardScaler()
data['Amount']=sc.fit_transform(pd.DataFrame(data['Amount']))


# In[13]:


data.head()


# In[14]:


data = data.drop(['Time'],axis=1)


# In[15]:


data.head()


# In[16]:


data.shape


# In[17]:


data.duplicated().any()


# # Let's Remove Duplicated Values

# In[18]:


data = data.drop_duplicates()


# In[19]:


data.shape


# In[20]:


284807- 275663


# # 6. Not Handling Imbalanced

# In[21]:


data['Class'].value_counts()


# In[22]:


import seaborn as sns


# In[23]:


sns.countplot(data['Class'])


# # 7. Store Feature Matrix In X And Response (Target) In Vector y

# In[24]:


X = data.drop('Class',axis=1)
y = data['Class']


# # 8. Splitting The Dataset Into The Training Set And Test Set

# In[25]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,
                                                 random_state=42)


# # 9. Handling Imbalanced Dataset

# # Undersampling

# In[26]:


normal = data[data['Class']==0]
fraud = data[data['Class']==1]


# In[27]:


normal.shape


# In[28]:


fraud.shape


# In[29]:


normal_sample=normal.sample(n=473)


# In[30]:


normal_sample.shape


# In[31]:


new_data = pd.concat([normal_sample,fraud],ignore_index=True)


# In[32]:


new_data['Class'].value_counts()


# In[33]:


new_data.head()


# In[34]:


X = new_data.drop('Class',axis=1)
y = new_data['Class']


# In[35]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,
                                                 random_state=42)


# # 10. Logistic Regression

# In[36]:


from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(X_train,y_train)


# In[37]:


y_pred1 = log.predict(X_test)


# In[38]:


from sklearn.metrics import accuracy_score


# In[39]:


accuracy_score(y_test,y_pred1)


# In[40]:


from sklearn.metrics import precision_score,recall_score,f1_score


# In[41]:


precision_score(y_test,y_pred1)


# In[42]:


recall_score(y_test,y_pred1)


# In[43]:


f1_score(y_test,y_pred1)


# # 11. Decision Tree Classifier

# In[44]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)


# In[45]:


y_pred2 = dt.predict(X_test)


# In[46]:


accuracy_score(y_test,y_pred2)


# In[47]:


precision_score(y_test,y_pred2)


# In[48]:


recall_score(y_test,y_pred2)


# In[49]:


f1_score(y_test,y_pred2)


# # 12. Random Forest Classifier

# In[50]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train,y_train)


# In[51]:


y_pred3 = rf.predict(X_test)


# In[52]:


accuracy_score(y_test,y_pred3)


# In[53]:


precision_score(y_test,y_pred3)


# In[54]:


recall_score(y_test,y_pred3)


# In[55]:


f1_score(y_test,y_pred3)


# In[56]:


final_data = pd.DataFrame({'Models':['LR','DT','RF'],
              "ACC":[accuracy_score(y_test,y_pred1)*100,
                     accuracy_score(y_test,y_pred2)*100,
                     accuracy_score(y_test,y_pred3)*100
                    ]})


# In[57]:


final_data


# In[58]:


sns.barplot(final_data['Models'],final_data['ACC'])


# # Oversampling

# In[59]:


X = data.drop('Class',axis=1)
y = data['Class']


# In[60]:


X.shape


# In[61]:


y.shape


# In[62]:


from imblearn.over_sampling import SMOTE


# In[63]:


pip install imblearn


# In[64]:


X_res,y_res = SMOTE().fit_resample(X,y)


# In[65]:


y_res.value_counts()


# In[66]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_res,y_res,test_size=0.20,
                                                 random_state=42)


# # 10. Logistic Regression

# In[67]:


log = LogisticRegression()
log.fit(X_train,y_train)


# In[68]:


y_pred1 = log.predict(X_test)


# In[69]:


accuracy_score(y_test,y_pred1)


# In[70]:


precision_score(y_test,y_pred1)


# In[71]:


recall_score(y_test,y_pred1)


# In[72]:


f1_score(y_test,y_pred1)


# # 11. Decision Tree Classifier

# In[73]:


dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)


# In[74]:


y_pred2 = dt.predict(X_test)


# In[75]:


accuracy_score(y_test,y_pred2)


# In[76]:


precision_score(y_test,y_pred2)


# In[77]:


recall_score(y_test,y_pred2)


# In[78]:


f1_score(y_test,y_pred2)


# # 12. Random Forest Classifier

# In[79]:


rf = RandomForestClassifier()
rf.fit(X_train,y_train)


# In[80]:


y_pred3 = rf.predict(X_test)


# In[81]:


accuracy_score(y_test,y_pred3)


# In[82]:


precision_score(y_test,y_pred3)


# In[83]:


recall_score(y_test,y_pred3)


# In[84]:


f1_score(y_test,y_pred3)


# In[85]:


final_data = pd.DataFrame({'Models':['LR','DT','RF'],
              "ACC":[accuracy_score(y_test,y_pred1)*100,
                     accuracy_score(y_test,y_pred2)*100,
                     accuracy_score(y_test,y_pred3)*100
                    ]})


# In[86]:


final_data


# In[87]:


sns.barplot(final_data['Models'],final_data['ACC'])


# # Save The Model

# In[88]:


rf1 = RandomForestClassifier()
rf1.fit(X_res,y_res)


# In[89]:


import joblib


# In[90]:


joblib.dump(rf1,"credit_card_model")


# In[91]:


model = joblib.load("credit_card_model")


# In[92]:


pred = model.predict([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])


# In[93]:


if pred == 0:
    print("Normal Transcation")
else:
    print("Fraudulent Transcation")

