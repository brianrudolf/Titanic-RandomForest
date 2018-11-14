
# coding: utf-8

# In[1]:


#Import useful packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


#For easier reading of numbers
pd.set_option('display.precision',2)


# In[3]:


#Import the training data
data = pd.read_csv('train.csv')
data.head(5)


# In[4]:


#For this Random Forest Classifier we decide which columns to make use of and discard the rest
#We are deciding not to look at the names of the passengers, the ticket, nor the cabin
data_train = data[['Survived', 'Pclass',  'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

#View the completeness of the data (this is only looking for NaNs, None, Null)
data_train.count()


# In[5]:


#The above shows that the age of 177 passengers is missing, and that the embarkation points are missing for two passengers
#Missing data can be resolved by either suplimenting an relatively neutral new value (mean, mode, etc.), or by dropping those data points

#Resolve missing age data by assigning mean value to passengers missing the age information
data_train = data_train.fillna(value={'Age' : np.mean(data_train['Age'])})
data_train.count()


# In[6]:


#Resolve missing embarkation data by removing data (only two data points)
#dropna() will remove the rows with NaNs, which we know are in the 'Embarked' col
data_train = data_train.dropna()


# In[7]:


data_train.count()


# In[8]:


#Our Random Forest Classifier requires numerical data as input to the model, so we need to use "one-hot-encoding"

#Resolve 'Sex' and 'Embarked' columns non-numerical data using Pandas' wonderful .get_dummies()
data_train = pd.get_dummies(data_train)


# In[9]:


data_train.head()


# In[10]:


#Define the Random Forest Classifier Model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)


# In[11]:


#train / fit the model
model.fit(data_train.drop(labels='Survived', axis=1).values, data_train['Survived'].values)


# In[12]:


estimator = model.estimators_[15]


# In[13]:


from sklearn.tree import export_graphviz
# Export as dot file
export_graphviz(estimator, out_file='tree.dot', 
                feature_names = data_train.drop(labels='Survived', axis=1).columns.values,
                class_names = ['Survived', 'Perished'],
                rounded = True, proportion = False, 
                precision = 2, filled = True)


# In[14]:


import graphviz
with open("tree.dot") as f:
    dot_graph = f.read()


# In[15]:


tree = graphviz.Source(dot_graph, format='pdf')


# In[16]:


tree.render(filename="Titanic-Tree")


# In[17]:


data_test = pd.read_csv('test.csv')
data_test.count()


# In[18]:


#Collect relevant data, including PassengerId
data_test = data_test[['PassengerId', 'Pclass',  'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
#Have a look at the completeness of the data (this is only looking for Nans)
data_test.count()


# In[19]:


#Set missing age and fare information with their mean values
data_test = data_test.fillna(value={'Age':np.mean(data_test['Age']), 
                                    'Fare':np.mean(data_test['Fare'])})
data_test = pd.get_dummies(data_test)
data_test.count()


# In[20]:


#Predict the survival of the passenger by adding a new data column
data_test['Survived'] = model.predict(data_test.drop(labels='PassengerId', axis=1))


# In[21]:


#Print a csv that is formatted for submission to the Kaggle competition
#CSV won't contain an index column, only the PassengerId and whether or not we predict that they survived
data_test.to_csv('submission_RFC_improved.csv',index=False,columns=['PassengerId', 'Survived'])

