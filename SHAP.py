#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install shap


# In[2]:


pip install xgboost


# In[3]:


pip install matplotlib


# In[1]:


#importing required libraries
import xgboost
import shap
import matplotlib
shap.initjs()


# In[2]:


#loading the dataset
X, y = shap.datasets.boston()


# In[3]:


#displaying the dataset
X


# In[ ]:


#:Attribute Information (in order):
#        - CRIM     per capita crime rate by town
#        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
#        - INDUS    proportion of non-retail business acres per town
#        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
#        - NOX      nitric oxides concentration (parts per 10 million)
#        - RM       average number of rooms per dwelling
#        - AGE      proportion of owner-occupied units built prior to 1940
#        - DIS      weighted distances to five Boston employment centres
#        - RAD      index of accessibility to radial highways
#        - TAX      full-value property-tax rate per $10,000
#        - PTRATIO  pupil-teacher ratio by town
#        - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
#        - LSTAT    % lower status of the population


# In[4]:


#training of model
d_param = {
    "learning_rate": 0.01
}

model = xgboost.train(params=d_param,
                      dtrain=xgboost.DMatrix(X, label=y), 
                      num_boost_round=100)


# In[5]:


#creating an explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)


# In[6]:


X.columns


# In[7]:


feature_list=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
       'PTRATIO', 'B', 'LSTAT']


# In[8]:


#plot-1:waterfall
shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0], feature_names=feature_list,max_display=14)


# In[9]:


#plot-2:force_plot
#single prediction explainer
i = 0
shap.force_plot(explainer.expected_value, shap_values[i,:], X.iloc[i,:])


# In[10]:


#all predictions explainer
shap.summary_plot(shap_values, X, plot_type="violin")


# In[11]:


#variable importance
shap.summary_plot(shap_values, X, plot_type="bar")


# In[ ]:




