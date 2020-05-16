#!/usr/bin/env python
# coding: utf-8

# # Marketing Analytics Final Case
# ## Qingyin Cheng (qc650)

# In[198]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[199]:


#read the subscriber file (most important one) and do some eda
subscriber = pd.read_pickle(r'subscribers')


# In[200]:


subscriber.shape


# In[201]:


subscriber.head()


# In[202]:


subscriber.columns


# In[203]:


subscriber[ 'weekly_consumption_hour'].unique()


# In[204]:


subscriber.isnull().sum()


# In[205]:


#read the engagement file
engagement = pd.read_pickle(r'engagement')


# In[206]:


engagement.head()


# In[207]:


#read the service file which contains data of churned 
customer_service = pd.read_pickle(r'customer_service_reps')


# In[208]:


customer_service_sorted = customer_service.sort_values(by=['account_creation_date'])


# In[209]:


customer_service_sorted.head()


# In[210]:


customer_service_sorted.shape


# In[211]:


customer_service_sorted = customer_service_sorted[['subid','current_sub_TF']]


# In[212]:


churn_data = customer_service_sorted.drop_duplicates(keep = 'last')


# In[213]:


#find the subsription status of customers
churn_data.head()


# In[214]:


#merge churned data with subscriber file
merged_sub = subscriber.merge(churn_data, how = 'left',left_on='subid', right_on='subid')


# In[215]:


merged_sub_churn = merged_sub[['current_sub_TF']]


# In[216]:


merged_sub[['current_sub_TF']] = merged_sub_churn.fillna(True)


# In[217]:


merged_sub.head()


# In[218]:


#deal with missing values (numerical)
merged_sub[['num_weekly_services_utilized','weekly_consumption_hour','num_ideal_streaming_services','revenue_net','join_fee']] = merged_sub[['num_weekly_services_utilized','weekly_consumption_hour','num_ideal_streaming_services','revenue_net','join_fee']].fillna(0)


# In[219]:


#deal with missing values (categorical)
merged_sub[['package_type','preferred_genre','intended_use','male_TF','attribution_survey','payment_type']] = merged_sub[['package_type','preferred_genre','intended_use','male_TF','attribution_survey','payment_type']].fillna('Unknown')


# In[220]:


#filter abnormal numbers
merged_sub = merged_sub[merged_sub['age']<=100] 


# In[221]:


merged_sub[['age']] = merged_sub[['age']].fillna(np.average(merged_sub['age'].dropna()))


# In[222]:


merged_sub = merged_sub[merged_sub.columns.drop('op_sys')]


# In[223]:


drop_list = ['creation_until_cancel_days','country', 'account_creation_date', 'trial_end_date', 'language'] 
for i in drop_list:
    merged_sub = merged_sub[merged_sub.columns.drop(i)]


# In[224]:


merged_sub['plan_type'].unique()


# In[225]:


merged_sub_base_uae_14_day_trial = merged_sub.loc[merged_sub['plan_type'] == 'base_uae_14_day_trial']
merged_sub_low_uae_no_trial = merged_sub.loc[merged_sub['plan_type'] == 'low_uae_no_trial']
merged_sub_base_uae_no_trial_7_day_guarantee = merged_sub.loc[merged_sub['plan_type'] == 'base_uae_no_trial_7_day_guarantee']


# In[226]:


merged_sub_base_uae_14_day_trial = merged_sub_base_uae_14_day_trial[['current_sub_TF','plan_type']]
merged_sub_low_uae_no_trial = merged_sub_low_uae_no_trial[['current_sub_TF','plan_type']]
merged_sub_base_uae_no_trial_7_day_guarantee = merged_sub_base_uae_no_trial_7_day_guarantee[['current_sub_TF','plan_type']]


# In[227]:


merge_AB = pd.concat([merged_sub_base_uae_14_day_trial , merged_sub_low_uae_no_trial], axis=0)
merge_AC = pd.concat([merged_sub_base_uae_14_day_trial , merged_sub_base_uae_no_trial_7_day_guarantee], axis=0)


# In[228]:


merge_AB= pd.get_dummies(merge_AB, prefix=['plan_type'],drop_first= True)
merge_AC= pd.get_dummies(merge_AC, prefix=['plan_type'],drop_first= True)


# ## AB Test

# ### 14-day and no trial

# In[229]:


import imblearn
print(imblearn.__version__)
from imblearn.under_sampling import RandomUnderSampler
undersample = RandomUnderSampler(sampling_strategy='majority')
X_under, y_under = undersample.fit_resample(merge_AB[['plan_type_low_uae_no_trial']],merge_AB[['current_sub_TF']])


# In[230]:


merge_AB_under = pd.concat([X_under, y_under], axis=1)


# In[231]:


merge_AB_under.head()


# In[232]:


import HW1 as ABTesting


# In[233]:


groupA =  merge_AB_under.loc[merge_AB_under['plan_type_low_uae_no_trial'] == 0]
groupB =  merge_AB_under.loc[merge_AB_under['plan_type_low_uae_no_trial'] == 1] 


# In[234]:


A_list = list(groupA['current_sub_TF'])
B_list = list(groupB['current_sub_TF'])


# In[235]:


import scipy
norm = scipy.stats.norm()
ABTesting.t_test(A_list, B_list,0.95)


# therefore, the no trial group is significantly less likely to churn statistically with 95% confidence

# ### 14-day and 7-day

# In[236]:


undersample = RandomUnderSampler(sampling_strategy='majority')
X_under, y_under = undersample.fit_resample(merge_AC[['plan_type_base_uae_no_trial_7_day_guarantee']],merge_AC[['current_sub_TF']])


# In[237]:


merge_AC_under = pd.concat([X_under, y_under], axis=1)
groupA =  merge_AC_under.loc[merge_AC_under['plan_type_base_uae_no_trial_7_day_guarantee'] == 0]
groupC =  merge_AC_under.loc[merge_AC_under['plan_type_base_uae_no_trial_7_day_guarantee'] == 1] 
A_list = list(groupA['current_sub_TF'])
C_list = list(groupC['current_sub_TF'])
norm = scipy.stats.norm()
ABTesting.t_test(A_list, B_list,0.95)


# therefore, the 7-day trial group is significantly less likely to churn statistically with 95% confidence

# # Customer Segmentation
# 

# In[238]:


engagement['date'] = pd.to_datetime(engagement['date'])
engagement
feature_list = ['app_opens', 'cust_service_mssgs',
       'num_videos_completed', 'num_videos_more_than_30_seconds',
       'num_videos_rated', 'num_series_started']


# In[239]:


len(engagement['subid'].unique())


# In[240]:


engagement['payment_period'].value_counts()


# In[241]:


customer_average = pd.pivot_table(engagement, values=feature_list, index='subid', aggfunc=np.mean)
customer_average.reset_index(drop=False, inplace=True)
customer_average.head()


# In[242]:


sub_dummy = subscriber[['package_type', 'preferred_genre', 'male_TF']]

sub_dummy = pd.get_dummies(sub_dummy)

sub_dummy.head()


# In[243]:


sub_seg = pd.merge(subscriber[['subid']], sub_dummy, left_index=True, right_index = True, how='left')

sub_seg.shape


# In[244]:


sub_seg = pd.merge(sub_seg, customer_average, on= 'subid', how='left')

sub_seg.shape


# In[245]:


sub_seg.dropna(axis=0, inplace=True)

sub_seg.set_index('subid',inplace=True)

sub_seg.shape


# In[246]:


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def fitting(df):
    Sum_of_squared_distances = []
    K = range(1,15)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(df)
        Sum_of_squared_distances.append(km.inertia_)
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()
    return 


# In[247]:


fitting(sub_seg)


# In[248]:


kmeans = KMeans(n_clusters = 3, random_state=0).fit(sub_seg)


# In[249]:


a = list(kmeans.cluster_centers_)

seg_result = pd.DataFrame(a, columns=sub_seg.columns)

seg_result


# ## Churn Model

# In[250]:


merged_sub = pd.get_dummies(merged_sub, prefix=['package_type', 'preferred_genre', 'intended_use','male_TF',  'attribution_technical', 'attribution_survey', 'plan_type',  'payment_type'], columns=['package_type', 'preferred_genre', 'intended_use','male_TF',  'attribution_technical', 'attribution_survey', 'plan_type',  'payment_type'])


# In[251]:


merged_sub.head()


# In[252]:


#merged_sub.to_csv('export_dataframe.csv',index=False)


# In[253]:


from sklearn.model_selection import train_test_split
X = merged_sub[merged_sub.columns.drop('current_sub_TF')]
X = X[X.columns.drop('subid')]
y = merged_sub['current_sub_TF'].ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[254]:


from sklearn.ensemble import GradientBoostingClassifier


# In[255]:


clf = GradientBoostingClassifier()


# In[256]:


clf.fit(X_train,y_train)


# In[257]:


y_pred = clf.predict(X_test)
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[258]:


y_pred = clf.predict(X)


# In[259]:


y_pred = y_pred.astype(int)


# In[260]:


revenue = np.array(X['revenue_net'])


# In[261]:


revenue 


# In[262]:


expected_revenue = np.dot(y_pred,revenue)


# In[263]:


expected_revenue


# In[264]:


expected_revenue/len(y_pred)


# In[265]:


len(y_pred)


# In[ ]:




