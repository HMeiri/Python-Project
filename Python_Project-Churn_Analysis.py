# %% [markdown]
# # Python Analysis Project
# ## By: Hedva Meiri

# %% [markdown]
#  ## Data preparation

# %%

# Load data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

churn = pd.read_csv('churn.csv')

# %%

# Check shape and look at a few rows
print(churn.shape)
churn.head()

# %%
# List columns
churn.columns

# %%
# Check for duplicates, drop as needed
print(churn.duplicated().value_counts())
n = len(pd.unique(churn['customerID']))
print('Number of unique values :', n)

# %%
# Lowercase letters for column names
churn = churn.rename(columns=str.lower)         

# Put aside customerID then drop
cst = churn['customerid']
churn.drop(['customerid'], axis=1, inplace=True)
print(churn.isna().sum(), '\n\n')               # Check for NULL values
print(churn.columns, '\n\n')

# Check data types 
print(churn.dtypes)

# %%
churn.shape

# %%
# Check valuecounts of all columns for invalid data and a list of entries.

columns_to_loop = churn.columns[0:]
for col in columns_to_loop:
    print(f'Value counts for column ''{col}'':')
    print(churn[col].value_counts())
    print('')


# %% [markdown]
# # Fix datatypes:

# %%
print(churn.dtypes)

# %% [markdown]
# ### Gender

# %%
# Change gender to string
churn['gender'] = churn.gender.astype(str)          

# Change partner, dependents, phoneservice, internetservice, paperlessbilling and churn to bool then int
pd.set_option('future.no_silent_downcasting', True)                                            

l1 = ['partner', 'dependents', 'phoneservice', 'internetservice', 'paperlessbilling', 'churn']
for item in l1:
    churn[item] = churn[item].str.strip()
    churn[item] = churn[item].map(lambda x: 1 if x == 'Yes' else (0 if x == 'No' else 1))              
    churn[item] = churn[item].astype('int') 

# Rename contract to contracttype
churn.rename(columns = {'contract': 'contracttype'}, inplace=True)


# %% [markdown]
# ### Total Charges

# %%
# Take a closer look at totalcharges
churn['totalcharges'].value_counts().head()        # Shows 11 cells with empty str

# %%
# totalcharges should equal tenure (the data is in months) * monthlycharges
churn['temp'] = churn['tenure'] * churn['monthlycharges']
churn.temp.value_counts().head(5)                           # Here we see 11 cells with a value of 0.00

# %%
# In order to check the correlation between temp and totalcharges, totalcharges must be changed to float
    # Fill in ' ' data values with 0
    # Change totalcharges to float

churn['totalcharges'] = churn['totalcharges'].replace({' ': 0})
churn['totalcharges'] = churn['totalcharges'].astype('float')


# %%
# Checking the correlation supports the decision to replace ' ' values with 0.00

print(churn['temp'].corr(churn['totalcharges']))

x = [churn['temp']]
y = [churn['totalcharges']]

plt.scatter(x, y, color='skyblue')

plt.show()

# %%
# Drop the temp column and check df columns again
churn.drop(['temp'], axis=1, inplace=True)
print([churn.columns])

# %%
print(churn.dtypes)


# %% [markdown]
# # EDA - Exploratory Data Analysis

# %%
# Basic attributes of the dataset
print(churn.shape)
print(churn.columns)

# %%
# Check label distribution - about 75% of the cases don't churn and 25% do
# Set one color for 0, another for 1

plt.style.use('fast')
sns.set_palette('hls')

fig, axs = plt.subplots(1, 2, figsize=(10, 4))

binary_values = [0, 1]
colors = ['green' if val == 0 else 'yellow' for val in binary_values]     # Define colors based on binary values

print(churn.churn.value_counts(), '\n\n', churn.churn.value_counts(normalize=True), '\n\n')

churn.churn.value_counts(normalize=True).plot(kind='pie', title='Churn', colors=colors, autopct='%1.1f%%', ax=axs[0])

churn_mean = churn['churn'].mean()

axs[1].pie([1 - churn_mean, churn_mean], labels=['No Churn', 'Churn'], colors=colors, autopct='%1.1f%%')
axs[1].set_title('Churn')

plt.show()

# %% [markdown]
# ## Features: dependents, partner

# %%
# Examine partner and dependents together
print('dependents_sum: ' + str(churn.dependents.sum()), end='\n')
print('dependents_mean: ' + str(churn.dependents.mean().round(3)), end='\n')
print('dependents_std: ' + str(churn.dependents.std().round(3)), end='\n')
print('dependents_median: ' + str(churn.dependents.median().round(3)), end='\n\n')  

print('partner_sum: ' + str(churn.partner.sum()), end='\n')
print('partner_mean: ' + str(churn.partner.mean().round(3)), end='\n')
print('partner_std: ' + str(churn.partner.std().round(3)), end='\n')
print('partner_median: ' + str(churn.partner.median().round(3)), end='\n')  

# Check churn and dependents stats
churn.groupby('churn').dependents.agg(['sum', 'describe', 'median'])

# %% [markdown]
# ### New feature: familyunit

# %%
# New feature: familyunit = partner & dependents 

churn['familyunit'] = churn['partner'] + churn['dependents']
churn.familyunit.value_counts()


# %%
sns.set_palette('Paired')
plt.figure(figsize=(10, 3))

plt.subplot(1, 3, 1)
x1 = churn.partner.value_counts(normalize=True)
x1.plot(kind='pie', title='Partner - Overall', colors=sns.color_palette(), autopct='%1.1f%%')

plt.subplot(1, 3, 2)
x2 = churn.dependents.value_counts(normalize=True)
x2.plot(kind='pie', title='Dependents - Overall', colors=sns.color_palette(), autopct='%1.1f%%')

# Plot 'familyunit' - those who have both a partner and dependents
plt.subplot(1, 3, 3)
x3 = churn.familyunit.value_counts(normalize=True)
x3.plot(kind='pie', title='Family Unit - Overall', colors=sns.color_palette(), autopct='%1.1f%%')

plt.subplots_adjust(wspace=0.5)
plt.show()

# Filter for churn == 1
filtered_df = churn[churn['churn'] == 1]

plt.figure(figsize=(10, 3))

plt.subplot(1, 3, 1)
x1 = filtered_df.partner.value_counts(normalize=True)
x1.plot(kind='pie', title='Partner - Churn', colors=sns.color_palette(), autopct='%1.1f%%')

plt.subplot(1, 3, 2)
x2 = filtered_df.dependents.value_counts(normalize=True)
x2.plot(kind='pie', title='Dependents - Churn', colors=sns.color_palette(), autopct='%1.1f%%')

# Plot 'familyunit' - those who have both
plt.subplot(1, 3, 3)
x3 = filtered_df.familyunit.value_counts(normalize=True)
x3.plot(kind='pie', title='Family Unit - Churn', colors=sns.color_palette(), autopct='%1.1f%%')

plt.subplots_adjust(wspace=0.5)
plt.show()

# %% [markdown]
# Conclusions: 
# - having a partner or (especially) dependents makes a customer less likely to churn. 
# - having both reduces it even further
# 

# %% [markdown]
# ### Check seniorcitizen and familyunit

# %%
# seniorcitizen and familyunit

plt.figure

sns.catplot(data=churn, x='seniorcitizen', y='churn', hue='familyunit', kind='point')
plt.show()

# %% [markdown]
# ### Phone Services
# 
# Create a df which is a subset of churn to examine phone service in relation to internet service and other related features

# %%
# Subcategories of internetservice: onlinesecurity, onlinebackup, deviceprotection, techsupport, streamingtv, streamingmovies 

# Create a filtered df solely of  customers who have internetservice
temp_df_phone = churn[churn['phoneservice'] == 1]
print(temp_df_phone.shape)

# Change 'multiplelines', 'internetservice' to int 
serv_list = ['multiplelines', 'internetservice']
temp_df_phone = temp_df_phone.copy()

for item in serv_list:
    temp_df_phone[item] = temp_df_phone[item].replace({'Yes': 1, 'No': 0})              
    temp_df_phone[item] = temp_df_phone[item].astype('int') 
    
for item in serv_list:
    print(temp_df_phone[item].value_counts())
    print(temp_df_phone[item].dtype)

# %%
# List of column names and indices for temp_df_phone
column_names_indices = [(i, column) for i, column in enumerate(temp_df_phone.columns)]
print(column_names_indices)

# %% [markdown]
# ### New feature: num_services_phone
# For customers who receive phone service

# %%
# num_services_phone = total number of services a customer receives from the following features:
#       phoneservice (this is intentionally included so that the min int in the dataset will be 1 and not 0)
#       multiplelines
#       internetservice 

columns_to_sum = temp_df_phone.columns[list(range(5, 8))]                 # List of column indices to sum over
temp_df_phone['num_services_phone'] = temp_df_phone[columns_to_sum].sum(axis=1)

temp_df_phone.num_services_phone.value_counts().sort_index()

# %%
sns.countplot(data=temp_df_phone, x='churn', hue='num_services_phone')
plt.show()

# %%
temp_df_phone['num_services_phone'].value_counts(normalize=True).sort_index()

# %%
x = temp_df_phone['num_services_phone'].value_counts(normalize=True).sort_index()

x_labels = [1, 2, 3]

plt.pie(data=temp_df_phone, x=x, labels=x_labels, autopct='%1.1f%%')
plt.title('Number of Phone Services')
plt.show()

# %% [markdown]
# ### Internet Services
# 
# Create a df which is a subset of churn to examine internet service in relation to phone service and other related features

# %%
# Subcategories of internetservice: onlinesecurity, onlinebackup, deviceprotection, techsupport, streamingtv, streamingmovies 

# Create a filtered df solely of  customers who have internetservice
temp_df_internet = churn[churn['internetservice'] == 1]
print(temp_df_internet.shape)

# Change onlinesecurity, onlinebackup, deviceprotection, techsupport, streamingtv, streamingmovies to int 
serv_list = ['onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport', 'streamingtv', 'streamingmovies']
temp_df_internet = temp_df_internet.copy()

for item in serv_list:
    temp_df_internet[item] = temp_df_internet[item].str.strip()
    temp_df_internet[item] = temp_df_internet[item].replace({'Yes': 1, 'No': 0})              
    temp_df_internet[item] = temp_df_internet[item].astype('int') 
    
for item in serv_list:
    print(temp_df_internet[item].value_counts())
    print(temp_df_internet[item].dtype)

# %%
temp_df_internet[['phoneservice', 'streamingtv', 'streamingmovies', 'onlinesecurity', 'onlinebackup', 'techsupport', 'deviceprotection']].dtypes

# %%
# List of column names and indices for temp_df_internet
column_names_indices = [(i, column) for i, column in enumerate(temp_df_internet.columns)]
print(column_names_indices)


# %% [markdown]
# ### New feature: num_services_internet
# For customers who receive internet service

# %%
# num_services_internet = total number of services a customer receives under the following categories:
#       phoneservice
#       internetservice (this is intentionally included so that the min int in the dataset will be 1 and not 0)
#       streamingtv, streamingmovies
#       onlinesecurity, onlinebackup, techsupport
#       deviceprotection

columns_to_sum = temp_df_internet.columns[[5] + list(range(7, 14))]                 # List of column indices to sum over
temp_df_internet['num_services_internet'] = temp_df_internet[columns_to_sum].sum(axis=1)

temp_df_internet.num_services_internet.value_counts().sort_index()

# %%
sns.countplot(data=temp_df_internet, x='churn', hue='num_services_internet')
plt.show()

# %%
print(temp_df_internet['churn'].corr(temp_df_internet['num_services_internet']))

# %% [markdown]
# There isn't a strong correlation between the number of services that internet customers get and the churn rate.

# %% [markdown]
# ## Senior citizens
# 

# %% [markdown]
# ### Check senior citizen churn rate overall

# %%
plt.figure(figsize=(4, 1))

sns.catplot(data=churn, x='seniorcitizen', y='churn', kind='point')
plt.show()

# %% [markdown]
# 
# ### Check senior citizen status and number of services among customers receiving internet service

# %%
print('Correlation of seniorcitizen and num_services_internet:', temp_df_internet['seniorcitizen'].corr(temp_df_internet['num_services_internet']))

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
sns.countplot(data=temp_df_internet, x='churn', hue='num_services_internet')
plt.title('Entire Population')

plt.subplot(1, 2, 2)
sns.countplot(data=temp_df_internet[temp_df_internet['seniorcitizen']==1], x='churn', hue='num_services_internet')
plt.title('Senior Citizens')
plt.legend(loc='upper right')

plt.subplots_adjust(wspace=0.2)
plt.show()

print('Correlation of churn and num_services_internet among seniorcitizen:', temp_df_internet[temp_df_internet['seniorcitizen']==1]['churn'].corr(temp_df_internet['num_services_internet']))

# %% [markdown]
# ## Tenure

# %%
# A relatively high standard devation indicates a high variability within tenure
print(
churn.tenure.describe(), '\n',
'med ' + str(churn.tenure.median()), '\n'
)

# %%
sns.displot(data=churn['tenure'], alpha=0.8, bins=25, kde=True)
plt.ylim(100, 900)
plt.show()

# %%
churn['tenure_grp'] = ''
grp_order = ['new', 'bronze', 'silver', 'gold', 'platinum']

churn.loc[churn['tenure'] <= 3, 'tenure_grp'] = grp_order[0]
churn.loc[(churn['tenure'] > 3) & (churn['tenure'] <= 6), 'tenure_grp'] = grp_order[1]
churn.loc[(churn['tenure'] > 6) & (churn['tenure'] <= 18), 'tenure_grp'] = grp_order[2]
churn.loc[(churn['tenure'] > 18) & (churn['tenure'] < 60), 'tenure_grp'] = grp_order[3]
churn.loc[churn['tenure'] >= 60, 'tenure_grp'] = grp_order[4]
    
print(churn['tenure_grp'].value_counts())

plt.figure(figsize=(13, 4))

plt.subplot(1, 3, 1)
ax = sns.histplot(data=churn, x='tenure', hue='tenure_grp', bins=30, legend=False, hue_order=grp_order)
plt.ylim(0, 900)
plt.title('Overall')

plt.subplot(1, 3, 2)
ax = sns.histplot(data=churn[churn['churn']==0], x='tenure', hue='tenure_grp', bins=30, legend=False, hue_order=grp_order)
plt.ylim(0, 900)
plt.title('No Churn')

plt.subplot(1, 3, 3)
ax = sns.histplot(data=churn[churn['churn']==1], x='tenure', hue='tenure_grp', bins=30, legend=True, hue_order=grp_order)
plt.ylim(0, 900)
plt.title('Churn')
sns.move_legend(ax, 'upper right')

plt.subplots_adjust(wspace=0.5)
plt.show()

# %% [markdown]
# Conclusion: Most customers who churn do so in the first 3 months.
# 
# Solutions?

# %% [markdown]
# Investigate monthlycharges

# %%
sns.histplot(churn['monthlycharges'])

plt.show()

# %%
# monthlycharges vs tenure

print(churn['monthlycharges'].corr(churn['tenure']))
sns.scatterplot(data=churn, x=churn['monthlycharges'], y=churn['tenure'])

plt.show()

# %% [markdown]
# It seems there is no correlation between tenure and monthlycharges

# %% [markdown]
# ## Gender

# %%
# Gender doesn't seem to have an effect on churn

plt.ylim(0, 0.5)
sns.barplot(x=churn['gender'], y=churn['churn'], hue=churn['gender'], palette='YlGnBu')
plt.show()

# %%
cols = ['internetservice','techsupport','onlinebackup','contracttype']

plt.figure(figsize=(14,4))

for i, col in enumerate(cols):
    ax = plt.subplot(1, len(cols), i+1)
    sns.countplot(x =churn['churn'], hue = str(col), data = churn, palette='YlGnBu')
    ax.set_title(f"{col}")

# %%
print(churn.groupby('gender').agg({'churn': ['count', 'sum', 'mean', 'std']}), '\n')
print(churn.groupby('seniorcitizen').agg({'churn': ['count', 'sum', 'mean', 'std']}), '\n')
print(churn.groupby('partner').agg({'churn': ['count', 'sum', 'mean', 'std']}), '\n')
print(churn.groupby('dependents').agg({'churn': ['count', 'sum', 'mean', 'std']}), '\n')


# %% [markdown]
# ## Paperlessbilling

# %%

sns.countplot(data=churn, x='paperlessbilling', hue='churn', palette='YlGnBu')
plt.show()


# %%

plt.ylim(0, 0.5)
sns.barplot(x=churn['paperlessbilling'], y=churn['churn'], hue=churn['paperlessbilling'], palette='YlGnBu')
plt.show()

# %% [markdown]
# Conclusion: Customers with paperless billing have a higher churn rate.

# %% [markdown]
# ## Contract Type

# %%
churn.contracttype.value_counts()

# %%
plt.figure(figsize=(4, 1))

sns.catplot(data=churn, x='contracttype', y='churn', kind='point')
plt.show()

# %% [markdown]
# ### Check tenure and contract type among customers who churn and don't churn

# %%
df_churn = churn[churn['churn'] == 1]
print(df_churn['tenure'].describe(), '\n')

df_no_churn = churn[churn['churn'] == 0]
print(df_no_churn['tenure'].describe())

plt.figure(figsize=(10, 4))

list_ord = ['Month-to-month', 'One year', 'Two year']

plt.subplot(1, 2, 1)
sns.barplot(data=df_churn, x='contracttype', y='tenure', hue='contracttype', order=list_ord, hue_order=list_ord)
plt.title('Churn - Tenure by Contract Type')

plt.subplot(1, 2, 2)
sns.barplot(data=df_no_churn, x='contracttype', y='tenure', hue='contracttype', order=list_ord, hue_order=list_ord)
plt.title('No Churn - Tenure by Contract Type')

plt.subplots_adjust(wspace=0.3)
plt.show()

# %%
sns.barplot(data=churn, x='contracttype', y='tenure', hue='churn', order=list_ord)
plt.show()

# %% [markdown]
# Create new features contracttype_monthly and contracttype_yearly

# %%
churn['contracttype_monthly'] = ''
churn['contracttype_yearly'] = ''

for i, val in enumerate(churn['contracttype']):
    if val == 'Month-to-month':
        churn.loc[i, 'contracttype_monthly'] = 1
        churn.loc[i, 'contracttype_yearly'] = 0
    else:
        churn.loc[i, 'contracttype_monthly'] = 0
        churn.loc[i, 'contracttype_yearly'] = 1

print(churn.contracttype_monthly.value_counts())
print(churn.contracttype_yearly.value_counts())

churn['contracttype_monthly'] = churn.contracttype_monthly.astype(int)
churn['contracttype_yearly'] = churn.contracttype_yearly.astype(int)


# %%
plt.figure(figsize=(4, 4))

sns.countplot(data=churn, x='contracttype_monthly', hue='churn')
plt.show()

# %% [markdown]
# # One-hot

# %%
# 'One hot' gender then change to int, drop gender

c = pd.get_dummies(churn['gender'])
c = c.astype('int')
churn = churn.join(c)
churn = churn.drop(['gender'], axis=1)

# One-hot the following columns then drop from df:

list1 = ['multiplelines', 'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport', 'streamingtv', 'streamingmovies', 'paymentmethod']

for col in list1:
    c = pd.get_dummies(churn[col], prefix=str(col))
    c = c.astype(int)
    churn = churn.join(c, rsuffix=str(col) + '_')
    churn = churn.drop(col, axis=1) 

# Change column names to lower case
churn = churn.rename(columns=str.lower)

# Check dtypes again
print(churn.dtypes)

# %% [markdown]
# Create new feature for automatic or non-automatic payments

# %%
churn['paymentmethod_auto'] = churn['paymentmethod_bank transfer (automatic)'] + churn['paymentmethod_credit card (automatic)']
churn['paymentmethod_not_auto'] = churn['paymentmethod_electronic check'] + churn['paymentmethod_mailed check']


# %%
churn.paymentmethod_auto.value_counts()

# %%
churn.paymentmethod_not_auto.value_counts()

# %%
churn.columns

# %% [markdown]
# Drop redundent or unnecessary features

# %%
drop_feat = ['tenure_grp', 'dependents', 'partner', 'contracttype', 'multiplelines_no phone service', 'onlinebackup_no internet service', 'techsupport_no internet service', 'streamingmovies_no internet service', 'onlinesecurity_no internet service', 'streamingtv_no internet service', 'deviceprotection_no internet service', 'paymentmethod_bank transfer (automatic)', 'paymentmethod_credit card (automatic)', 'paymentmethod_electronic check', 'paymentmethod_mailed check']

for i in drop_feat:
    churn.drop(i, axis=1, inplace=True)

# %%
plt.figure(figsize=(10, 10))
sort_as_abs = abs(churn.corr()['churn'].drop('churn')).sort_values(ascending=True)
sort_as_abs.plot(kind='barh')
plt.ylabel('feature')
plt.show()

# %%
plt.figure(figsize=(10, 10))
sns.heatmap(churn.corr(), cmap='viridis')
plt.show()

# %% [markdown]
# # ML Algorithms

# %% [markdown]
# Make adjustments to the original df in order to run prediction models

# %%
print(cst.shape, churn.shape)

# %%
churn_ML = churn.join(cst)
print(churn_ML.shape)
churn_ML.to_csv('churn_ML.csv', index=False)

# %%
for col in churn_ML.columns:
    print(col)

# %%
churn_ML.dtypes

# %% [markdown]
# ## Train and Test Splitting

# %%
from sklearn.model_selection import train_test_split

# %%
train, test = train_test_split(churn_ML, test_size=0.2, random_state=20, shuffle=True)

label = 'churn'
cst = 'customerid' 

x_train = train.drop(label, axis=1)
x_train = x_train.drop(cst, axis=1)

y_train = train[label]
cst_train = train[cst] 

x_test = test.drop(label, axis=1)
x_test = x_test.drop(cst, axis=1) 
y_test = test[label]
cst_test = test[cst] 

# %%
x_train.shape, y_train.shape, cst_train.shape, x_test.shape, y_test.shape, cst_test.shape

# %% [markdown]
# ## K Nearest Neighbors

# %%
from sklearn.metrics import accuracy_score 

# %%
from sklearn.neighbors import KNeighborsClassifier  

clf = KNeighborsClassifier(n_neighbors=3)      
clf.fit(x_train, y_train)

y_test_pred_Knn = clf.predict(x_test)

output = pd.DataFrame({'customerid': cst_test, 'churn_data':y_test, 'churn_pred': y_test_pred_Knn}) 
output

# %% [markdown]
# ## KNN accuracy (acc)

# %%
knn_test_acc_3n = accuracy_score(y_test, y_test_pred_Knn)
knn_test_acc_3n

# %%
from sklearn.neighbors import KNeighborsClassifier  

# %%

clf = KNeighborsClassifier(n_neighbors=6)      
clf.fit(x_train, y_train)

y_test_pred_Knn = clf.predict(x_test)

output = pd.DataFrame({'customerid': cst_test, 'churn_data':y_test, 'churn_pred': y_test_pred_Knn}) 
output

# %%

knn_test_acc_6n = accuracy_score(y_test, y_test_pred_Knn)
knn_test_acc_6n

# %% [markdown]
# Run the algorithm in a loop to check various k values.

# %%
acc_df_knn = pd.DataFrame()

acc_df_knn['k_val'] = ''
acc_df_knn['k_acc'] = ''

for i in range(1, 14):
    clf = KNeighborsClassifier(n_neighbors=i)      
    clf.fit(x_train, y_train)
    y_test_pred_Knn = clf.predict(x_test)
    knn_test_acc = accuracy_score(y_test, y_test_pred_Knn)
    acc_df_knn.loc[i-1, 'k_val'] = i
    acc_df_knn.loc[i-1, 'k_acc'] = knn_test_acc


max_acc = acc_df_knn['k_acc'].max()
max_k = acc_df_knn.loc[acc_df_knn['k_acc'].idxmax(), 'k_val']

print('The highest accuracy rate is %f with %d neighbors.' % (max_acc, max_k))

# %%
acc_df_knn

# %%
clf = KNeighborsClassifier(n_neighbors=10)      
clf.fit(x_train, y_train)

y_test_pred_Knn = clf.predict(x_test)

output = pd.DataFrame({'customerid': cst_test, 'churn_data':y_test, 'churn_pred': y_test_pred_Knn}) 
output

# %%
knn_test_acc_10n = accuracy_score(y_test, y_test_pred_Knn)
knn_test_acc_10n

# %% [markdown]
# ## Decision Tree

# %%
from sklearn.tree import DecisionTreeClassifier

# %%
clf = DecisionTreeClassifier(criterion='gini', random_state=20, max_depth=4)
clf.fit(x_train, y_train)                                             
y_test_pred_DecisionTree = clf.predict(x_test)

output = pd.DataFrame({'customerid': cst_test, 'churn_data': y_test, 'churn_pred': y_test_pred_DecisionTree}) 
output

# %%
# Evaluation for Decision Tree
test_acc = accuracy_score(y_test, y_test_pred_DecisionTree)
test_acc

# %% [markdown]
# ## Plotting the decision tree

# %%
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from IPython.display import SVG
from graphviz import Source
from IPython.display import display

# %%
def plot_tree(tree, features, labels):
    graph = Source(export_graphviz(tree, feature_names=features, class_names=labels, filled = True))
    display(SVG(graph.pipe(format='svg')))

plot_tree(clf, x_train.columns, ['Churn', 'No Churn'])

# %% [markdown]
# ## Run the algorithm with several maxdepths and check accuracy

# %%
acc_df = pd.DataFrame()

acc_df['dec_tree_depth'] = ''
acc_df['dec_tree_acc'] = ''

for i in range(1, 14):
    clf = DecisionTreeClassifier(criterion='gini', random_state=20, max_depth=i)
    clf.fit(x_train, y_train)                                             
    y_test_pred_DecisionTree = clf.predict(x_test)
    test_acc = accuracy_score(y_test, y_test_pred_DecisionTree)
    acc_df.loc[i-1, 'dec_tree_depth'] = i
    acc_df.loc[i-1, 'dec_tree_acc'] = test_acc

max_acc = acc_df['dec_tree_acc'].max()
max_depth = acc_df.loc[acc_df['dec_tree_acc'].idxmax(), 'dec_tree_depth']

print('The highest accuracy rate is %f for a tree depth of %d.' % (max_acc, max_depth))

# %%
acc_df

# %%
# Run the model with the highest acc
clf = DecisionTreeClassifier(criterion='gini', random_state=20, max_depth=6)
clf.fit(x_train, y_train)                                             
y_test_pred_DecisionTree = clf.predict(x_test)

output = pd.DataFrame({'customerid': cst_test, 'churn_data': y_test, 'churn_pred': y_test_pred_DecisionTree}) 
output

# %%
# Evaluation for Decision Tree
dt_test_acc = accuracy_score(y_test, y_test_pred_DecisionTree)
dt_test_acc

# %% [markdown]
# ### Decision tree overfitting

# %%
# Run max-depth and acc for 1-100 nodes in order to graph results

acc_df_graph = pd.DataFrame()

acc_df_graph['tree_depth'] = ''
acc_df_graph['train_dec_tree_acc'] = ''
acc_df_graph['test_dec_tree_acc'] = ''

num_features = len(x_test.columns)

for i in range(1, num_features+1):
    clf = DecisionTreeClassifier(criterion='gini', random_state=20, max_depth=i)
    clf.fit(x_train, y_train)  
    acc_df_graph.loc[i-1, 'tree_depth'] = i                                           
    y_test_pred_decisiontree = clf.predict(x_test)
    test_acc = accuracy_score(y_test, y_test_pred_decisiontree)
    acc_df_graph.loc[i-1, 'test_dec_tree_acc'] = test_acc
    y_train_pred_decisiontree = clf.predict(x_train)
    train_acc = accuracy_score(y_train, y_train_pred_decisiontree)
    acc_df_graph.loc[i-1, 'train_dec_tree_acc'] = train_acc


# %%
acc_df_graph

# %%

plt.figure(figsize=(8, 4))
sns.lineplot(data=acc_df_graph, x='tree_depth', y='train_dec_tree_acc', label='Train Accuracy')
sns.lineplot(data=acc_df_graph, x='tree_depth', y='test_dec_tree_acc', label='Test Accuracy')
plt.title('Decision Tree Accuracy vs. Tree Depth')
plt.xlabel('Tree Depth')
plt.ylabel('Accuracy')
plt.ylim(0.7, 1)
plt.xticks(range(1, num_features+1))
plt.legend()
plt.show()

# %% [markdown]
# ## Random Forest

# %%
from sklearn.ensemble import RandomForestClassifier 

# %%
model_rfc = RandomForestClassifier(n_estimators=101, max_depth=6, random_state=20)
model_rfc.fit(x_train, y_train) 
y_test_pred_RandomForest = model_rfc.predict(x_test) 

output = pd.DataFrame({'customerid': cst_test, 'churn_data': y_test, 'churn_pred': y_test_pred_RandomForest}) 
output

# %%
# Evaluation for Random Forest
rf_test_acc = accuracy_score(y_test, y_test_pred_RandomForest)
rf_test_acc

# %%
acc_df_rf = pd.DataFrame()

acc_df_rf['rf_tree_depth'] = ''
acc_df_rf['rf_acc'] = ''
acc_df_rf['rf_n_est'] = ''

for i_md in range(1, 8):
    for n_est in range(1, 102):
        model_rfc = RandomForestClassifier(n_estimators=n_est, max_depth=i_md, random_state=20)
        model_rfc.fit(x_train, y_train)                                              
        y_test_pred_RandomForest = model_rfc.predict(x_test)
        test_acc = accuracy_score(y_test, y_test_pred_RandomForest)
        acc_df_rf.loc[i_md-1, 'rf_tree_depth'] = i_md
        acc_df_rf.loc[i_md-1, 'rf_acc'] = test_acc
        acc_df_rf.loc[i_md-1, 'rf_n_est'] = n_est

acc_df_rf['rf_acc'] = pd.to_numeric(acc_df_rf['rf_acc'])
max_acc = acc_df_rf['rf_acc'].max()
max_depth = int(acc_df_rf.loc[acc_df_rf['rf_acc'].idxmax(), 'rf_tree_depth'])
num_n_est = int(acc_df_rf.loc[acc_df_rf['rf_acc'].idxmax(), 'rf_n_est'])

print('The highest accuracy rate is %f for a tree depth of %d with %d n-estimators.' % (max_acc, max_depth, num_n_est))

# %%
acc_df_rf

# %% [markdown]
# ### Check feature importance

# %%
feature_importances = model_rfc.feature_importances_ 
feature_importances                                     # Every feature importance (%)

# %%
features = x_train.columns                              # all the features
features_stats = pd.DataFrame({'feature':features, 'importance': feature_importances})
features_stats.sort_values('importance', ascending=False)

# %%
features_stats.importance.sum()         # Check feature_importances sum is 1

# %%
plt.figure(figsize=(4, 7))
stats_sort = features_stats.sort_values('importance', ascending=True)
stats_sort.plot(y='importance', x='feature', kind='barh')         
plt.title('Feature Importance of Random Forest')
plt.show()

# %% [markdown]
# The feature importances align with the previous correlations. We may conclude that the model is correctly predicting which features will affect churn.

# %% [markdown]
# ## Accuracy summary

# %%
# Evaluation for KNN

knn_test_acc_10n

# %%
# Evaluation for Decision Tree

dt_test_acc

# %%
# Evaluation for Random Forest

rf_test_acc

# %% [markdown]
# # Check benchmarks

# %%
y_train.value_counts()

# %%
benchmark_value = 0   # the most common value in the label set

# The function returns a numpy array on the same length as x with all values equal to benchmark_value
def get_benchmark_predictions(x, benchmark_value): 
    return np.ones(len(x))*benchmark_value
   
y_test_pred_benchmark = get_benchmark_predictions(x_test, benchmark_value)
y_test_pred_benchmark

# %%
bench_test_acc = accuracy_score(y_test, y_test_pred_benchmark)
bench_test_acc

# %% [markdown]
# We can see that our models are much more accurate than the basic benchmark performance.


