# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# #### 1. Train data set is visualized in several ways using swarmplot in seaborn to provide guidelines for feature engineering followed by one-hot encoding. A total of 6 features are used:
# #### (1) Pclass and Sex are used without any modification,
# #### (2) Fare is binned into 4 groups with 'unknown' and by using bin edges 10.5 and 75,
# #### (3) SibSp and Parch are added together and binned to unkown, 0 (travel alone), 4 below, and 4 and above to form a new feature,
# #### (4) Persons with 'Master' in Name are identified and form a new feature,
# #### (5) Female in Pclass 3 with Fare > 24 is identified and forms a new feature, 
# #### 2. Eight models with hyper-parameter tuning are constructed for predictions: logistic regression, random forest, gradient boosting, XGBoost, multinomial naive Bayes, k nearest neighbors, stack, and majority vote. The stack model uses all the first 6 models above as the 1st-level models and random forest as the 2nd-level model. 
# #### 3. In summary, gradient boost and stack models have the highest mean cross-validation scores (both 0.842), followed by random forest and XGBoost (0.837 and 0.836, respectively), followed by logistic regression and k nearest neighbors (0.828 and 0.827, respectively), and multinomial naive Bayes has the lowest score of 0.780.
# #### However, random forest, together with stack, achieve the highest public scores of 0.799, followed by logistic regression, gradient boost, and XGBoost (all 0.794), followed by k nearest neighbors with 0.789, and multinomial naive Bayes has the lowest public score of 0.746. The majority vote also achieves the highest public score of 0.799.
# #### It is found that model performance (model's public score) may be highly dependent on the number of features chosen and the ways the features are enginnered.

# %autosave 0

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import operator
from scipy.stats import uniform, norm
# %matplotlib inline

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, \
cross_val_score, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost
from xgboost import XGBClassifier
import sklearn

# ## Data visualization

data_train = pd.read_csv('train.csv')
data_train.info()

data_train.head()

sns.set(style='darkgrid')
colors = ['darkred', 'darkseagreen']

# ### 1. Visualize the survival chance of persons with different fare

data_train[data_train['Fare'] > 300]

fig, ax = plt.subplots(figsize=(17,12))
ax.set_ylim(0, 300)
ax.set_yticks(np.arange(0, 300, 10))
sns.swarmplot(y=data_train['Fare'], x=[""]*len(data_train), size=4, 
              hue=data_train['Survived'], palette=colors)

# #### The plot above shows that persons with fare above 75 had a relatively good chance of survival, and those with fare below about 10.5 the chance was quite bad, and those with fare in between seems to have a chance somewhere in the middle. 

# ### 2. Add 'SibSp' and 'Parch' together and visualize the chance of survival

df_try = data_train.copy()
df_try['SibSp_Parch'] = df_try['SibSp'] + df_try['Parch']
df_try.groupby('SibSp_Parch')['Survived'].value_counts()

fig, ax = plt.subplots(figsize=(17, 7))
sns.swarmplot(y=df_try['SibSp_Parch'], x=[""]*len(df_try), size=4, hue=df_try['Survived'], 
              palette=colors)

# #### The plot above shows that persons with 4 relatives or above had a relatively small chance of survival, and the same is true (to a lesser extent) with persons who traveled alone with 0 relatives. In contrast, persons with 1 to 3 relatives had a better chance of survival.   

# ### 3. Visualize chance of survival in plots combining sex, age, and Pclass

g1 = sns.FacetGrid(data_train, col='Pclass', hue='Survived', palette=colors, size=5, aspect=1)
g1 = g1.map(sns.swarmplot, 'Sex', 'Age', order=['male', 'female'], size=5)
g1.add_legend()

# #### It can be seen from the plot above that male with age less than about 12 years old had a better chance of survival compared to male older than this age. We will later create a new feature to reflect this.

mask_master = pd.Series('Master' in i for i in data_train['Name'])
data_train[mask_master].sort_values('Age', ascending=False).head(10)

# #### From the table above it can be seen that if a person has 'Master' in 'Name' then this person is a male with age less than or equal to 12 years old.

fig, ax = plt.subplots(figsize=(17, 5))
sns.swarmplot(x='Sex', y='Fare', data=data_train[data_train['Pclass']==3], size=4, 
              hue='Survived', palette=colors)

# #### It can be seen from the plot above that female in Pclass 3 with fare greater than about 24 almost all did not make it. We will also later create a new feature to reflect this.

# ## Data cleaning and preprocessing

y = data_train['Survived']
X = data_train.drop('Survived', axis=1)
X.head()


# +
def combine_Sib_Par(df):
    """Sum the two columns SibSp and Parch together."""
    df['SibSp_Parch'] = df['SibSp'] + df['Parch']

def add_name_master_feature(df):
    """Create a new feature: if Master in Name, then Yes, otherwise, No."""
    mask_master = pd.Series('Master' in i for i in df['Name'])
    df1 = df['Name'].mask(mask_master, 'Yes')
    df['Name_Master'] = df1.where(mask_master, 'No')    
    
def add_female_pclass_3_high_fare_feature(df):
    """Create a new feature: if female, in Pclass 3, and Fare > 24, Yes, otherwise, No."""
    df_temp = df[((df['Pclass']==3) & (df['Sex']=='female')) & (df['Fare']>24.)]
    mask = df.index.isin(df_temp.index)
    df['Fem_Hfare_Pcl3'] = pd.Series(range(df.shape[0])).mask(mask, 'Yes')
    df['Fem_Hfare_Pcl3'] = df['Fem_Hfare_Pcl3'].where(mask, 'No')

def drop_feature(df):
    df.drop(['PassengerId', 'Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], \
            axis=1, inplace=True)
    
def fill_feature(df):
    """Fill all NaN values."""
    df['Pclass'] = df['Pclass'].fillna(-1)
    df['Sex'] = df['Sex'].fillna('Unknown')
    df['SibSp_Parch'] = df['SibSp_Parch'].fillna(-1)
    df['Fare'] = df['Fare'].fillna(-0.5)

def bin_fare_and_SibSpParch(df):
    """Bin Fare and SibSp_Parch based on previous visualization results."""
    bins = (-1, 0, 10.5, 75, 1500)
    group_names = ['Unknown', '10.5_below', '10.5_to_75', '75_above']
    df['Fare'] = pd.cut(df['Fare'], bins, labels=group_names, right=False)
    
    bins = (-1, -0.1, 0.1, 4, 50)
    group_names = ['Unknown', '0', '4_below', '4_above']
    df['SibSp_Parch'] = pd.cut(df['SibSp_Parch'], bins, labels=group_names, right=False)

def data_transform(df):
    combine_Sib_Par(df)
    add_name_master_feature(df)
    add_female_pclass_3_high_fare_feature(df)
    drop_feature(df)
    fill_feature(df)
    bin_fare_and_SibSpParch(df)


# -

data_transform(X)

X.head(10)

X.info()

ohe = OneHotEncoder(handle_unknown='ignore')
X_1 = ohe.fit_transform(X).toarray()
list(X_1)[:5] 

ohe.categories_

x_ax = ohe.get_feature_names(['Pclass', 'Sex', 'Fare', 'SibSp_Parch', 'Name_Master', 
                              'Fem_Hfare_Pcl3'])
x_ax

# Create a DataFrame for correlation plot
X_1_frame = pd.DataFrame(X_1, columns=x_ax)
X_1_frame.head()

plt.figure(figsize=(12, 12))
plt.title('Corelation Matrix', size=8)
sns.heatmap(X_1_frame.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=plt.cm.RdBu, 
            linecolor='white', annot=True)
plt.show()

# ## Data training and parameter tuning

# define a cv splitter
cv_splitter = StratifiedKFold(n_splits=5, random_state=42)

# ### 1. First Model: Logistic Regressionfrom 
#

logis = LogisticRegression(solver='liblinear', random_state=42)
C_param = sorted(10**np.random.uniform(-2, 0, size=100))  # log-uniform distrbution from 0.01 to 1
# Since if there are multiple parameter combinations rank first, GridSearchCV will choose the
# first encountered one as the best result, sort the array so the smallest possible C can be 
# picked. 
parameter_grid = {
                'C': C_param,
                'class_weight': ['balanced', None]
                }
grid_logis = GridSearchCV(logis, parameter_grid, cv=cv_splitter, refit=True)
grid_logis.fit(X_1, y)

logis_best_param = grid_logis.best_params_  
logis_best_param
# best parameter values to be used in the stack model

results = pd.DataFrame(grid_logis.cv_results_)
results.iloc[:,4:].sort_values('rank_test_score')

x_ax = ohe.get_feature_names(['Pclass', 'Sex', 'Fare', 'SibSp_Parch', 'Name_Master', 
                              'Fem_Hfare_Pcl3'])
x_ax

fig, ax = plt.subplots(figsize=(30,8))
ax.bar(x_ax, grid_logis.best_estimator_.coef_[0])
ax.grid

scores_logis = cross_val_score(grid_logis.best_estimator_, X_1, y, cv=cv_splitter, n_jobs=-1)
print(scores_logis)
print('Mean (logis): '+str(scores_logis.mean()))
print('SD (logis): '+str(scores_logis.std()))

# ## Test data preprocessing

data_test = pd.read_csv('test.csv')
passenger_id = data_test['PassengerId']
num_row_test = data_test.shape[0]
data_test.head()

data_test.info()

data_transform(data_test)
data_test.head()

X_test = ohe.transform(data_test).toarray()
list(X_test)[:5]

# ## Model Predictions

# ### 1. Logistic regression

# +
y_test_predict_logis = grid_logis.predict(X_test)
submission_logis_2 = pd.DataFrame({'PassengerId': passenger_id, 'Survived': y_test_predict_logis})

existing_file = glob.glob('submission_logis_2.csv')
assert (not existing_file), 'File already existed.'
submission_logis_2.to_csv('submission_logis_2.csv', index=False)
# (This submission got a public score of 0.794)
# -


