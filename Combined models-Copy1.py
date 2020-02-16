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

# ### 2. Second Model: Random Forest

# set max_features normal distribution sample array
num_feature = X_1.shape[1]
max_feature = norm.rvs(np.sqrt(num_feature), 2, size=200, random_state=42).astype(int)
max_feature[max_feature <= 0] = 1
max_feature[max_feature > num_feature] = num_feature
max_feature

# set min_samples_split normal distribution sample array
min_sample_split = norm.rvs(4, 2, size=200, random_state=42).astype(int)
min_sample_split[min_sample_split <= 1] = 2
min_sample_split

rf = RandomForestClassifier(random_state=42)
parameter_grid = {
                 'n_estimators': np.arange(50, 800, step=5),
                 'max_features': max_feature,
                 'min_samples_split': min_sample_split,
                 'min_samples_leaf': np.arange(1, 5, 1),
                 'bootstrap': [True, False]
                 }
grid_random = RandomizedSearchCV(rf, parameter_grid, n_iter=100, cv= cv_splitter, 
                                 random_state=42, refit=True, n_jobs=-1)
grid_random.fit(X_1, y)

random_forest_best_param = grid_random.best_params_  
random_forest_best_param
# best parameter values to be used in the stack model

grid_random.n_splits_

grid_random.best_estimator_.get_params

fig, ax = plt.subplots(figsize=(35,8))
ax.bar(x_ax, grid_random.best_estimator_.feature_importances_)
ax.grid

scores_random = cross_val_score(grid_random.best_estimator_, X_1, y, cv=cv_splitter, n_jobs=-1)
print(scores_random)
print('Mean (random): '+str(scores_random.mean()))
print('SD (random): '+str(scores_random.std()))

# ### 3. Third Model: Gradient Boosting

# #### 1. Tune learning_rate and n_estimators

gb = GradientBoostingClassifier(
                                learning_rate=0.1, 
                                n_estimators=100, 
                                max_features='sqrt',
                                subsample=0.8, 
                                random_state=42
                                )
parameter_grid = {
                    'learning_rate': np.arange(0.001, 0.003, 0.0005),
                    'n_estimators': np.arange(1000, 3000, 500)
                    }
grid_gradient = GridSearchCV(gb, parameter_grid, cv=cv_splitter, n_jobs=-1)
grid_gradient.fit(X_1, y)

gradient_best_param = grid_gradient.best_params_
gradient_best_param
# best parameter values to be used in the stack model

# update gb with the optimal parameters
gb.set_params(**gradient_best_param)

# #### 2. Tune max_depth and min_sample_split

parameter_grid = {
                    'max_depth': np.arange(1, 5),
                    'min_samples_split': np.arange(2, 6, 1)
                    }
grid_gradient = GridSearchCV(gb, parameter_grid, cv=cv_splitter, n_jobs=-1)
grid_gradient.fit(X_1, y)

grid_gradient.best_params_

gradient_best_param.update(grid_gradient.best_params_)
gradient_best_param
# update best parameter values to be used in the stack model

# update gb with the optimal parameters
gb.set_params(**gradient_best_param)

# #### 3. Tune max_features and subsample

parameter_grid = {
                    'max_features': np.arange(2, 6),
                    'subsample': np.arange(0.4, 0.8, step=0.1)
                    }
grid_gradient = GridSearchCV(gb, parameter_grid, cv=cv_splitter, n_jobs=-1)
grid_gradient.fit(X_1, y)

grid_gradient.best_params_

gradient_best_param.update(grid_gradient.best_params_)
gradient_best_param
# update best parameter values to be used in the stack model

grid_gradient.best_estimator_

fig, ax = plt.subplots(figsize=(35,8))
ax.bar(x_ax, grid_gradient.best_estimator_.feature_importances_)
ax.grid

scores_gradient = cross_val_score(grid_gradient.best_estimator_, X_1, y, cv=cv_splitter, n_jobs=-1)
print(scores_gradient)
print('Mean (gradient): '+str(scores_gradient.mean()))
print('SD (gradient): '+str(scores_gradient.std()))

# ### 4. Fourth Model: XGBoost

# #### (The tuning steps can be found in the article by Aarshay Jain at https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/)

# #### 1. Fix learning rate at 0.01 and find the optimal number of trees (n_estimators) 

xgtrain = xgboost.DMatrix(X_1, label=y.values)

xgb = XGBClassifier(
                     learning_rate =0.01,
                     n_estimators=1000,
                     max_depth=5,
                     min_child_weight=1,
                     gamma=0,
                     subsample=0.8,
                     colsample_bytree=0.8,
                     n_jobs=-1,
                     random_state=42
                    )

xgb_param = xgb.get_xgb_params()
xgb_param

cvresult = xgboost.cv(xgb_param, xgtrain, 
                  num_boost_round=xgb.get_params()['n_estimators'], 
                  nfold=5,
                  metrics='auc', 
                  early_stopping_rounds=50,
                  seed=42
                  )

cvresult.head()

cvresult.shape

xgb_best_param = {'n_estimators': cvresult.shape[0]}
xgb_best_param
# best n_estimators value to be used in the stack model

# update xgb with the optimal n_estimators
xgb.set_params(**xgb_best_param)

# #### 2. Tune max_depth and min_child_weight

parameter_grid = {
                    'max_depth': np.arange(2, 4),
                    'min_child_weight': np.arange(1, 4)
                 }
grid_xgb = GridSearchCV(xgb, parameter_grid, cv=cv_splitter, n_jobs=-1)
grid_xgb.fit(X_1, y)

grid_xgb.best_params_

xgb_best_param.update(grid_xgb.best_params_)
xgb_best_param
# best parameter values to be used in the stack model

xgb.set_params(**xgb_best_param)
# update xgb parameters

# #### 3. Tune gamma

10**np.random.uniform(-3, 0, size=10)  # log-uniform distrbution from 0.001 to 1

parameter_grid = {
                    'gamma': 10**np.random.uniform(-3, 0, size=10)
                 }
grid_xgb = GridSearchCV(xgb, parameter_grid, cv=cv_splitter, n_jobs=-1)
grid_xgb.fit(X_1, y)

grid_xgb.best_params_

results = pd.DataFrame(grid_xgb.cv_results_)
results.iloc[:,4:].sort_values('rank_test_score')

xgb_best_param.update(grid_xgb.best_params_)
xgb_best_param
# best parameter values to be used in the stack model

# update xgb with the optimal gamma
xgb.set_params(**xgb_best_param)

# #### 4. Tune subsample and colsample_bytree

parameter_grid = {
                    'subsample': np.arange(0.6, 1.0, 0.1),
                    'colsample_bytree': np.arange(0.6, 1.0, 0.1)
                 }
grid_xgb = GridSearchCV(xgb, parameter_grid, cv=cv_splitter, n_jobs=-1)
grid_xgb.fit(X_1, y)

grid_xgb.best_params_

xgb_best_param.update(grid_xgb.best_params_)
xgb_best_param
# best parameter values to be used in the stack model

# update xgb with the optimal parameters
xgb.set_params(**xgb_best_param)

# #### 5. Finally tune reg_alpha

parameter_grid = {
                    'reg_alpha': np.arange(0., 0.005, 0.001),
                 }
grid_xgb = GridSearchCV(xgb, parameter_grid, cv=cv_splitter, n_jobs=-1)
grid_xgb.fit(X_1, y)

grid_xgb.best_params_

results = pd.DataFrame(grid_xgb.cv_results_)
results.iloc[:,4:].sort_values('rank_test_score')

xgb_best_param.update(grid_xgb.best_params_)
xgb_best_param
# best parameter values to be used in the stack model

grid_xgb.best_estimator_

# #### 6. Visualize the feature importance of the final model

fig, ax = plt.subplots(figsize=(35,8))
ax.bar(x_ax, grid_xgb.best_estimator_.feature_importances_)
ax.grid

scores_xgb = cross_val_score(grid_xgb.best_estimator_, X_1, y, cv=cv_splitter, n_jobs=-1)
print(scores_xgb)
print('Mean (xgb): '+str(scores_xgb.mean()))
print('SD (xgb): '+str(scores_xgb.std()))

# ### 5. Fifth Model: Multinomial Naive Bayes

mnb = MultinomialNB()
parameters = {'alpha':np.arange(0.1, 1, 0.1)}
grid_mnb = GridSearchCV(mnb, param_grid=parameters, cv=cv_splitter, n_jobs=-1)
grid_mnb.fit(X_1, y)

grid_mnb.best_params_

pd.DataFrame(grid_mnb.cv_results_)

mnb_best_param = grid_mnb.best_params_
mnb_best_param

grid_mnb.best_estimator_

scores_mnb = cross_val_score(grid_mnb.best_estimator_, X_1, y, cv=cv_splitter, n_jobs=-1)
print(scores_mnb)
print('Mean (mnb): '+str(scores_mnb.mean()))
print('SD (mnb): '+str(scores_mnb.std()))

# ### 6. Sixth Model: K Nearest Neighbor

knn = KNeighborsClassifier()
parameter_grid = {
                'n_neighbors': np.arange(9, 19, 2),
                'weights': ['uniform', 'distance'],
                'metric': ['minkowski', 'manhattan'],
                'leaf_size': np.arange(10, 60, 10)
                 }
grid_knn = GridSearchCV(knn, parameter_grid, cv=cv_splitter, n_jobs=-1)
grid_knn.fit(X_1, y)

grid_knn.best_params_

results = pd.DataFrame(grid_knn.cv_results_)
results.iloc[:,4:].sort_values('rank_test_score')

knn_best_param = grid_knn.best_params_
knn_best_param

grid_knn.best_estimator_

scores_knn = cross_val_score(grid_knn.best_estimator_, X_1, y, cv=cv_splitter, n_jobs=-1)
print(scores_knn)
print('Mean (knn): '+str(scores_knn.mean()))
print('SD (knn): '+str(scores_knn.std()))

# ### 7. Compare Results of the above 5 Models

cross_val_results = pd.Series([scores_logis, scores_random, scores_gradient, 
                       scores_xgb, scores_mnb, scores_knn], 
            index=['Logistic', 'Random Forrest', 'Gradirnt Boost', 'XGBoost', 'MN Bayes', 'Knn'])

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x=cross_val_results.index, y=cross_val_results.apply(np.mean))
ax.set_ylim(0.77, 0.85)
ax.set_yticks(np.arange(0.77, 0.85, step=0.01))
fig.suptitle('Mean of Cross-Validation Scores')

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x=cross_val_results.index, y=cross_val_results.apply(np.std))
ax.set_ylim(0.01, 0.06)
ax.set_yticks(np.arange(0.01, 0.06, step=0.005))
fig.suptitle('Standard Deviation of Cross-Validation Scores')

# ### 8. Seventh Model: Stackiing
# #### (1) In stacking, kFold cross-validated predictions of 1st-level models are used as input (where the 1st-level models become the new features) for training by a 2nd-level model. 
# #### (2) Cross-validated 1st-level models are also used to predict (not train) on the test data set and the outcome (with 1st-level models as the new features) are used as input to the final prediction by the 2nd-level model.   
# #### (3) Some discussions and code can be found at https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python, https://www.kdnuggets.com/2017/02/stacking-models-imropved-predictions.html.

# #### 1. First test data need to be processed and encoded in the same way as train data (Cross-validated 1st-level models will predict (not train) on the data and the outcome becomes input to the final prediction of the 2nd-level model).    

data_test = pd.read_csv('test.csv')
passenger_id = data_test['PassengerId']
num_row_test = data_test.shape[0]
data_test.head()

data_test.info()

data_transform(data_test)
data_test.head()

X_test = ohe.transform(data_test).toarray()
list(X_test)[:5]

# #### 2. Define 1st-level models (using the optimal hyperparameters previously) 

logis = LogisticRegression(solver='liblinear', random_state=42)
logis.set_params(**logis_best_param)

rf = RandomForestClassifier(random_state=42)
rf.set_params(**random_forest_best_param)

gb = GradientBoostingClassifier(random_state=42)
gb.set_params(**gradient_best_param)

xgb = XGBClassifier(learning_rate=0.01, n_jobs=-1, random_state=42)
xgb.set_params(**xgb_best_param)

mnb = MultinomialNB()
mnb.set_params(**mnb_best_param)

knn = KNeighborsClassifier()
knn.set_params(**knn_best_param)

# #### 3. Prepare 2nd-level training and test data

folds = 5
skf = StratifiedKFold(n_splits=folds, random_state=42)


def get_oof(clf_, X_1_, y_, X_test_, folds_):
    """ Obtain out-of-fold predictions of a model on train and test data sets. 
    
    Parameters:
    ----------
    clf_: sklearn classifier
            a 1st-level model
    X_1_: numpy array
            train data with shape (891, 15)
    y_: pandas series
            train data labels with a shape (891,)
    X_test_: numpy array
            test data with shape (418, 15)
    folds: int
            number of folds in stratified K-fold cross validator
    
    Returns:
    --------
    X_train_oof: numpy array
            2nd-level train data: out-of-fold predictions of clf_ with shape (891,)
    X_test_oof: numpy array
            2nd-level test data: mean of out-of-fold predictions of clf_ on test data with 
            shape (418,)
    """

    X_train_oof = np.zeros((X_1_.shape[0],))    # 2nd-level train data with shape (891,)
    X_test_oof = np.zeros((X_test_.shape[0],))  # 2nd-level test data with shape (418,)
    X_test_oof_folds = np.zeros((folds_, X_test_.shape[0]))  
    # with shape (5, 418), a temporary array holding out-of-fold predictions of clf_ on test data 

    for i, (train_index, valid_index) in enumerate(skf.split(X_1_, y_)):  
        # i: out-of-fold group index (e.g. 0)
        # train_index: numpy array holding all train X_1_ and y_ row indices (e.g. from 179-890) 
        # valid_index: numpy array holding all valid X_1_ and y_ row indices (e.g. from 1-178)

        X_train_folds = X_1_[train_index]        # select data for train folds
        y_train_folds = y_[train_index]          # select labels for train folds 
        clf_.fit(X_train_folds, y_train_folds)   # train clf_ on train folds
        
        X_train_valid_fold = X_1_[valid_index]   # select data for valid (out-of-fold) fold
        X_train_oof[valid_index] = clf_.predict(X_train_valid_fold)  
        # clf_ predicts on valid fold and save to 2nd-level train data

        X_test_oof_folds[i, :] = clf_.predict(X_test_)
        # clf_ predicts on the entire set of test data and save the results to the i-th row in the
        # temporary array X_test_oof_folds

    X_test_oof = X_test_oof_folds.mean(axis=0)  
    # calculate the mean of out-of-fold predcitons by collapsing in the 0-th axid (with 5 rows) 
    
    return X_train_oof, X_test_oof


# +
# construct 2nd-level train and test data
clfs = [logis, rf, gb, xgb, mnb, knn]
X_train_oof_final = np.zeros((X_1.shape[0], len(clfs)))       # with shape (891, 5)
X_test_oof_final = np.zeros((X_test.shape[0], len(clfs)))     # with shape (418, 5)

for i, clf in enumerate(clfs):
    clf_train_off, clf_test_off = get_oof(clf, X_1, y, X_test, folds)
    X_train_oof_final[:, i] = clf_train_off
    X_test_oof_final[:, i] = clf_test_off
# -

X_train_oof_final.shape

X_train_oof_final[:,0]

X_test_oof_final[:, 0]

# #### 4. Perform the final level 2 modeling using random forest

# set max_features normal distribution sample array
num_feature = X_train_oof_final.shape[1]  # 6 features
max_feature = norm.rvs(np.sqrt(num_feature), 2, size=200, random_state=42).astype(int)
max_feature[max_feature <= 0] = 1
max_feature[max_feature > num_feature] = num_feature
max_feature

# set min_samples_split normal distribution sample array
min_sample_split = norm.rvs(4, 2, size=200, random_state=42).astype(int)
min_sample_split[min_sample_split <= 1] = 2
min_sample_split

# Use the 2 previously defined normal distribution sample arrays 'max_feature' and 
# 'min_sample_split' 
rf = RandomForestClassifier(random_state=42)
parameter_grid = {
                 'n_estimators': np.arange(50, 800, step=5),
                 'max_features': max_feature,
                 'min_samples_split': min_sample_split,
                 'min_samples_leaf': np.arange(1, 5, 1),
                 'bootstrap': [True, False]
                 }
grid_random_stack = RandomizedSearchCV(rf, parameter_grid, n_iter=100, cv= cv_splitter, 
                                 random_state=42, refit=True, n_jobs=-1)
grid_random_stack.fit(X_train_oof_final, y)

random_forest_stack_best_param = grid_random_stack.best_params_  
random_forest_stack_best_param

grid_random_stack.n_splits_

grid_random_stack.best_estimator_

x_clfs = ['Logistic', 'Random Forrest', 'Gradirnt Boost', 'XGBoost', 'MN Bayes', 'Knn']

grid_random_stack.best_estimator_.feature_importances_

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(x_clfs, grid_random_stack.best_estimator_.feature_importances_)
ax.grid
fig.suptitle('Feature Importance in Stack Model')

scores_random_stack = cross_val_score(grid_random_stack.best_estimator_, X_train_oof_final, y, cv=cv_splitter, n_jobs=-1)
print(scores_random_stack)
print('Mean (random): '+str(scores_random_stack.mean()))
print('SD (random): '+str(scores_random_stack.std()))

# ### 9. Compare results of all 7 models

cross_val_results_all = pd.Series([scores_logis, scores_random, scores_gradient, 
                       scores_xgb, scores_mnb, scores_knn, scores_random_stack], 
            index=['Logistic', 'Random Forrest', 'Gradirnt Boost', 'XGBoost', 'MN Bayes', 'Knn', 
                   'Stack'])

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=cross_val_results_all.index, y=cross_val_results_all.apply(np.mean))
ax.set_ylim(0.77, 0.85)
ax.set_yticks(np.arange(0.77, 0.85, step=0.01))
fig.suptitle('Mean of Cross-Validation Scores')

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=cross_val_results_all.index, y=cross_val_results_all.apply(np.std))
ax.set_ylim(0.01, 0.06)
ax.set_yticks(np.arange(0.01, 0.06, step=0.005))
fig.suptitle('Standard Deviation of Cross-Validation Scores')

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

# ### 2. Random forrest

# +
y_test_predict_random = grid_random.predict(X_test)
submission_random_2 = pd.DataFrame({'PassengerId': passenger_id, 
                                    'Survived': y_test_predict_random})

existing_file = glob.glob('submission_random_2.csv')
assert (not existing_file), 'File already existed.'
submission_random_2.to_csv('submission_random_2.csv', index=False)
# (This submission got a public score of 0.799)
# -

# ### 3. Gradient boosting

# +
y_test_predict_gradient = grid_gradient.predict(X_test)
submission_gradient_2 = pd.DataFrame({'PassengerId': passenger_id, 
                                      'Survived': y_test_predict_gradient})

existing_file = glob.glob('submission_gradient_2.csv')
assert (not existing_file), 'File already existed.'
submission_gradient_2.to_csv('submission_gradient_2.csv', index=False)
# (This submission got a public score of 0.794)
# -

# ### 4. XGboost

# +
y_test_predict_xgb = grid_xgb.predict(X_test)
submission_xgb_2 = pd.DataFrame({'PassengerId': passenger_id, 'Survived': y_test_predict_xgb})

existing_file = glob.glob('submission_xgb_2.csv')
assert (not existing_file), 'File already existed.'
submission_xgb_2.to_csv('submission_xgb_2.csv', index=False)
# (This submission got a public score of 0.794)
# -

# ### 5. Multinomial Naive Bayes

# +
y_test_predict_mnb = grid_mnb.predict(X_test)
submission_mnb_2 = pd.DataFrame({'PassengerId': passenger_id, 'Survived': y_test_predict_mnb})

existing_file = glob.glob('submission_mnb_2.csv')
assert (not existing_file), 'File already existed.'
submission_mnb_2.to_csv('submission_mnb_2.csv', index=False)
# (This submission got a public score of 0.746)
# -

# ### 6. K Neareat Neighbors

# +
y_test_predict_knn = grid_knn.predict(X_test)
submission_knn_2 = pd.DataFrame({'PassengerId': passenger_id, 'Survived': y_test_predict_knn})

existing_file = glob.glob('submission_knn_2.csv')
assert (not existing_file), 'File already existed.'
submission_knn_2.to_csv('submission_knn_2.csv', index=False)
# (This submission got a public score of 0.789)
# -

# ### 7. Stack

# +
y_test_predict_random_stack = grid_random_stack.predict(X_test_oof_final)
submission_stack_2 = pd.DataFrame({'PassengerId': passenger_id, 
                                   'Survived':y_test_predict_random_stack})

existing_file = glob.glob('submission_stack_2.csv')
assert (not existing_file), 'File already existed.'
submission_stack_2.to_csv('submission_stack_2.csv', index=False)
# (This submission got a public score of 0.799)
# -

# ### 8. Finally perform a majority vote using all 7 model predictions

predict_array = np.array([y_test_predict_logis, y_test_predict_random, y_test_predict_gradient, 
                    y_test_predict_xgb, y_test_predict_mnb, y_test_predict_knn, 
                    y_test_predict_random_stack])

vote_df = pd.DataFrame(predict_array, index=['Logistic', 'Random Forrest', 'Gradirnt Boost', 
                                           'XGBoost', 'MN Bayes', 'Knn', 'Stack'])

y_test_predict_vote = np.array(vote_df.mode(axis=0))[0]

# +
submission_vote_2 = pd.DataFrame({'PassengerId': passenger_id, 'Survived': y_test_predict_vote})

existing_file = glob.glob('submission_vote_2.csv')
assert (not existing_file), 'File already existed.'
submission_vote_2.to_csv('submission_vote_2.csv', index=False)
# (This submission got a public score of 0.799)
# -

# ### 9. Public score comparison

fig, ax = plt.subplots(figsize=(11, 5))
ax.set_ylim(0.735, 0.805)
ax.set_yticks(np.arange(0.735, 0.805, step=0.01))
sns.barplot(x=['Logistic', 'Random Forrest', 'Gradirnt Boost', 'XGBoost', 'MN Bayes', 'Knn', 
           'Stack', 'Vote'], y=[0.794, 0.799, 0.794, 0.794, 0.746, 0.789, 0.799, 0.799])

pwd

test = pd.read_csv('submission_xgb_2_Kaggle.csv')
test_array = np.array(test['Survived'])
test_array

# compare with Kaggle with submission_xgb_2.csv
p = np.equal([0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1,
       1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1,
       1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1,
       1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1,
       1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
       0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
       0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1,
       1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1,
       0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0,
       1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
       0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1,
       0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0,
       0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
       0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
       1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0,
       0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0,
       1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1,
       0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1], 
            [0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1,
       1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1,
       1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1,
       1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1,
       1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
       0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
       0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1,
       1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1,
       0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0,
       1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
       0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1,
       0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0,
       0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
       0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
       1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0,
       0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0,
       1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1,
       0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1])
[i for i in p if ~i]


