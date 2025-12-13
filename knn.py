# %% [markdown]
# ## Importation of important libraries

# %%
RD_STATE = 42
Z_ALPHA = 0.5
INTERRACTIVE = False
TEST_SIZE = 0.25

# Data manipulation
import kagglehub
import pandas as pd
import numpy as np

# DataViz
import seaborn as sns
import matplotlib.pyplot as plt

# Prerpoc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

import warnings

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_validate
from sklearn.metrics import make_scorer, multilabel_confusion_matrix, classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import roc_curve, roc_auc_score


warnings.filterwarnings("ignore")

# %% [markdown]
# ## Get the datasets from kaggle
#
# > By doing so, you can get the datasets only with the code

# %%
# Download latest version
sch_path = kagglehub.dataset_download("uom190346a/global-coffee-health-dataset")

print("Path to dataset file:", sch_path)

# %%
# Import dataset
sch_db = pd.read_csv(sch_path + '/synthetic_coffee_health_10000.csv')
sch_db

# %% [markdown]
# # DataViz

# %%
# Check the dataset
for col in sch_db.columns:
    print(col, sch_db[col].isnull().sum(), end=" ")
    print(sch_db[col].unique())

# Check dataset
sch_db.dtypes

# %% [markdown]
# ## **Data Cleaning**
# * Goal: Fix or remove incorrect, corrupted, or incomplete data.
# * Typical Tasks:
#     * Handling missing values (e.g., imputation or deletion)
#         + Done (removed NaN in Health Issues column) -> Replaced with 'No' class
#     * Removing duplicates and irrelevant variables
#         + Done (removed ID column)
#     * Fixing data entry errors (e.g., inconsistent capitalization or typos)
#         + Done (none)
#     * Correcting inconsistencies (e.g., "USA" vs. "United States") and incomplete values
#         + Done (none)
#     * Handling outliers (depending on the use case)
#         + detect_outliers_iqr
#

# %% [markdown]
# #### 1. Handling missing values

# %%
# Check whether there is missing values or not
sch_db.isnull().sum()

# %%
# There are missing values in 'Health Issues'
# Let's assume rows without data in 'Health Issues' represent a person with good health
sch_db['Health_Issues'].fillna('No', inplace=True)
print(sch_db['Health_Issues'].unique())


# %%

def show_plots():
    if not INTERRACTIVE:
      return
    # One way we can extend this plot is adding a layer of individual points on top of it through Seaborn's striplot
    # We'll use jitter=True so that all the points don't fall in single vertical lines above the species
    # Saving the resulting axes as ax each time causes the resulting plot to be shown on top of the previous axes
    for col in sch_db:
        ax = sns.boxplot(x='Health_Issues', y=str(col), data=sch_db)
        ax = sns.stripplot(x='Health_Issues', y=str(col), data=sch_db, jitter=True, edgecolor="gray")
        plt.show()


# %% [markdown]
# #### 2. Removing duplicates and irrelevant variables

# %%
# Delete the "other" gender since non-sense
sch_db = sch_db[sch_db['Gender'] != 'Other']
# Severe is deleting it is too "niche"
sch_db = sch_db[sch_db['Health_Issues'] != 'Severe']

# %%
# Here, the 'ID' column is irrelevant for the ML algorithm, so we can just drop it
sch_db = sch_db.drop(["ID"], axis=1)

# Try to standardize without Physical_Activity_Hours and Occupation
sch_db = sch_db.drop(["Physical_Activity_Hours"], axis=1)
sch_db = sch_db.drop(["Occupation"], axis=1)
sch_db = sch_db.drop(["Country"], axis=1)

# Handle every 1 as outliers worth to delete!
sch_db = sch_db.drop(["Smoking"], axis=1)

sch_db

# %% [markdown]
# #### 3.4.5. Fixing data entry errors, Inconsistencies, Handling outliers

# %%
# Let's check whether there are data entry errors or inconsistencies
for col in sch_db.columns:
    print("=====" + col + "=====")
    if len(x := sch_db[col].value_counts()) < 50:
        print(x)
    print(sch_db[col].describe())
    print("\n")


# high Physical_Activity_Hours !

# It seems there are no mistypes values nor inconsistencies

show_plots()

# %%
# https://www.statology.org/top-5-statistical-techniques-detect-handle-outliers-data
# Source for the IQR method for the outliers handling (modified)
def detect_outliers_iqr(data, col):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    # >1.5 (up to 3) don't change the amount of rows (a lot) so stay with the default
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return lower_bound, upper_bound


# Appliquer uniquement aux colonnes numériques
numeric_cols = sch_db.select_dtypes(include=['float64', 'int64']).columns
outliers_info = {}

for col in numeric_cols:
    print(sch_db[col].dtype)
    lb, ub = detect_outliers_iqr(sch_db, col)
    sch_db = sch_db[(sch_db[col] >= lb) & (sch_db[col] <= ub)]

show_plots()
# Remaining values after outliers cleaning: 7671 (for n*IQR:=1.5*IQR)
sch_db

# %% [markdown]
# # Data split train data vs test data
# Assign IV and TV

# %%
# Target value for the classification model
x = sch_db.drop('Health_Issues', axis=1)
y = sch_db['Health_Issues']
# Since our database is a bit small (< 10k lines), we keep 20% of the data for testing purpose
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=RD_STATE)


def all_to_csv(x_train, x_test, y_train, y_test):
    x_train.to_csv("x_train.csv")
    y_train.to_csv("y_train.csv")
    x_train.to_csv("x_test.csv")
    y_test.to_csv("y_test.csv")


# %% [markdown]
# ## **Data Preprocessing**
#
# * Goal: Prepare raw data for modelling or analysis.
# * Includes data cleaning, plus additional transformations, such as:
#     * Encoding categorical variables (e.g., one-hot encoding)
#       * Done
#     * Feature scaling (e.g., normalization, standardization)
#       * Done - delete outliers
#     * Feature selection/extraction
#        * Done previously
#     * Data transformation (e.g., log transformations, binning)
#     * Handling imbalanced datasets
#        * In progress


# %% [markdown]
# ### Label encoding


# %%

### ENCODE X ###

ohe_x = OneHotEncoder()
cols = ["Gender", "Sleep_Quality", "Stress_Level"]#, "Occupation", "Country"]


def transformationEncoder(df: pd.DataFrame, colName: str) -> pd.DataFrame:
    onehot_train = ohe_x.fit_transform(df[[colName]])
    column_names = [colName + "_" + str(cat) for cat in ohe_x.categories_[0]]
    dfOneHot = pd.DataFrame(onehot_train.toarray(), columns=column_names, index=df.index)
    return pd.concat([df.drop(colName, axis=1), dfOneHot], axis=1)

def transformationEncoderTestTrain(df_train: pd.DataFrame, df_test: pd.DataFrame, colName: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_train = transformationEncoder(df_train, colName)
    df_test = transformationEncoder(df_test, colName)
    
    return df_train, df_test

"""for col in cols:
    x_train = transformationEncoder(x_train, col)
    x_test = transformationEncoder(x_test, col)

def transformationEncoder(df_train: pd.DataFrame, df_test: pd.DataFrame, colName: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    le = LabelEncoder()

    df_train[colName] = le.fit_transform(df_train[colName])
    df_test[colName] = le.transform(df_test[colName])

    return df_train, df_test"""

# List of columns to label encode
cols = ["Gender", "Sleep_Quality", "Stress_Level"]  #, "Occupation", "Country"]

# Apply label encoding to each column
for col in cols:
    x_train, x_test = transformationEncoderTestTrain(x_train, x_test, col)


# %%

### ENCODE Y ###

y_train_dumb = pd.get_dummies(y_train, columns=["Health_Issues"], dtype=int)
y_test_dumb = pd.get_dummies(y_test, columns=["Health_Issues"], dtype=int)
y_test = y_test_dumb
y_train = y_train_dumb

#le = LabelEncoder()
#y_train = le.fit_transform(y_train)
#y_test = le.transform(y_test)

#y_train, y_test = transformationEncoderTrainTest(pd.DataFrame(y_train), pd.DataFrame(y_test), "Health_Issues")


#all_to_csv(x_train, x_test, y_train, y_test)
y_train


## %% [mardown]
## ### Class Balancing before encoding
## BAD IDEA 
#
##%%
#from collections import Counter
#from imblearn.over_sampling import ADASYN
#x_train, y_train = ADASYN(random_state=RD_STATE).fit_resample(x_train, y_train)
#all_to_csv(x_train, x_test, y_train, y_test)

# %% [markdown]
# ### Feature Scaling

# %%

# We have to normalize with standardization [-1,1] since almost any data folls gausian distrib:

# STANDARDIZATION

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# pd.DataFrame(x_train_norm).to_csv("norm.csv")
# %% [markdown]
# ## KNN
# %%

# We first have to build the model :
knn_classifier = KNeighborsClassifier(n_jobs=-1)

# Let's find the best hyperparameters
hyperparameters = {
    "n_neighbors": [3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
    "weights": ["uniform", "distance"],
    "metric": ["minkowski", "manhattan", "chebyshev", "euclidean"]
}

search = GridSearchCV(knn_classifier, hyperparameters, cv=5, scoring='accuracy')
search.fit(x_train, y_train)
print(f"Best parameters: {search.best_params_}")
print(f"Best Score: {search.best_score_}")

best_knn_classifier = search.best_estimator_

y_predict = best_knn_classifier.predict(x_test)

y_predict
# %% [markdown]
# ## Evaluation of the model
# %%

# Get the accuracy score
knn_acc = accuracy_score(y_test, y_predict)*100
knn_pre = precision_score(y_test, y_predict, average = 'weighted')
knn_recall = recall_score(y_test, y_predict, average = 'weighted')
knn_f1_ = f1_score(y_test, y_predict, average = 'weighted')

print("\nKNN - Accuracy: {:.3f}".format(knn_acc))
print("KNN - Precision: {:.3f}".format(knn_pre))
print("KNN - Recall: {:.3f}".format(knn_recall))
print("KNN - F1_Score: {:.3f}".format(knn_f1_))
print ('\n Classification Report:\n', classification_report(y_test, y_predict))
print()

print(f"Shapes : [ {y_test.shape} | {y_predict.shape} ]")

conf_mat = multilabel_confusion_matrix(y_test, y_predict)
print("Confusion matrix :")
print(conf_mat)

#%%

print()
# Wrap it with MultiOutputClassifier
multi_label_classifier = MultiOutputClassifier(best_knn_classifier, n_jobs=-1)

# Stratifies KFold is impossible in multilabel prediction
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=RD_STATE)

# Define scorers with zero_division=0
scoring = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted'),
    'f1': make_scorer(f1_score, average='weighted'),
    'roc_auc': make_scorer(roc_auc_score)
}

cv_results = cross_validate(multi_label_classifier, x_train, y_train, cv=rkf, scoring=scoring)


for score in cv_results:
    print(f"KNN (RepeatedKFold) - {score}: {np.mean(cv_results[score]):.3f}")

# %%
