# %% [markdown]
# ## Importation of important libraries

# %%
RD_STATE=42
Z_ALPHA = 0.5

# Data manipulation
import kagglehub
import pandas as pd

# Prerpoc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


import warnings 
warnings.filterwarnings("ignore")

# %% [markdown]
# ## Get the datasets from kaggle
# 
# > By doing so, you can get the datasets only with the code

# %%
# Download latest version
sch_path = kagglehub.dataset_download("uom190346a/global-coffee-health-dataset")

print("Path to dataset files:", sch_path)

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
#         * Done (removed NaN in Health Issues column)
#     * Removing duplicates and irrelevant variables
#         * Done (removed ID column)
#     * Fixing data entry errors (e.g., inconsistent capitalization or typos)
#         * Done (none)
#     * Correcting inconsistencies (e.g., "USA" vs. "United States") and incomplete values
#         * Done (none)
#     * Handling outliers (depending on the use case)
#         * TODO
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

sch_db

# %% [markdown]
# #### 3.4.5. Fixing data entry errors, Inconsistencies, Handling outliers

# %%
# Let's check whether there are data entry errors, inconsistencies or outliers in the DB
for col in sch_db.columns:
    print("=====" + col + "=====")
    if len(x:= sch_db[col].value_counts()) < 50:
        print(x)
    print(sch_db[col].describe())
    print("\n")
    
# high Physical_Activity_Hours !

# It seems there are no mistypes values, no inconsistencies and not outliers

# %% [markdown]
# # Data split train data vs test data 
# Assign IV and TV

# %%
# Target value for the classification model
x = sch_db.drop('Health_Issues', axis = 1)
y = sch_db['Health_Issues']
# Since our database is small (< 10k lines), we keep 20% of the data for testing purpose
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = RD_STATE)

# %% [markdown]
# ## **Data Preprocessing**
# 
# * Goal: Prepare raw data for modelling or analysis.
# * Includes data cleaning, plus additional transformations, such as:
#     * Encoding categorical variables (e.g., one-hot encoding)
#     * Feature scaling (e.g., normalization, standardization)
#     * Feature selection/extraction
#     * Data transformation (e.g., log transformations, binning)
#     * Handling imbalanced datasets

# %% [markdown]
# Label encoding


# %%

### ENCODE X ###

ohe_x = OneHotEncoder()
cols = ["Gender", "Country", "Sleep_Quality", "Stress_Level", "Occupation"]

def transformationEncoder(df: pd.DataFrame, colName: str) -> pd.DataFrame:
  onehot_train = ohe_x.fit_transform(df[[colName]])
  column_names = [colName + "_" + str(cat) for cat in ohe_x.categories_[0]]
  dfOneHot = pd.DataFrame(onehot_train.toarray(), columns = column_names, index = df.index)
  return pd.concat([df.drop(colName, axis = 1), dfOneHot], axis = 1)

for col in cols:
  x_train = transformationEncoder(x_train , col)
  x_test = transformationEncoder(x_test , col)
  
# %% 

### ENCODE Y ###

print(y)
ohe_y = OneHotEncoder()
onehot_train = ohe_y.fit_transform(y)
print(onehot_train)
