# %% [markdown]
# ## Importation of important libraries

# %%
RD_STATE = 42
Z_ALPHA = 0.5

# Data manipulation
import kagglehub
import pandas as pd
import numpy as np

# Data visualisation
import seaborn as sns
import matplotlib.pyplot as plt

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

# Class balancing
from imblearn.over_sampling import SMOTE

# Model + evaluation
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

import warnings
import math

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

# %%
# Check distribution
numerical_cols = sch_db.select_dtypes(include=['int64', 'float64']).columns

def plot_numeric_distributions(df, cols):
    n = len(cols)
    rows = int(np.ceil(n / 3))
    palette = sns.color_palette("viridis", n)
    plt.figure(figsize=(20, rows * 4))
    for i, col in enumerate(cols, 1):
        plt.subplot(rows, 3, i)
        sns.histplot(df[col], kde=True, bins=30, color=palette[i-1])
        plt.title(f"Distribution of {col}", fontsize=12, fontweight='bold')
        plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
plot_numeric_distributions(sch_db, numerical_cols)

# %%
# Check percentage
plt.figure(figsize=(22, 18))

categorical_cols = sch_db.select_dtypes(include=['object', 'category']).columns
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(4, 4, i)
    sch_db[col].value_counts().plot.pie(
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=0.85,
        textprops={'fontsize': 10},
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
        cmap="tab20"
    )
    centre_circle = plt.Circle((0,0),0.50,color='white', fc='white')
    plt.gca().add_artist(centre_circle)
    plt.title(f"{col}", fontsize=12, fontweight='bold')
    plt.ylabel("")

plt.tight_layout()
plt.show()

# %%
def plot_histograms_by_target(df, target='Health_Issues', exclude_cols=None, bins=30, cols_per_row=3, palette='Dark2'):
    sns.set(style="whitegrid")
    # Missing target values were temporarily imputed for visualization purposes only during EDA.
    df[target].fillna('No', inplace=True)
    if exclude_cols is None:
        exclude_cols = []
    num_cols = [
        col for col in df.select_dtypes(include='number').columns
        if col not in exclude_cols + [target]
    ]
    n = len(num_cols)
    nrows = math.ceil(n / cols_per_row)
    fig, axes = plt.subplots(
        nrows,
        cols_per_row,
        figsize=(6.5 * cols_per_row, 4.8 * nrows)
    )
    axes = axes.flatten()
    unique_classes = sorted(df[target].unique())
    colors = sns.color_palette(palette, len(unique_classes))
    for i, col in enumerate(num_cols):
        ax = axes[i]
        for cls, color in zip(unique_classes, colors):
            subset = df[df[target] == cls][col]
            sns.kdeplot(
                subset,
                ax=ax,
                color=color,
                linewidth=2.5,
                label=str(cls),
                alpha=0.9
            )
            sns.histplot(
                subset,
                ax=ax,
                bins=bins,
                color=color,
                stat="density",
                alpha=0.25
            )
        ax.set_title(
            f"{col} vs {target}",
            fontsize=14,
            fontweight='bold'
        )
        ax.legend(
            title=target,
            fontsize=10,
            title_fontsize=11,
            frameon=True,
            facecolor="white",
            edgecolor="black"
        )
        ax.grid(alpha=0.25)
    for j in range(i + 1, len(axes)):
        axes[j].remove()
    plt.tight_layout()
    plt.show()

plot_histograms_by_target(sch_db, target='Health_Issues')

# %%
def show_plots(df, target='Health_Issues', cols_per_row=3, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = []

    cols = [col for col in df.columns if col not in exclude_cols + [target]]

    n = len(cols)
    nrows = math.ceil(n / cols_per_row)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=cols_per_row,
        figsize=(6 * cols_per_row, 4.5 * nrows)
    )

    axes = axes.flatten()

    for i, col in enumerate(cols):
        ax = axes[i]

        sns.boxplot(
            x=target,
            y=col,
            data=df,
            ax=ax,
            showfliers=False
        )

        sns.stripplot(
            x=target,
            y=col,
            data=df,
            ax=ax,
            jitter=True,
            color="black",
            alpha=0.4
        )

        ax.set_title(f"{col} vs {target}")
        ax.grid(alpha=0.3)

    # Supprimer les axes vides
    for j in range(i + 1, len(axes)):
        axes[j].remove()

    plt.tight_layout()
    plt.show()

show_plots(sch_db, target='Health_Issues')

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
#         + Done (removed outliers with IQR method)
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

# We have a problem with "severe" since it has too few data to be used properly
# We decided to mix "moderate" and "severe" into one single value "high risk".
sch_db['Health_Issues'] = sch_db['Health_Issues'].replace({
    'Severe': 'HighRisk',
    'Moderate': 'HighRisk'
})
# %% [markdown]
# #### 2. Removing duplicates and irrelevant variables

# %%
# Delete the "other" gender since non-sense
sch_db = sch_db[sch_db['Gender'] != 'Other']

# %%
# Here, the 'ID' column is irrelevant for the ML algorithm, so we can just drop it
sch_db = sch_db.drop(["ID"], axis=1)

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

# It seems there are no mistypes values nor inconsistencies
show_plots(sch_db, target='Health_Issues')

# %%
# https://www.statology.org/top-5-statistical-techniques-detect-handle-outliers-data
# Source for the IQR method for the outliers handling (modified a bit)
def detect_outliers_iqr(data, col):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return lower_bound, upper_bound


# Only apply on numerical columns
numeric_cols = sch_db.select_dtypes(include=['float64', 'int64']).columns
outliers_info = {}

for col in numeric_cols:
    print(sch_db[col].dtype)
    lb, ub = detect_outliers_iqr(sch_db, col)
    sch_db = sch_db[(sch_db[col] >= lb) & (sch_db[col] <= ub)]

show_plots(sch_db, target='Health_Issues')

# %% [markdown]
# # Data split train data vs test data
# Assign IV and TV

# %%
# Target value for the classification model
x = sch_db.drop('Health_Issues', axis=1)
y = sch_db['Health_Issues']
# Since our database is small (< 10k lines), we keep 20% of the data for testing purpose
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=RD_STATE)

print("\n################")
print(y_test.value_counts())
print("\n\n################")
print(y_train.value_counts())
print()

# Get a csv of all x and y
def all_to_csv(x_train, x_test, y_train, y_test):
    x_train.to_csv("x_train.csv")
    y_train.to_csv("y_train.csv")
    x_test.to_csv("x_test.csv")
    y_test.to_csv("y_test.csv")

# %% [markdown]
# ## **Data Preprocessing**
#
# * Goal: Prepare raw data for modelling or analysis.
# * Includes data cleaning, plus additional transformations, such as:
#     * Encoding categorical variables (e.g., one-hot encoding)
#       * Done
#     * Feature scaling (e.g., normalization, standardization)
#       * Done
#     * Feature selection/extraction
#        * Done
#     * Data transformation (e.g., log transformations, binning)
#       * Not necessary
#     * Handling imbalanced datasets
#        * Done

# %% [markdown]
# ### Label encoding


# %%

# Encode X with OHE

ohe_x = OneHotEncoder()
cols = ["Gender", "Country", "Sleep_Quality", "Stress_Level", "Occupation"]


def transformationEncoder(df: pd.DataFrame, colName: str) -> pd.DataFrame:
    onehot_train = ohe_x.fit_transform(df[[colName]])
    column_names = [colName + "_" + str(cat) for cat in ohe_x.categories_[0]]
    dfOneHot = pd.DataFrame(onehot_train.toarray(), columns=column_names, index=df.index)
    return pd.concat([df.drop(colName, axis=1), dfOneHot], axis=1)

# Apply label encoding to each column
for col in cols:
    x_train = transformationEncoder(x_train, col)
    x_test = transformationEncoder(x_test, col)

# %%

# Encode y with LabelEncoder

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# %% [markdown]
# ### Feature Scaling is useless for a Random Forest classifier

# %% [markdown]
# ### Class balancing
# %%
# We use SMOTE to balance our classes
smote = SMOTE(random_state=20)
x_train, y_train = smote.fit_resample(x_train, y_train)

# Convert np array into panda series
y_test = pd.Series(y_test)
y_train = pd.Series(y_train)

print("\n################")
print(y_test.value_counts())
print("\n\n################")
print(y_train.value_counts())
print()

# %% [markdown]
# ## Random Forest
# %%

# We first have to build the model :
random_forest_classifier = RandomForestClassifier(n_jobs=-1, random_state=RD_STATE)

# Let's find the best hyperparameters using GridSeachSV
hyperparameters = {
    "n_estimators":[100, 200],
    "max_depth":[None, 10, 20],
    "min_samples_split":[2, 5],
    "min_samples_leaf":[1, 2],
    "max_features":["sqrt", "log2"],
}

search = GridSearchCV(random_forest_classifier, hyperparameters, cv=5, scoring='accuracy')
search.fit(x_train, y_train)
print(f"Best parameters: {search.best_params_}")
print(f"Best Score: {search.best_score_}")

best_rf_classifier = search.best_estimator_


# Generate predictions on the unseen test set
y_predict = best_rf_classifier.predict(x_test)

print(y_predict)
# %% [markdown]
# ## Evaluation of the model
# %%

# Get the evaluation metrics
knn_acc = accuracy_score(y_test, y_predict)*100
knn_pre = precision_score(y_test, y_predict, average = 'weighted')
knn_recall = recall_score(y_test, y_predict, average = 'weighted')
knn_f1_ = f1_score(y_test, y_predict, average = 'weighted')

print("\nRF - Accuracy: {:.3f}".format(knn_acc))
print("RF - Precision: {:.3f}".format(knn_pre))
print("RF - Recall: {:.3f}".format(knn_recall))
print("RF - F1_Score: {:.3f}".format(knn_f1_))
print ('\n Classification Report:\n', classification_report(y_test, y_predict))
print('\n Confusion Matrix:\n', confusion_matrix(y_test, y_predict))
print()


