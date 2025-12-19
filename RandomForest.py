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
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_validate, RandomizedSearchCV
from sklearn.metrics import make_scorer, classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, RocCurveDisplay

from sklearn.model_selection import ValidationCurveDisplay, LearningCurveDisplay

from imblearn.over_sampling import SMOTE

from sklearn.metrics import roc_curve, roc_auc_score

from sklearn.base import clone

from sklearn.preprocessing import LabelBinarizer


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

# Try to standardize without Physical_Activity_Hours and Occupation
#sch_db = sch_db.drop(["Physical_Activity_Hours"], axis=1)
#sch_db = sch_db.drop(["Occupation"], axis=1)
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

print(x.columns)

# Since our database is small (< 10k lines), we keep 20% of the data for testing purpose
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=RD_STATE)

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


cols = ["Gender", "Sleep_Quality", "Stress_Level", "Occupation"] # Country

def transformationEncoder(df_train, df_test, colName: str) -> pd.DataFrame:
    ohe_x = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ohe_x.fit(df_train[[colName]])
    column_names = [colName + "_" + str(cat) for cat in ohe_x.categories_[0]]

    #onehot_train = ohe_x.fit_transform(df[[colName]])
    #column_names = [colName + "_" + str(cat) for cat in ohe_x.categories_[0]]
    #dfOneHot = pd.DataFrame(onehot_train.toarray(), columns=column_names, index=df.index)
    #return pd.concat([df.drop(colName, axis=1), dfOneHot], axis=1)
    
    # Transform training data
    onehot_train = ohe_x.transform(df_train[[colName]])
    df_train_onehot = pd.DataFrame(onehot_train, columns=column_names, index=df_train.index)
    df_train = pd.concat([df_train.drop(colName, axis=1), df_train_onehot], axis=1)

    # Transform test data
    onehot_test = ohe_x.transform(df_test[[colName]])
    df_test_onehot = pd.DataFrame(onehot_test, columns=column_names, index=df_test.index)
    df_test = pd.concat([df_test.drop(colName, axis=1), df_test_onehot], axis=1)
    
    return df_train, df_test


# Apply label encoding to each column
for col in cols:
    x_train, x_test = transformationEncoder(x_train, x_test, col)


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

# We have to normalize :

#normalize = MinMaxScaler()
#x_train = normalize.fit_transform(x_train)
#x_test = normalize.transform(x_test)



# pd.DataFrame(x_train_norm).to_csv("norm.csv")

# %% [markdown]
# ## RF UNDERFIT

# %%

# Remove 90% of the training records (keep only 75%)
#subset_fraction = 0.5
#subset_indices = np.random.choice(len(x_train), size=int(subset_fraction * len(x_train)), replace=False)

# Subset the data
#x_train_subset = x_train.iloc[subset_indices] if hasattr(x_train, 'iloc') else x_train[subset_indices]
#y_train_subset = y_train.iloc[subset_indices] if hasattr(y_train, 'iloc') else y_train[subset_indices]

#x_train = x_train_subset
#y_train = y_train_subset

# %% [markdown]
# ### Oversampling
#%%
smote = SMOTE(random_state=RD_STATE)
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
# Let's find the best hyperparameters
#hyperparameters = {
#    "n_estimators":[100, 200],
#    "max_depth":[None, 10, 20],
#    "min_samples_split":[2, 5, 10],
#    "min_samples_leaf":[1, 2, 4],
#    "max_features":["sqrt", "log2"],
#}
from scipy.stats import randint

# Define the search space
param_distributions = {
    'n_estimators': randint(100, 500),         # Random integer between 100 and 500
    'max_depth': [i for i in range(1,11)],     # Discrete list
    'min_samples_split': randint(2, 11),       # Random integer between 2 and 10
    'min_samples_leaf': randint(1, 5),         # Random integer between 1 and 4
    'max_features': ['sqrt'],                  # Categorical
}

#search = GridSearchCV(random_forest_classifier, hyperparameters, cv=5, scoring='accuracy')
# Initialize RandomizedSearchCV
# n_iter=50 means it will try 50 random combinations
search = RandomizedSearchCV(
    estimator=random_forest_classifier,
    param_distributions=param_distributions,
    n_iter=50, 
    cv=3, 
    scoring='recall_macro', 
    verbose=1, 
    random_state=RD_STATE,
    n_jobs=-1
)

if False:
    search.fit(x_train, y_train)
    print(f"Best parameters: {search.best_params_}")
    print(f"Best Score: {search.best_score_}")


    # The best parameters found by GridSearchCV
    # 'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100
    # The best parameters found by RandomSearchCV
    # {'max_depth': 6, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 8, 'n_estimators': 225}


    best_rf_classifier = search.best_estimator_
else:
    best_rf_classifier = RandomForestClassifier(n_jobs=-1, random_state=RD_STATE, **{'max_depth': 3, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 8, 'n_estimators': 225})


#%%
best_rf_classifier.fit(x_train, y_train)
y_predict = best_rf_classifier.predict(x_test)
print(y_predict)


# %% [markdown]
# ## Evaluation of the model
# %%

# Get the evaluation metrics
knn_acc = accuracy_score(y_test, y_predict)*100
knn_pre = precision_score(y_test, y_predict, average = 'weighted')
knn_recall = recall_score(y_test, y_predict, average = 'macro')
knn_f1_ = f1_score(y_test, y_predict, average = 'weighted')

print("\nRF - Accuracy: {:.3f}".format(knn_acc))
print("RF - Precision: {:.3f}".format(knn_pre))
print("RF - Recall: {:.3f}".format(knn_recall))
print("RF - F1_Score: {:.3f}".format(knn_f1_))
print ('\n Classification Report:\n', classification_report(y_test, y_predict))
print('\n Confusion Matrix:\n', confusion_matrix(y_test, y_predict))
print()

#%%

ValidationCurveDisplay.from_estimator(
   best_rf_classifier, x_train, y_train, param_name="max_depth", param_range=[i for i in range(1, 10)], n_jobs=-1,
   scoring="accuracy", cv=5
)

plt.title('Validation Curve for RandomForest (max_depth)')
plt.grid(True)
plt.show()

LearningCurveDisplay.from_estimator(
   best_rf_classifier, x_train, y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=5, n_jobs=-1,
   scoring="accuracy"
)

plt.title('Learning Curve for RandomForest')
plt.grid(True)
plt.show()

#%%

# Binarize labels (required for multiclass ROC)
lb = LabelBinarizer().fit(y_train)
y_onehot_test = lb.transform(y_test)
y_score_rf = best_rf_classifier.predict_proba(x_test)

fig, ax = plt.subplots()

for i, class_name in enumerate(lb.classes_):
    RocCurveDisplay.from_predictions(
        y_onehot_test[:, i],
        y_score_rf[:, i],
        name=f"ROC curve for {class_name}",
        ax=ax,
    )

plt.plot([0, 1], [0, 1], "k--", label="Random algorithm (AUC = 0.5)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Random Forest Multi-class ROC (One-vs-Rest)")
plt.grid(True)
plt.legend()
plt.show()
#%%
exit()
print()

# Stratifies KFold is impossible in multilabel prediction
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=RD_STATE)

# Define scorers with zero_division=0
scoring = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted'),
    'f1': make_scorer(f1_score, average='weighted')
}

cv_results = cross_validate(best_rf_classifier, x_train, y_train, cv=rkf, scoring=scoring)


for score in cv_results:
    print(f"KNN (RepeatedKFold) - {score}: {np.mean(cv_results[score]):.3f}")
    
# %%