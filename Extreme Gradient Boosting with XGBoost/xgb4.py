import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import mean_squared_error 
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing.imputation import Imputer
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import cross_val_score
from sklearn_pandas import DataFrameMapper
from sklearn_pandas import CategoricalImputer
from sklearn.preprocessing.imputation import Imputer
from sklearn.impute import SimpleImputer

#%%

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#%%

df=pd.read_csv('data/ames_unprocessed_data.csv')

#%%

# Import LabelEncoder
from sklearn.preprocessing import LabelEncoder

# Fill missing values with 0
df.LotFrontage = df.LotFrontage.fillna(0)

# Create a boolean mask for categorical columns
categorical_mask = (df.dtypes == object)

# Get list of categorical column names
categorical_columns = df.columns[categorical_mask].tolist()

# Print the head of the categorical columns
print(df[categorical_columns].head())

# Create LabelEncoder object: le
le = LabelEncoder()

# Apply LabelEncoder to categorical columns
df[categorical_columns] = df[categorical_columns].apply(lambda x: le.fit_transform(x))

# Print the head of the LabelEncoded categorical columns
print(df[categorical_columns].head())


#%%

# Import OneHotEncoder
from sklearn.preprocessing import OneHotEncoder

# Create OneHotEncoder: ohe
ohe = OneHotEncoder(categorical_features=categorical_mask,sparse=False)

# Apply OneHotEncoder to categorical columns - output is no longer a dataframe: df_encoded
df_encoded = ohe.fit_transform(df)

# Print first 5 rows of the resulting dataset - again, this will no longer be a pandas dataframe
print(df_encoded[:5, :])

# Print the shape of the original DataFrame
print(df.shape)

# Print the shape of the transformed array
print(df_encoded.shape)

#%%

df=pd.read_csv('data/ames_unprocessed_data.csv')

#%%

# =============================================================================
# encode any categorical columns in the dataset to numerically value 
# =============================================================================
# Import DictVectorizer
from sklearn.feature_extraction import DictVectorizer

# Convert df into a dictionary: df_dict
df_dict = df.to_dict("records")

# Create the DictVectorizer object: dv
dv = DictVectorizer(sparse=False)

# Apply dv on df: df_encoded
df_encoded = dv.fit_transform(df_dict)

# Print the resulting first five rows
print(df_encoded[:5,:])

# Print the vocabulary
print(dv.vocabulary_)

print(df.shape)
print(df_encoded.shape)

#%%

df=pd.read_csv('data/ames_unprocessed_data.csv')

X=df.iloc[:,:-1]
y=df.iloc[:,-1]

#%%

# Import necessary modules
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

# Fill LotFrontage missing values with 0
X.LotFrontage = X.LotFrontage.fillna(0)

# Setup the pipeline steps: steps
steps = [("ohe_onestep", DictVectorizer(sparse=False)),
         ("xgb_model", xgb.XGBRegressor())]

# Create the pipeline: xgb_pipeline
xgb_pipeline = Pipeline(steps)

# Fit the pipeline
xgb_pipeline.fit(X.to_dict("records"),y)

#%%

df=pd.read_csv('data/ames_unprocessed_data.csv')

X=df.iloc[:,:-1]
y=df.iloc[:,-1]

#%%

# Import necessary modules
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# Fill LotFrontage missing values with 0
X.LotFrontage = X.LotFrontage.fillna(0)

# Setup the pipeline steps: steps
steps = [("ohe_onestep", DictVectorizer(sparse=False)),
         ("xgb_model", xgb.XGBRegressor(max_depth=2, objective="reg:linear"))]

# Create the pipeline: xgb_pipeline
xgb_pipeline = Pipeline(steps)

# Cross-validate the model
cross_val_scores = cross_val_score(xgb_pipeline,X.to_dict("records"),y,cv=10,scoring="neg_mean_squared_error")

# Print the 10-fold RMSE
print("10-fold RMSE: ", np.mean(np.sqrt(np.abs(cross_val_scores))))


#%%

from sklearn.preprocessing import LabelEncoder
df=pd.read_csv('data/chronic_kidney_disease.csv',na_values=["?"])
X=df.iloc[:,:-1]
y=df.iloc[:,-1]
y=LabelEncoder().fit_transform(y)

#%%

# Import necessary modules
from sklearn_pandas import DataFrameMapper
from sklearn_pandas import CategoricalImputer
from sklearn.preprocessing.imputation import Imputer
from sklearn.impute import SimpleImputer

# Check number of nulls in each feature column
nulls_per_column = X.isnull().sum()
print(nulls_per_column)

# Create a boolean mask for categorical columns
categorical_feature_mask = X.dtypes == object

# Get list of categorical column names
categorical_columns = X.columns[categorical_feature_mask].tolist()

# Get list of non-categorical column names
non_categorical_columns = X.columns[~categorical_feature_mask].tolist()

# Apply numeric imputer
numeric_imputation_mapper = DataFrameMapper(
        [([numeric_feature], SimpleImputer(strategy="median")) for numeric_feature in non_categorical_columns],
        input_df=True,
        df_out=True
        )

# Apply categorical imputer
categorical_imputation_mapper = DataFrameMapper(
        [(category_feature, CategoricalImputer()) for category_feature in categorical_columns],
        input_df=True,
        df_out=True
        )

#%%

# Import FeatureUnion
from sklearn.pipeline import FeatureUnion

# Combine the numeric and categorical transformations
numeric_categorical_union = FeatureUnion([
                                          ("num_mapper", numeric_imputation_mapper),
                                          ("cat_mapper", categorical_imputation_mapper)
                                         ])


#%%
from sklearn.base import BaseEstimator, TransformerMixin

class Dictifier(BaseEstimator, TransformerMixin):
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        if type(X) == pd.core.frame.DataFrame:
            return X.to_dict("records")
        else:
            return pd.DataFrame(X).to_dict("records")
    
    
   
    
#%%
# Create full pipeline
pipeline = Pipeline([
                     ("featureunion", numeric_categorical_union),
                     ("dictifier", Dictifier()),
                     ("vectorizer", DictVectorizer(sort=False)),
                     ("clf", xgb.XGBClassifier(max_depth=3))
                    ])

# Perform cross-validation
cross_val_scores = cross_val_score(pipeline, X, y, scoring="roc_auc", cv=3)

# Print avg. AUC
print("3-fold AUC: ", np.mean(cross_val_scores))

#%%

from sklearn.preprocessing import LabelEncoder
df=pd.read_csv('data/chronic_kidney_disease.csv',na_values=["?"])
X=df.iloc[:,:-1]
y=df.iloc[:,-1]
y=LabelEncoder().fit_transform(y)

#%%
# Create the parameter grid
gbm_param_grid = {
    'clf__learning_rate': np.arange(0.05,1.05,0.05),
    'clf__max_depth': range(3, 10),
    'clf__n_estimators': range(50, 200, 50)
}

# Perform RandomizedSearchCV
randomized_roc_auc = RandomizedSearchCV(estimator=pipeline,param_distributions=gbm_param_grid,verbose=1, scoring='roc_auc' ,n_iter=2, cv=2)

# Fit the estimator
randomized_roc_auc.fit(X,y)

# Compute metrics
print('\n')
print(randomized_roc_auc.best_score_)
print('\n')
print(randomized_roc_auc.best_estimator_)
print('\n')
print(randomized_roc_auc.best_params_)



