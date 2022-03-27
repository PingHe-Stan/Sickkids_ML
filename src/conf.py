"""
File Paths, Global variables for modelling
"""
__author__ = 'Stan He@Sickkids.ca'

PATH = '../data/'

BIRTH_DATA = PATH + 'Birth Q107CBIRTHCD no withdraw.xlsx'

# Dir
CHILD_RAW_DIR = '../output/'
ADDON_DIR = '../data/addon/'

# Files to load
CHILD_DATA_PATH = CHILD_RAW_DIR + "CHILD_raw.xlsx"
CHILD_BREASTFEEDING_PATH = ADDON_DIR + "breastfeeding data.xlsx"
CHILD_WHEEZEPHENOTYPE_PATH = ADDON_DIR + "wheeze phenotypes.xlsx"
CHILD_ETHNICITY_PATH = ADDON_DIR + "Prenatal Q91PRNMH18WK.xlsx"
CHILD_SPT_PATH = ADDON_DIR + "Child SPT 1y 3y 5y.xlsx"

# Variables category

FEATURE_SELECTED_BASELINE = [
    "Gender",
    "Mode_of_delivery",
    "Anesthetic_delivery",
    "Prolonged_Expiration_1yCLA",
    "Crackles_1yCLA",
    "Wheeze_1yCLA",
    "Mother_Asthma",
    "Father_Asthma",
    "Parental_Asthma",
    "Mom_Atopy",
    "Wheeze_Mother",
    "Hayfever_Mother",
    "Pollentress_Mother",
    "FAllergies_Mother",
    "Wheeze_Father",
    "Hayfever_Father",
    "AD_Father",
    "Pollentress_Father",
    "Number_of_AntibioticsCourse",
    "Antibiotics_Usage",
    "Gest_Days",
    "Apgar_Score_5min",
    "y",
]

# Categorical Features
SUB_CAT_MUST = [
    "Antibiotics_Usage",
]
SUB_CAT_BINARY_NOMISSING = [
    "F10min_Intubation",
    "F10min_Mask_Ventilation",
    "F10min_Free_Flow_Oxygen",
    "F10min_Oxygen_Mask",
    "F10min_Positive_Pressure_Ventilation",
    "F10min_Perineum_suction",
    "F10min_Suction",
    "F10min_No_Measure_Needed",
    "Prenatal_Bleeding",
    "Prenatal_Nausea",
    "Prenatal_Infections",
    "Prenatal_Induced_Hypertension",
    "Prenatal_Gestational_Diabetes",
    "Prenatal_Cardiac_Disorder",
    "Prenatal_Hypertension",
    "Prenatal_Hypotension",
    "Prenatal_None_Conditions",
    "Prenatal_Other_Conditions",
]
SUB_CAT_BINARY_WITHMISSING = [
    "BF_1m",
    "BF_9m",
    "BF_12m",
    "Anesthetic_delivery",
    "Analgesics_usage_delivery",
    "Respiratory_Problems_Birth",
    "Jaundice_Birth",
    "Complications_Birth",
    "Home_Furry_Pets_6m",
    "Home_New_Furnitures_6m",
    "Home_Presence_Smoke_6m",
    "Mother_Asthma",
    "Father_Asthma",
    "Parental_Asthma",
    "Father_Caucasian",
    "Mother_Caucasian",
    "Dad_Atopy",
    "Dad_Food",
    "Dad_Inhalant",
    "Mom_Atopy",
    "Mom_Food",
    "Mom_Inhalant",
    "Prenatal_Second_Hand",
    "Prenatal_Maternal_Smoke",
    'Prolonged_Expiration_1yCLA',
    'Crackles_1yCLA',
    'Wheeze_1yCLA',
    'Wheeze_Mother',
    'Asthma_Mother',
    'Hayfever_Mother',
    'AD_Mother',
    'Pollentress_Mother',
    'FAllergies_Mother',
    'Wheeze_Father',
    'Asthma_Father',
    'Hayfever_Father',
    'AD_Father',
    'Pollentress_Father',
    'FAllergies_Father'
]
SUB_CAT_OHE = [
    "Study_Center",  # No missing
    "Gender",  # No missing
    "BF_Status_3m",  # Unknown represent missing
    "BF_Status_6m",  # Unknown represent missing
    "Child_Ethinicity",  # np.nan represent missing
]
SUB_CAT_ENGI = [
    "Mode_of_delivery",
    "Prenatal_Mother_Condition",
    "First_10min_Measure",
]
SUB_CAT_RESP = [
    "Respiratory_Report_Months",
    "Respiratory_Infections",
    "Severity_of_Respiratoryinfections",
]
SUB_CAT_WHEEZE_EARLY = [
    "Wheeze_3m",
    "Noncold_Wheeze_3m",
]
CAT_VARS_REPO = SUB_CAT_MUST + SUB_CAT_BINARY_NOMISSING + SUB_CAT_BINARY_WITHMISSING + SUB_CAT_OHE + SUB_CAT_ENGI + SUB_CAT_RESP + SUB_CAT_WHEEZE_EARLY

# Numeric Features
SUB_NUM_WITHMISSING = [
    "Gest_Days",
    "Weight_0m",
    "Weight_3m",
    "Weight_12m",
    "BF_Implied_Duration",
    "PSS_36week",
    "CESD_36week",
    "PSS_18week",
    "CESD_18week",
    "PSS_6m",
    "PSS_12m",
    "CESD_6m",
    "CESD_12m",
]
SUB_NUM_LOG1P = [
    "No_of_Pregnancy",
    "Stay_Duration_Hospital",
    "Number_of_AntibioticsCourse",
    "Epi_Noncold_Wheeze_3m",
]
SUB_NUM_ORDINAL = ["Apgar_Score_1min", "Apgar_Score_5min"]
SUB_NUM_HOMEDUST = [
    "Home_DEP_3m",
    "Home_DiBP_3m",
    "Home_DNBP_3m",
    "Home_BzBP_3m",
    "Home_DEHP_3m",
]
NUM_VARS_REPO = SUB_NUM_WITHMISSING + SUB_NUM_LOG1P + SUB_NUM_ORDINAL + SUB_NUM_HOMEDUST

# Target Variables
TARGET_OF_SELECT = [
    "Asthma_Diagnosis_3yCLA",
    "Asthma_Diagnosis_5yCLA",
    "Recurrent_Wheeze_1y",  # Self-report Wheeze at earliest time point
    "Recurrent_Wheeze_3y",
    "Recurrent_Wheeze_5y",
    "Wheeze_Traj_Type",  # Derived by Vera Dai, Less NaN Value, 2+4 could be useful
    "Medicine_for_Wheeze_5yCLA",  # More objective
    "Viral_Asthma_3yCLA",  # No need to decide "possible" category
    "Triggered_Asthma_3yCLA",
    "Viral_Asthma_5yCLA",  # No need to decide "possible" category
    "Triggered_Asthma_5yCLA",
]
TARGET_TO_ENGI = [
    "Cumulative_Wheeze_36m",  # No missing value
    "Cumulative_Wheeze_60m",  # No missing value
]
TARGET_VAR_REPO = TARGET_OF_SELECT + TARGET_TO_ENGI

# Features to drop
FEA_OF_TOPIC = [
    "Time_of_AntibioticsUsage",  # Could be used like the HOME DUST availability data to see if there is a non-linear relationship between time and asthma outcome
]
FEA_OF_INTEREST = [
    "BF_18m",  # Too much missing value
    "BF_24m",
]
FEA_OF_LITTLE = [
    "Percentage_OutdoorsActivity_6m",  # Majority Missing
    "Weight_36m",
    "Weight_60m",  # Too late
    "Weight_for_age_0m",
    "Weight_for_age_3m",
    "Weight_for_age_12m",  # Correlated with Weight_0m
    "Weight_for_age_36m",
    "Weight_for_age_60m",  # Too late
    "PSS_18m",  # Too late , No of Missing Value
    "PSS_24m",  # Too late , No of Missing Value
    "CESD_18m",  # See Above
    "CESD_24m",  # See Above
    "BF_3m",  # Repetitive Information with BF_Status_3m
    "BF_6m",  # Repetitive Information with BF_Status_6m
    "Systolic_BP_3yCLA",
    "Diastolic_BP_3yCLA",
    "Pulse_Rate_3yCLA",
    "Wheeze_3yCLA",  # Could be a potential target variable
    "Wheeze_5yCLA",  # Could be a potential target variable
    "Wheeze_withoutcold_5yCLA",  # See Above
    "Medical_Conditions_5yCLA",  # See Above
    "Regular_Controller_5yCLA",
    "Intermittent_Controller_5yCLA",
    "Reliever_5yCLA",
    "Frequency_Oral_Steroid_5yCLA",
    "Wheeze_Frequency_5yCLA",
    "Systolic_BP_5yCLA",
    "Diastolic_BP_5yCLA",
    "Pulse_Rate_5yCLA",
]
FEA_OF_UNUSEDWHEEZE = list({"Wheeze_3m", "Noncold_Wheeze_3m", "Epi_Noncold_Wheeze_3m", "Wheeze_6m", "Noncold_Wheeze_6m",
     "Epi_Noncold_Wheeze_6m", "Wheeze_1y", "Noncold_Wheeze_1y", "Epi_Noncold_Wheeze_1y", "Wheeze_18m",
     "Noncold_Wheeze_18m", "Epi_Noncold_Wheeze_18m", "Wheeze_2y", "Noncold_Wheeze_2y", "Epi_Noncold_Wheeze_2y",
     "Wheeze_2yh", "Noncold_Wheeze_2hy", "Epi_Noncold_Wheeze_2hy", "Wheeze_3y", "Noncold_Wheeze_3y",
     "Epi_Noncold_Wheeze_3y", "Wheeze_4y", "Noncold_Wheeze_4y", "Epi_Noncold_Wheeze_4y", "Wheeze_5y",
     "Noncold_Wheeze_5y", "Epi_Noncold_Wheeze_5y", "Cumulative_Wheeze_3m", "Cumulative_Wheeze_6m",
     "Cumulative_Wheeze_12m", "Cumulative_Wheeze_18m", "Cumulative_Wheeze_24m", "Cumulative_Wheeze_30m",
     "Cumulative_Wheeze_36m", "Cumulative_Wheeze_48m"} - set(TARGET_TO_ENGI) - set(SUB_CAT_WHEEZE_EARLY) - {"Epi_Noncold_Wheeze_3m"})
FEA_TO_DROP = FEA_OF_TOPIC + FEA_OF_INTEREST + FEA_OF_UNUSEDWHEEZE + FEA_OF_LITTLE







#
# # Basic
# import matplotlib.pyplot as plt
# import numpy as np
#
# #Visualization
# import pandas as pd
# import seaborn as sns
#
# #Preference of Jupyter
# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"
# import warnings
# warnings.filterwarnings("ignore")
#
# #Misc
# import os
# import random
# import researchpy as rp  # For auto-statistics/EDA of dataframe
# from tqdm.notebook import tqdm  # For process display
# from functools import reduce
#
#
# #MACHINE LEARNING
# # Preprocessing
# from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
#
# # Feature selection
# from sklearn.feature_selection import (
#     SelectPercentile,
#     SelectKBest,
#     SelectFwe,
#     SelectFpr,
#     SelectFromModel,
#     chi2,
#     mutual_info_classif,
#     f_classif,
#     SequentialFeatureSelector,
#     RFECV,
# )
# from sklearn.inspection import permutation_importance
#
# # Imbalance Learn
# from imblearn.over_sampling import RandomOverSampler
#
# # Models
# from sklearn.linear_model import LogisticRegression, Lasso, LassoLarsCV, LassoCV
# from sklearn.naive_bayes import GaussianNB
# from sklearn.tree import DecisionTreeClassifier, export_graphviz
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
#
# # Ensemble
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
#
# # Model Metrics, Parameters, Evaluation
# from sklearn.model_selection import (
#     StratifiedKFold,
#     GridSearchCV,
#     cross_val_score,
#     train_test_split,
# )
#
#
# from sklearn.metrics import (
#     confusion_matrix,
#     accuracy_score,
#     roc_curve,
#     classification_report,
#     f1_score,
#     precision_score,
#     recall_score,
#     roc_auc_score,
# )
#
# # Pipeline
# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
#
#
#
#
# import sklearn.neighbors._base
# import sys
# sys.modules["sklearn.neighbors.base"] = (
#     sklearn.neighbors._base)  # To conquer the version naming conflict for use of missingpy
# from missingpy import MissForest  # For Imputation using Random Forest
# from sklearn.impute import SimpleImputer, KNNImputer