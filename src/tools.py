__author__ = 'Stan He@Sickkids.ca'
__contact__ = 'stan.he@sickkids.ca'
__date__ = ['2021-10-21', '2021-10-26', '2021-10-29', '2021-11-01',
            '2021-11-08', '2021-11-19', '2021-12-08', '2021-12-14', '2022-01-04',
            '2022-01-12', '2022-01-27']

"""Gadgets for various tasks 
"""
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pingouin as pg  # Perform statistical testing
import plotly.graph_objects as go

from conf import *  # Import Global Variables
from utils import (ApgarTransformer,
                   BirthTransformer,
                   Log1pTransformer,
                   RespiratoryTransformer,
                   DiscretizePSS,
                   ColumnFilter,
                   ImputerStrategizer,
                   CatNaNImputer,
                   NumNaNimputer,
                   CollinearRemover)
# Imbalance dataset
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

# ML Libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    #    roc_curve,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    plot_confusion_matrix,
    precision_recall_fscore_support,
)

from sklearn.metrics import (precision_recall_curve, PrecisionRecallDisplay)
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.ensemble import StackingClassifier, VotingClassifier

from sklearn.inspection import permutation_importance
from sklearn.model_selection import PredefinedSplit, StratifiedKFold, LeaveOneOut
from mlxtend.evaluate import PredefinedHoldoutSplit
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

#TODO: (1) Strategy 1: Manually Selected Feature Set Result with GridSearchCV paramters for each time points - Multiple Model Predictivity + Weighted Feature Importance with the same set of features
#TODO: (2) Strategy 2: Automated Feature Selection for each Specific Estimator throughout - Singular Model Predictivity + Feature Importance
#TODO: (3) 3.9 Visualization of Model Predictivity at multiple time points for f1 variation (Recall & Precision of Asthmatic) for each ML types - based on (2)
#TODO: (4) 3.10 Visualization of a combined heatmap for feature importance at multiple time points table for each model or using a selection of valid models - good performing models
#TODO: -----Ensembling for better model----


# Load more variables to existing xlsx
def load_child_with_more(
        child_data_path=CHILD_DATA_PATH,
        child_breastfeeding_path=CHILD_BREASTFEEDING_PATH,
        child_wheezephenotype_path=CHILD_WHEEZEPHENOTYPE_PATH,
        child_ethnicity_path=CHILD_ETHNICITY_PATH,
        child_spt_path=CHILD_SPT_PATH
):
    """Load CHILD data with add-on features from additional files
    """

    print(
        f"Loading {child_ethnicity_path, child_data_path, child_breastfeeding_path, child_wheezephenotype_path, child_spt_path}, and merging")
    df_child = pd.read_excel(child_data_path)

    df_bf = pd.read_excel(child_breastfeeding_path)
    df_bf.rename(
        columns={
            "SubjectNumber": "Subject_Number",
            "BF_3m_status": "BF_Status_3m",
            "BF_6m_status": "BF_Status_6m",
        },
        inplace=True,
    )

    df_bf = df_bf.replace({"unknown": np.nan})

    df_wp = pd.read_excel(child_wheezephenotype_path, usecols=["SubjectNumber", "GROUP"])

    df_wp.rename(
        columns={"SubjectNumber": "Subject_Number", "GROUP": "Wheeze_Traj_Type"},
        inplace=True,
    )

    df_ethnicity = pd.read_excel(
        child_ethnicity_path,
        usecols=["SubjectNumber", "PRNMH18WQ3_1a", "PRNMH18WQ3_2a"],
    )
    df_ethnicity.columns = ["Subject_Number", "Mother_Caucasian", "Father_Caucasian"]
    df_ethnicity["Child_Ethinicity"] = (
            df_ethnicity.Mother_Caucasian + df_ethnicity.Father_Caucasian
    ).map({2: "Caucasian", 1: "HalfCaucas", 0: "NonCaucas"})
    df_ethnicity = df_ethnicity.replace({8888: np.nan})

    df_spt = pd.read_excel(
        child_spt_path,
        usecols=["SubjectNumber", "atopy1y", "food1y", "inhalant1y", "atopy3y", "food3y", "inhalant3y",
                 "atopy5y", "food5y", "inhalant5y"]
    )

    df_spt.columns = ["Subject_Number", "Child_Atopy_1y", "Child_Food_1y", "Child_Inhalant_1y"
        , "Child_Atopy_3y", "Child_Food_3y", "Child_Inhalant_3y"
        , "Child_Atopy_5y", "Child_Food_5y", "Child_Inhalant_5y"]

    df_child = pd.merge(
        pd.merge(df_child, df_bf, on="Subject_Number", how="left"),
        df_wp,
        on="Subject_Number",
        how="left",
    )

    # Add on child ethinicity

    df_child = pd.merge(df_child, df_ethnicity, on="Subject_Number", how="left")

    # Add on child skin prick test
    df_child = pd.merge(df_child, df_spt, on="Subject_Number", how="left")

    # Write new file
    df_child.to_excel(CHILD_RAW_DIR + "CHILD_with_addon.xlsx", index=False)
    print(f"The dataframe merged with more information is saved to {CHILD_RAW_DIR} as 'CHILD_with_addon.xlsx'")

    return df_child


def grouped_feature_generator(df):
    """
    Automaticly generate subset of variables to facilitate the feature selection, feature imputation, as well as time-point longitudinal ML analysis process.
    :param df Dataframe to be operated on
    :return: two dictionaries containing groupped features, and features at different time points

    Example:
    feature_dict, timepoint_dict, feature_grouped_overview, time_grouped_overview = grouped_feature_generator(df)
    """

    # To facilitate the selections of features as well as the multivariate imputation for grouped features to build models
    feature_grouped_dict = {}

    feature_grouped_mapping = {
        "1_weight": "^Weight_",
        "2_mother_condition": "^Prenatal_",
        "3_first10min": "10min_",
        "4_breastfeeding": "^BF_",
        "5_home": "^Home",
        "6_mental": "^PSS_|^CSED_",
        "7_parental": "Mother|Father|Dad|Mom|Parental",
        "8_smoke": "Smoke",
        "9_wheeze": "Wheeze",
        "10_resp": "Respiratory|RI",
        "11_antibiotic": "Antibiotic",
    }

    for k, v in feature_grouped_mapping.items():
        feature_grouped_dict[k] = set(df.columns[df.columns.str.contains(v)])

    feature_grouped_dict["1_11"] = set()
    for i in feature_grouped_dict.values():
        feature_grouped_dict["1_11"].update(i)

    feature_grouped_dict["12_misc"] = (
            set(df.columns)
            - feature_grouped_dict["1_11"]
            - set(df.columns[df.columns.str.contains("yCLA")])
            - {"y"}
    )

    temp_current = set()
    for i in feature_grouped_dict.values():
        temp_current.update(i)

    feature_grouped_dict["13_remainder"] = set(df.columns) - temp_current

    # To facilitate the incorporation of features at different time points to build models
    feature_timepoint_dict = {}

    feature_timepoint_mapping = {
        "3m": "3m",
        "6m": "_6m",
        "12m": "_12m|_1y",
        "18m": "18m|BF_implied",
        "24m": "24m|2y",
        "36m": "36m|3y",
        "48m": "48m|4y",
        "60m": "60m|5y|Traj_Type",
        "1_9m_2hy": "_1m$|9m|_2hy|_30m"
    }

    for k, v in feature_timepoint_mapping.items():
        feature_timepoint_dict[k] = set(df.columns[df.columns.str.contains(v)])

    feature_timepoint_dict["after_birth"] = set()

    for i in feature_timepoint_dict.values():
        feature_timepoint_dict["after_birth"].update(i)

    feature_timepoint_dict["at_birth"] = (
            set(df.columns) - feature_timepoint_dict["after_birth"] - {"y"}
    )

    target_repo = {
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
    }

    print("------------------------------------------------------")
    print(
        "The available grouped feature can be one of: \n", feature_grouped_dict.keys()
    )
    print("------------------------------------------------------")
    print(
        "The available time-points for features can be one of: \n",
        feature_timepoint_dict.keys(),
    )
    print("------------------------------------------------------")
    print("Note: Target variable can be one of: \n", target_repo)
    print("------------------------------------------------------")

    feature_grouped_overview = pd.DataFrame(
        [feature_grouped_dict.keys(), feature_grouped_dict.values()], index=["Type", "Features"]
    ).T.set_index("Type").drop(index=["1_11"])

    feature_grouped_overview.loc[-1] = str(target_repo)

    feature_grouped_overview.rename(index={-1: "14_target"}, inplace=True)

    time_variable_overview = (
        pd.DataFrame(
            [feature_timepoint_dict.keys(), feature_timepoint_dict.values()], index=["Time_Points", "Features"]
        )
            .T.set_index("Time_Points")
            .drop(index=["1_9m_2hy"])
    )

    return feature_grouped_dict, feature_timepoint_dict, feature_grouped_overview, time_variable_overview


# grouped_feature_generator will be applied
def features_four_timepoint(df):
    _, features_all_timepoint, _, _ = grouped_feature_generator(df)
    feature_fourtime_dict = {}
    feature_fourtime_dict['at_birth_feature'] = features_all_timepoint["at_birth"]
    feature_fourtime_dict['with_6m'] = features_all_timepoint["3m"] | features_all_timepoint["6m"] | {i for i in
                                                                                                      features_all_timepoint[
                                                                                                          "1_9m_2hy"] if
                                                                                                      "_1m" in i}
    feature_fourtime_dict['with_12m'] = features_all_timepoint["12m"] | {i for i in features_all_timepoint["1_9m_2hy"]
                                                                         if "_9m" in i}
    feature_fourtime_dict['with_36m_all'] = (
            features_all_timepoint["18m"]
            | features_all_timepoint["24m"]
            | features_all_timepoint["36m"]
            | {i for i in features_all_timepoint["1_9m_2hy"] if ("_2hy" in i) | ("_30m" in i)}
    )

    feature_fourtime_dict['with_36m_exclude_diagnosis'] = feature_fourtime_dict['with_36m_all'] - {
        "Asthma_Diagnosis_3yCLA", "Triggered_Asthma_3yCLA", "Viral_Asthma_3yCLA"}

    feature_fourtime_dict['all_four_exclude_3yDiagnosis'] = feature_fourtime_dict['at_birth_feature'] | \
                                                            feature_fourtime_dict['with_6m'] | feature_fourtime_dict[
                                                                'with_12m'] | feature_fourtime_dict[
                                                                'with_36m_exclude_diagnosis']
    feature_fourtime_dict['all_four_include_3yDiagnosis'] = feature_fourtime_dict['at_birth_feature'] | \
                                                            feature_fourtime_dict['with_6m'] | feature_fourtime_dict[
                                                                'with_12m'] | feature_fourtime_dict['with_36m_all']

    print(f"The available keys are {feature_fourtime_dict.keys()}")

    return feature_fourtime_dict


# Extract index_number from raw dataset for all further modelling, training, evaluation, and holdout testing
def df_holdout_throughout(
        df_child,
        include_dust=False,
        treat_possible_as_3yCLA={2: 1},
        # treat_possible_as --- Only happen when include_dust is TRUE,
        # Persistent Number will be perserved if 3yCLA possible is treated as 1 for our algorithm
        treat_possible_as_5yCLA={2: np.nan},  # Possible will be dropped for modelling
        holdout_random_state=100,
        ingredient_persist=15,
        ingredient_transient=5,
        ingredient_emerged=5,
        ingredient_no_asthma=110,
):
    # Display the sample number change
    print("The original sample dimension is", df_child.shape)

    df_child_ml = df_child[
        (df_child.Asthma_Diagnosis_3yCLA.notna())
        & (df_child.Asthma_Diagnosis_5yCLA.notna())
        ].copy()

    print(
        "The sample dimension with both 3y and 5y clinical assessment is",
        df_child_ml.shape,
    )

    # Retrieve dust samples
    dust_column_names = df_child_ml.columns[
        df_child_ml.columns.str.contains("Home_.*P_3m")
    ]

    if include_dust:
        # Treat possible diagnosis
        df_child_ml["Asthma_Diagnosis_3yCLA"] = df_child_ml[
            "Asthma_Diagnosis_3yCLA"
        ].replace(treat_possible_as_3yCLA)
        df_child_ml["Asthma_Diagnosis_5yCLA"] = df_child_ml[
            "Asthma_Diagnosis_5yCLA"
        ].replace(treat_possible_as_5yCLA)

        df_child_ml = df_child_ml[
            (df_child_ml.Asthma_Diagnosis_3yCLA.notna())
            & (df_child_ml.Asthma_Diagnosis_5yCLA.notna())
            ].copy()

        print(
            "The sample dimension with both 3y and 5y clinical assessment after possible is treated is",
            df_child_ml.shape,
        )

        df_child_ml = df_child_ml.dropna(axis="index", subset=dust_column_names)
        print("The sample dimension with dust sample data is", df_child_ml.shape)
    else:
        print("The removed dust columns are", dust_column_names)
        df_child_ml = df_child_ml.drop(columns=dust_column_names)
        print("The sample dimension without dust sample data is", df_child_ml.shape)

    # Group division
    no_asthma = df_child_ml[
        (df_child_ml.Asthma_Diagnosis_5yCLA == 0)
        & (df_child_ml.Asthma_Diagnosis_3yCLA == 0)
        ]

    print("The size of no asthma subject group is:", no_asthma.shape[0])

    transient_asthma = df_child_ml[
        (df_child_ml.Asthma_Diagnosis_5yCLA == 0)
        & (df_child_ml.Asthma_Diagnosis_3yCLA == 1)
        ]

    print(
        "Transient asthma is defined as those who were diagnosed as definite asthma at 3y but rediagnosed as no asthma at 5y."
    )
    print("The size of transient asthma subject group is:", transient_asthma.shape[0])

    emerged_asthma = df_child_ml[
        (df_child_ml.Asthma_Diagnosis_5yCLA == 1)
        & (df_child_ml.Asthma_Diagnosis_3yCLA == 0)
        ]

    print(
        "Emerged asthma is defined as those who were diagnosed as no asthma at 3y but rediagnosed as definite asthma at 5y."
    )
    print("The size of emerged asthma subject group is:", emerged_asthma.shape[0])

    persistent_asthma = df_child_ml[
        (df_child_ml.Asthma_Diagnosis_5yCLA == 1)
        & (df_child_ml.Asthma_Diagnosis_3yCLA == 1)
        ]

    print("The size of persistent asthma subject group is:", persistent_asthma.shape[0])

    rng = np.random.RandomState(holdout_random_state)

    permutated_persist_asthma_index = rng.permutation(persistent_asthma.index)
    permutated_transient_asthma_index = rng.permutation(transient_asthma.index)
    permutated_emerged_asthma_index = rng.permutation(emerged_asthma.index)
    permutated_no_asthma_index = rng.permutation(no_asthma.index)

    holdout_persist_subjectnumber = permutated_persist_asthma_index[:ingredient_persist]
    rest_persist_subjectnumber = permutated_persist_asthma_index[ingredient_persist:]
    holdout_emerged_subjectnumber = permutated_emerged_asthma_index[:ingredient_emerged]
    rest_emerged_subjectnumber = permutated_emerged_asthma_index[ingredient_emerged:]

    holdout_noasthma_subjectnumber = permutated_no_asthma_index[:ingredient_no_asthma]
    rest_noasthma_subjectnumber = permutated_no_asthma_index[ingredient_no_asthma:]
    holdout_transient_subjectnumber = permutated_transient_asthma_index[
                                      :ingredient_transient
                                      ]
    rest_transient_subjectnumber = permutated_transient_asthma_index[
                                   ingredient_transient:
                                   ]

    holdout_index = (
            list(holdout_persist_subjectnumber)
            #            + list(holdout_transient_subjectnumber)
            + list(holdout_noasthma_subjectnumber)
            + list(holdout_emerged_subjectnumber)
    )

    rest_index = (
            list(rest_persist_subjectnumber)
            #            + list(rest_transient_subjectnumber)
            + list(rest_noasthma_subjectnumber)
            + list(rest_emerged_subjectnumber)
    )

    print(
        f"The shrunk dataframe to be processed (engineered and imputed): \n Distribution of Asthma: \n{df_child_ml.Asthma_Diagnosis_5yCLA.value_counts()}, with the dimension of {df_child_ml.shape}"
    )

    print(
        f"The train & evalution dataframe to be processed (engineered and imputed): \n Distribution of Asthma: \n{df_child_ml.loc[rest_index, :].Asthma_Diagnosis_5yCLA.value_counts(normalize=False)}, with the dimension of {df_child_ml.loc[rest_index, :].shape}"
    )

    print(
        f"The holdout dataframe to be processed (engineered and transformed): \n Distribution of Asthma: \n{df_child_ml.loc[holdout_index, :].Asthma_Diagnosis_5yCLA.value_counts(normalize=False)}, with the dimension of {df_child_ml.loc[holdout_index, :].shape}"
    )

    return df_child_ml, rest_index, holdout_index

# Auto-tuning for multiple models with manually selected features, print out best params and display confusion matrix results
def ml_tuned_run(df_train_eval,
                 df_holdout,
                 feature_columns_selected,
                 target_name,
                 scalar=MinMaxScaler(),
                 cv_for_tune=StratifiedKFold(n_splits=3, random_state=4, shuffle=True),
                 scoring_for_tune="average_precision",
                 ):
    # List of Model to perform Grid Search
    clf_lr = LogisticRegression(class_weight="balanced")
    clf_dt = DecisionTreeClassifier(class_weight="balanced", random_state=2021)
    clf_svc = SVC(class_weight="balanced", probability=True, random_state=2021)
    clf_rf = RandomForestClassifier(class_weight="balanced", random_state=2021)
    clf_xgb = XGBClassifier(random_state=2021, verbosity=0)

    # Define param grid for hyperparmeter tuning
    param_grid_lr = {
        "solver": ["lbfgs", "liblinear", "saga"], #default=’lbfgs’
        "C": [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1], #  default: 1
    }

    param_grid_dt = {
        "criterion": ["gini", "entropy"], #default=”gini”
        "max_depth": [3, 4, 5, 6, 7, None], #default=None
        "min_samples_split": [2, 4],#default=2
        "max_features": ["sqrt", 0.8, None], #default=None
    }

    param_grid_svc = {
        "kernel": ["linear", "poly", "rbf", "sigmoid"], #default=’rbf’
        "C": [0.02, 0.05, 0.1, 0.2, 0.5, 1], #default=1
    }

    param_grid_rf = {
        "max_depth": [3, 4, 5, 6], #default=None
        "max_features": [4, 5, 6, 7, 8], #default=None
    }

    param_grid_xgb = {
        "learning_rate": [1e-2, 5e-2, 1e-1, 3e-1], #default=0.3
        "max_depth": [3, 4, 5, 6], #default=6
        "colsample_bytree": [0.5, 0.75, 1],#default=1
        "scale_pos_weight": [7, 10, 15], # equivalent to class_weight, default = 1
    }

    # Best Param dict
    gs_param_dict = {}
    gs_param_dict['nb'] = {}
    gs_param_dict['knn'] = {}

    # Print out the current best parameters for evaluation dataset
    # Before fit search, scale the dataset first.
    X_train_eval = df_train_eval[feature_columns_selected]
    y_train_eval = df_train_eval[target_name]
    X_test = df_holdout[feature_columns_selected]
    y_test = df_holdout[target_name]

    scalar.fit(X_train_eval)

    X_train_eval = pd.DataFrame(
        scalar.transform(X_train_eval), columns=X_train_eval.columns, index=X_train_eval.index
    )

    X_test = pd.DataFrame(
        scalar.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    ###############################Logistic Regression########################################
    gs_lr = GridSearchCV(clf_lr, param_grid_lr, cv=cv_for_tune, scoring=scoring_for_tune)
    print(f"Search for the best parameters for lr in {param_grid_lr}....")
    gs_lr.fit(
        X_train_eval, y_train_eval
    )
    gs_param_dict['lr']=gs_lr.best_params_
    print(
        f"The best parameters for Logistic Regression are: {gs_lr.best_params_} with the score of {gs_lr.best_score_}")

    ###############################Decision Tree########################################
    gs_dt = GridSearchCV(clf_dt, param_grid_dt, cv=cv_for_tune, scoring=scoring_for_tune)
    print(f"Search for the best parameters for dt in {param_grid_dt}....")
    gs_dt.fit(
        X_train_eval, y_train_eval
    )
    gs_param_dict['dt'] = gs_dt.best_params_
    print(f"The best parameters for Decision Tree are: {gs_dt.best_params_} with the score of {gs_dt.best_score_}")

    ###############################Support Vector Machine########################################
    gs_svc = GridSearchCV(clf_svc, param_grid_svc, cv=cv_for_tune, scoring=scoring_for_tune)
    print(f"Search for the best parameters for svc  in {param_grid_svc}....")
    gs_svc.fit(
        X_train_eval, y_train_eval
    )
    gs_param_dict['svc'] = gs_svc.best_params_
    print(
        f"The best parameters for Support Vector Machine are:{gs_svc.best_params_} with the score of {gs_svc.best_score_}")

    ###############################Random Forest########################################
    gs_rf = GridSearchCV(clf_rf, param_grid_rf, cv=cv_for_tune, scoring=scoring_for_tune)
    print(f"Search for the best parameters for rf in {param_grid_rf}....")
    gs_rf.fit(
        X_train_eval, y_train_eval
    )
    gs_param_dict['rf'] = gs_rf.best_params_
    print(f"The best parameters for Random Forest are: {gs_rf.best_params_} with the score of {gs_rf.best_score_}")

    ###############################XGBoost########################################
    gs_xgb = GridSearchCV(clf_xgb, param_grid_xgb, cv=cv_for_tune, scoring=scoring_for_tune)
    print(f"Search for the best parameters for xgb in {param_grid_xgb}....")
    gs_xgb.fit(
        X_train_eval, y_train_eval
    )
    gs_param_dict['xgb'] = gs_xgb.best_params_
    print(f"The best parameters for XGBoost are: {gs_xgb.best_params_} with the score of {gs_xgb.best_score_}")


    # Quick Visualize Result with tuned hyperparameters
    # (1) Logistic Regression
    lr_cv_performance = model_result_holdout(
        df_train_eval,
        df_holdout,
        feature_columns_selected,
        target_name,
        estimator=LogisticRegression(class_weight="balanced", **gs_param_dict['lr']),
        scalar=MinMaxScaler(),
    )
    ConfusionMatrixDisplay.from_predictions(
        lr_cv_performance[0]["y_true_holdout"],
        lr_cv_performance[0]["y_predicted_holdout_altered_threshold"],
    )

    print(
        classification_report(
            lr_cv_performance[0]["y_true_holdout"],
            lr_cv_performance[0]["y_predicted_holdout_altered_threshold"],
        )
    )

    lr_imp_features = pd.DataFrame(
        data=lr_cv_performance[1].coef_.reshape(1, -1)[0],
        index=list(feature_columns_selected),
        columns=["Logistic Regression"],
    )
    lr_imp_features.sort_values("Logistic Regression", ascending=False, inplace=True)
    plt.figure(figsize=(12, 8), dpi=200)
    sns.barplot(
        data=lr_imp_features, y=lr_imp_features.index, x=lr_imp_features["Logistic Regression"],
    )

    # (2) Decision Tree
    dt_cv_performance = model_result_holdout(
        df_train_eval,
        df_holdout,
        feature_columns_selected,
        target_name,
        estimator=DecisionTreeClassifier(class_weight="balanced", random_state=2021, **gs_param_dict['dt']),
        scalar=MinMaxScaler(),
    )

    ConfusionMatrixDisplay.from_predictions(
        dt_cv_performance[0]["y_true_holdout"],
        dt_cv_performance[0]["y_predicted_holdout_altered_threshold"],
    )

    print(
        classification_report(
            dt_cv_performance[0]["y_true_holdout"],
            dt_cv_performance[0]["y_predicted_holdout_altered_threshold"],
        )
    )

    dt_imp_features = pd.DataFrame(
        data=dt_cv_performance[1].feature_importances_.reshape(1, -1)[0],
        index=list(feature_columns_selected),
        columns=["Decision Tree"],
    )
    dt_imp_features.sort_values("Decision Tree", ascending=False, inplace=True)
    plt.figure(figsize=(12, 8), dpi=200)
    sns.barplot(
        data=dt_imp_features, y=dt_imp_features.index, x=dt_imp_features["Decision Tree"],
    )

    # (3) Support Vector Machine
    svc_cv_performance = model_result_holdout(
        df_train_eval,
        df_holdout,
        feature_columns_selected,
        target_name,
        estimator=SVC(
            class_weight="balanced",
            probability=True,
            random_state=2021,
            **gs_param_dict['svc']
        ),
        scalar=MinMaxScaler(),
    )
    ConfusionMatrixDisplay.from_predictions(
        svc_cv_performance[0]["y_true_holdout"],
        svc_cv_performance[0]["y_predicted_holdout_altered_threshold"],
    )

    print(
        classification_report(
            svc_cv_performance[0]["y_true_holdout"],
            svc_cv_performance[0]["y_predicted_holdout_altered_threshold"],
        )
    )

    permutation_result = permutation_importance(
        svc_cv_performance[1],
        df_train_eval[feature_columns_selected],
        df_train_eval[target_name],
        n_repeats=12,
        random_state=2021,
        scoring="average_precision",
    )

    # Visualization of feature importance for permutation importance
    perm_sorted_idx = permutation_result.importances_mean.argsort()
    plt.figure(figsize=(20, 10))
    plt.title("Feature Importance for {}".format("SVC"))
    plt.barh(
        width=permutation_result.importances_mean[perm_sorted_idx].T,
        y=df_train_eval[feature_columns_selected].columns[perm_sorted_idx],
    )

    # (4) Random Forest
    rf_cv_performance = model_result_holdout(
        df_train_eval,
        df_holdout,
        feature_columns_selected,
        target_name,
        estimator=RandomForestClassifier(
            class_weight="balanced",
            random_state=2021,
            **gs_param_dict['rf']
        ),
        scalar=MinMaxScaler(),
    )
    ConfusionMatrixDisplay.from_predictions(
        rf_cv_performance[0]["y_true_holdout"],
        rf_cv_performance[0]["y_predicted_holdout_altered_threshold"],
    )

    print(
        classification_report(
            rf_cv_performance[0]["y_true_holdout"],
            rf_cv_performance[0]["y_predicted_holdout_altered_threshold"],
        )
    )

    rf_imp_features = pd.DataFrame(
        data=rf_cv_performance[1].feature_importances_.reshape(1, -1)[0],
        index=list(feature_columns_selected),
        columns=["Random Forest"],
    )
    rf_imp_features.sort_values("Random Forest", ascending=False, inplace=True)
    plt.figure(figsize=(12, 8), dpi=200)
    sns.barplot(
        data=rf_imp_features, y=rf_imp_features.index, x=rf_imp_features["Random Forest"],
    )

    # (5) XGBoost
    xgb_cv_performance = model_result_holdout(
        df_train_eval,
        df_holdout,
        feature_columns_selected,
        target_name,
        estimator=XGBClassifier(
            random_state=2021,
            verbosity=0,
            **gs_param_dict['xgb']
        ),
        scalar=MinMaxScaler(),
    )
    ConfusionMatrixDisplay.from_predictions(
        xgb_cv_performance[0]["y_true_holdout"],
        xgb_cv_performance[0]["y_predicted_holdout_altered_threshold"],
    )

    print(
        classification_report(
            xgb_cv_performance[0]["y_true_holdout"],
            xgb_cv_performance[0]["y_predicted_holdout_altered_threshold"],
        )
    )

    xgb_imp_features = pd.DataFrame(
        data=xgb_cv_performance[1].feature_importances_.reshape(1, -1)[0],
        index=list(feature_columns_selected),
        columns=["XGBoost"],
    )
    xgb_imp_features.sort_values("XGBoost", ascending=False, inplace=True)
    plt.figure(figsize=(12, 8), dpi=200)
    sns.barplot(
        data=xgb_imp_features, y=xgb_imp_features.index, x=xgb_imp_features["XGBoost"],
    )

    return gs_param_dict



# Automatic Feature Selection At Multiple time-points  with specific ML model
def ml_autofs_multiplepoints(
        df_train_eval,
        df_holdout,
        four_time_dict,
        scalar=MinMaxScaler(),
        estimator=LogisticRegression(class_weight="balanced"),
        estimator_name='lr',
        cv=StratifiedKFold(n_splits=3, random_state=3, shuffle=True),
        priori_k=25,
        precision_inspection_range=0.005,
        fixed_features=None,
        scoring="average_precision",
):
    # Keep feature selection results in multiple time points
    res = {}
    res["at_birth"] = ml_feature_selection(
        X=df_train_eval[four_time_dict["at_birth_feature"]],
        y=df_train_eval["Asthma_Diagnosis_5yCLA"],
        scalar=scalar,
        cv=cv,
        priori_k=priori_k,
        scoring=scoring,
        is_floating=True,
        fixed_features=fixed_features,
        precision_inspection_range=precision_inspection_range,
        test_model_number=0,
        clf=(estimator, estimator_name)
    )

    res["before_6m"] = ml_feature_selection(
        X=df_train_eval[set(res["at_birth"][1].index) | four_time_dict["with_6m"]],
        y=df_train_eval["Asthma_Diagnosis_5yCLA"],
        scalar=scalar,
        cv=cv,
        priori_k=priori_k,
        scoring=scoring,
        is_floating=True,
        fixed_features=fixed_features,
        precision_inspection_range=precision_inspection_range,
        test_model_number=0,
        clf=(estimator, estimator_name)
    )
    res["before_12m"] = ml_feature_selection(
        X=df_train_eval[set(res["before_6m"][1].index) | four_time_dict["with_12m"]],
        y=df_train_eval["Asthma_Diagnosis_5yCLA"],
        scalar=scalar,
        cv=cv,
        priori_k=priori_k,
        scoring=scoring,
        is_floating=True,
        fixed_features=fixed_features,
        precision_inspection_range=precision_inspection_range,
        test_model_number=0,
        clf=(estimator, estimator_name)
    )
    res["before_36m"] = ml_feature_selection(
        X=df_train_eval[
            set(res["before_12m"][1].index)
            | four_time_dict["with_36m_exclude_diagnosis"]
        ],
        y=df_train_eval["Asthma_Diagnosis_5yCLA"],
        scalar=scalar,
        cv=cv,
        priori_k=priori_k,
        scoring=scoring,
        is_floating=True,
        fixed_features=fixed_features,
        precision_inspection_range=precision_inspection_range,
        test_model_number=0,
        clf=(estimator, estimator_name)
    )

    # List feature selection at different time points
    res_df = pd.concat(
        [
            res["at_birth"][0],
            res["before_6m"][0],
            res["before_12m"][0],
            res["before_36m"][0],
        ]
    )
    res_df.index = res.keys()

    # View confusion matrix and keep model performance on holdout dataset

    holdout_res = {} # Contains prediction and estimators
    feature_res = {} # Contains feature importance for each model

    for time_points in list(res.keys()):
        holdout_res[time_points] = model_result_holdout(
            df_train_eval,
            df_holdout,
            feature_columns_selected=list(res[time_points][1].index),
            target_name="Asthma_Diagnosis_5yCLA",
            estimator=estimator,
            scalar=scalar,
        )
        ConfusionMatrixDisplay.from_predictions(
            holdout_res[time_points][0]["y_true_holdout"],
            holdout_res[time_points][0]["y_predicted_holdout_altered_threshold"],
        )

        print(
            classification_report(
                holdout_res[time_points][0]["y_true_holdout"],
                holdout_res[time_points][0]["y_predicted_holdout_altered_threshold"],
            )
        )

        if estimator_name in ['lr']:  # Coefficient for LogisticRegression

            feature_res[time_points] = pd.DataFrame(
                data=holdout_res[time_points][1].coef_.reshape(1, -1)[0],
                index=list(res[time_points][1].index),
                columns=[estimator_name],
            )


        elif estimator_name in ['dt', 'rf', 'xgb']:  # Feature Importance
            feature_res[time_points] = pd.DataFrame(
                data=holdout_res[time_points][1].feature_importances_.reshape(1, -1)[0],
                index=list(res[time_points][1].index),
                columns=[estimator_name],
            )

        else:  # Permutation importance

            permutation_result = permutation_importance(
                holdout_res[time_points][1],
                df_train_eval[list(res[time_points][1].index)],
                df_train_eval["Asthma_Diagnosis_5yCLA"],
                n_repeats=15,
                random_state=2021,
                scoring="average_precision",
            )

            feature_res[time_points] = pd.DataFrame(
                data=permutation_result["importances_mean"].reshape(1, -1)[0],
                index=list(res[time_points][1].index),
                columns=[estimator_name],
            )
        feature_res[time_points].sort_values(estimator_name, ascending=False, inplace=True)
        plt.figure(figsize=(12, 8), dpi=200)
        sns.barplot(
            data=feature_res[time_points], y=feature_res[time_points].index, x=feature_res[time_points][estimator_name],
        )

    return res_df, holdout_res, feature_res


# Calculate and visualize the feature importance change and model predictability over 4 different time points
def ml_res_visualization(
    df_train_eval,
    df_holdout,
    four_time_dict,
    scalar=MinMaxScaler(),
    cv=StratifiedKFold(n_splits=3, random_state=3, shuffle=True),
    priori_k=25,
    precision_inspection_range=0.005,
    fixed_features=None,
    scoring="average_precision",
):
    # Define the model parameter here for auto feature selection

    # Logistic Regression
    lr = LogisticRegression(C=0.02, solver="lbfgs", class_weight="balanced")

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        max_depth=3,
        max_features=5,
        random_state=2021,
    )

    # XGB
    xgb = XGBClassifier(
        max_depth=3,
        learning_rate=0.01,
        colsample_bytree=0.8,
        scale_pos_weight=15,
        subsample=0.8,
        random_state=2021,
    )
    # SVC
    svc = SVC(
        C=0.02,
        kernel="linear",
        class_weight="balanced",
        probability=True,
        random_state=2021,
    )
    # Decision Tree
    dt = DecisionTreeClassifier(
        criterion="gini",
        max_depth=6,  # Previous is None
        class_weight="balanced",
        random_state=2021,
    )

    ml_name_dict = {
        "lr": [lr, "Logistic Regression"],
        "rf": [rf, "Random Forest"],
        "xgb": [xgb, "eXtreme Gradient Boost"],
        "svc": [svc, "Support Vector Machine"],
        "dt": [dt, "Decision Tree"],
    }

    ml_res_dict = {}
    for ml_name in ml_name_dict.keys():
        ml_res_dict[ml_name] = ml_autofs_multiplepoints(
            df_train_eval,
            df_holdout,
            four_time_dict,
            scalar=scalar,
            estimator=ml_name_dict[ml_name][0],
            estimator_name=ml_name,
            cv=cv,
            priori_k=priori_k,
            precision_inspection_range=precision_inspection_range,
            fixed_features=fixed_features,
            scoring=scoring,
        )

    # Visualize the model performance at each time point
    model_performance_dict = defaultdict(list)
    model_name_full = [i[1] for i in list(ml_name_dict.values())]
    model_res = list(ml_res_dict.values())
    time_points = ["at_birth", "before_6m", "before_12m", "before_36m"]
    model_metrics = [
        "Precision",
        "Recall",
        "F1",
        "Average_precision",
        "Roc_auc",
        "Support",
    ]
    model_multiindex = []
    for model_name, content in zip(model_name_full, model_res):
        for timepoint in time_points:
            model_performance_dict[model_name + "-" + timepoint] = [
                content[1][timepoint][0]["precision_recall_f1_support"][:, 1][0],
                content[1][timepoint][0]["precision_recall_f1_support"][:, 1][1],
                content[1][timepoint][0]["precision_recall_f1_support"][:, 1][2],
                content[1][timepoint][0]["average_precision_score"],
                content[1][timepoint][0]["roc_auc_score"],
                content[1][timepoint][0]["precision_recall_f1_support"][:, 1][3],
            ]
            model_multiindex.append((model_name, timepoint))

    df_model_performance = pd.DataFrame(model_performance_dict, index=model_metrics).T
    df_model_performance.index = pd.MultiIndex.from_tuples(
        model_multiindex, names=["Model", "Time Point"]
    )

    df_model_performance.drop(columns=["Support"])

    model_perform_df = df_model_performance.drop(columns=["Support"])

    # Create feature importance dictionary for each model used
    feature_importance_dict = {}

    for i in ml_res_dict.keys():
        feature_importance_dict[i] = pd.DataFrame()
        for t in time_points:
            ml_res_dict[i][2][t].columns = [t]
            feature_importance_dict[i] = pd.concat(
                [feature_importance_dict[i], ml_res_dict[i][2][t]], axis=1
            )

        # Create visualization
        plt.figure(figsize=(9, 14))
        plt.title(ml_name_dict[i][1])

        # Set vmin, vmax, for seaborn heatmap color bar
        if i in ["lr"]:
            vmin = -1
            vmax = 1
        elif i in ["dt", "xgb", "rf"]:
            vmin = 0
            vmax = 0.4
        else:  # Permutation importance
            vmin = -0.02
            vmax = 0.1

        # Plotting
        g = sns.heatmap(
            feature_importance_dict[i],
            cmap="vlag",
            center=0,
            vmax=vmax,
            vmin=vmin,
            linewidths=0.05,
            annot=True,
            linecolor="lightgrey",
            cbar_kws={"shrink": 0.45},
        )
        g.xaxis.set_ticks_position("top")

    # Visualize the model performance in colored frame
    model_perform_df.style.background_gradient(cmap="Greens")

    return ml_res_dict, feature_importance_dict, model_perform_df

def filter_features(x, threshold=0.015):
    if x <= threshold:
        x = np.nan
    return x


def feature_progression_merge(
    ml_res_final,
    ml_list=["lr", "rf", "xgb", "svc", "dt"],
    coef_thresh=0.15,
    featimp_thresh=0.015,
    permutation_thresh=0.001,
    how="sum",
    normalize=True,
):
    """
    params: how, string
    Available options include: 'sum','avg'
    """
    feature_filtered = {}
    # 1. Filtering
    for i in ml_list:
        if i in ["lr"]:
            feature_filtered[i] = ml_res_final[1][i].applymap(
                lambda x: filter_features(abs(x), threshold=coef_thresh)
            )
        elif i in ["rf", "xgb", "dt"]:
            feature_filtered[i] = ml_res_final[1][i].applymap(
                lambda x: filter_features(x, threshold=featimp_thresh)
            )
        else:
            feature_filtered[i] = ml_res_final[1][i].applymap(
                lambda x: filter_features(x, threshold=permutation_thresh)
            )

    # 2. Normalization & Filtering out those without values
    for i in ml_list:
        for col in feature_filtered[i].columns:
            feature_filtered[i][col] = feature_filtered[i][col] / max(
                feature_filtered[i][col].fillna(0)
            )
        feature_filtered[i].dropna(how="all", inplace=True)

    # 3. Combination
    feature_df_merged = pd.DataFrame()
    for i in ml_list:
        feature_df_merged = pd.concat([feature_df_merged, feature_filtered[i]])

    if how == "sum":
        feature_df_merged = (
            feature_df_merged.reset_index().groupby("index").sum().replace({0: np.nan})
        )
    elif how == "avg":
        feature_df_merged = (
            feature_df_merged.reset_index().groupby("index").mean().replace({0: np.nan})
        )
    else:
        print("Wrong paramters for 'how'")

    feature_df_merged.sort_values(
        by=list(feature_df_merged.columns),
        ascending=[False, False, False, False],
        inplace=True,
    )

    if normalize:
        for col in feature_df_merged.columns:
            feature_df_merged[col] = feature_df_merged[col] / max(
                feature_df_merged[col].fillna(0)
            )

    # For Visualization Only
    ml_final_features = feature_df_merged.copy()

    # Rename Feature Names
    ml_final_features.index = [i.replace("_", " ") for i in ml_final_features.index]
    ml_final_features.columns = [
        i.replace("_", " ").capitalize() for i in ml_final_features.columns
    ]
    ml_final_features.index.set_names("Features", inplace=True)
    ml_final_features.rename_axis("Time Point", axis="columns", inplace=True)

    # Set Figure Size
    plt.figure(figsize=(12, 32))
    g = sns.heatmap(
        ml_final_features,
        cmap="vlag",
        center=0,
        vmax=1,
        vmin=0,
        linewidths=0.05,
        annot=True,
        linecolor="lightgrey",
        cbar_kws={"shrink": 0.45},
    )
    g.xaxis.set_ticks_position("top")
    g.xaxis.set_label_position("top")

    plt.savefig("../output/Feature_Importance_Final.pdf", dpi=150)

    return feature_df_merged

# Calculate and visualize the ensemble model performance at different time points with input of merged feature dataframe
def ml_ensemble_res(df_train_eval, df_holdout, ml_merged_features, scalar=MinMaxScaler(), threshold_for_selection=0.1):
    """
    Use the selected feature at different time points from merged feature table to create multiple ensemble models.
    :param df_train_eval
    :param df_holdout
    :param ml_merged_features: DataFrame, result of the function "feature_progression_merge()"
    :param scalar: to process the train, eval, holdout dataset
    :param threshold_for_selection: float between 0 to 1, used to select features of importance at different time point
    :return: Dataframe to overview the ensemble performance with visualization of ensemble performance
    """
    # Features collection extracted using merged feature table at different time points
    feature_dict = {}
    for i in ml_merged_features.columns: # Columns are time points
        feature_dict[i] = list(ml_merged_features[i][ml_merged_features[i] > threshold_for_selection].index)

    # Set parameters of the individual estimators
    ensemble_collections = [
        ("lr", LogisticRegression(C=0.02, solver="lbfgs", class_weight="balanced")),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                max_depth=3,
                max_features=5,
                random_state=2021,
            ),
        ),
        (
            "xgb",
            XGBClassifier(
                max_depth=3,
                learning_rate=0.01,
                colsample_bytree=0.8,
                scale_pos_weight=15,
                subsample=0.8,
                random_state=2021,
            ),
        ),
        (
            "svc",
            SVC(
                C=0.02,
                kernel="linear",
                class_weight="balanced",
                probability=True,
                random_state=2021,
            ),
        ),
        (
            "dt",
            DecisionTreeClassifier(
                criterion="gini",
                max_depth=6,  # Previous is None
                class_weight="balanced",
                random_state=2021,
            ),
        ),
    ]
    ensemble_weights = [0.25, 0.35, 0.2, 0.15, 0.05]

    # Define a series of ensemble algorithms
    # Voting Hard, Soft, Weighted, Uniformed

    soft_vote = VotingClassifier(estimators=ensemble_collections, voting="soft")
    hard_vote = VotingClassifier(estimators=ensemble_collections, voting="hard")
    weighted_soft = VotingClassifier(
        estimators=ensemble_collections, weights=ensemble_weights, voting="soft"
    )
    weighted_hard = VotingClassifier(
    estimators=ensemble_collections, weights=ensemble_weights, voting="hard")


    # Stacking LR, RF, XGB, SVC, DT
    stacking_xgb = StackingClassifier(
    estimators=ensemble_collections, final_estimator=XGBClassifier(random_state=2021)
)
    stacking_svc = StackingClassifier(estimators=ensemble_collections, final_estimator=SVC(random_state=2021, probability=True))
    stacking_rf = StackingClassifier(
    estimators=ensemble_collections, final_estimator=RandomForestClassifier(random_state=2021)
)
    stacking_lr = StackingClassifier(
        estimators=ensemble_collections, final_estimator=LogisticRegression(random_state=2021)
    )
    stacking_dt = StackingClassifier(
        estimators=ensemble_collections, final_estimator=DecisionTreeClassifier(random_state=2021)
    )

    ensemble_algorithms = { "hard_vote":hard_vote, "soft_vote":soft_vote,
                            'weighted_hard':weighted_hard, 'weighted_soft':weighted_soft,
                            'stacking_lr':stacking_lr,'stacking_rf':stacking_rf,
                            'stacking_xgb':stacking_xgb,'stacking_svc':stacking_svc,
                            'stacking_dt':stacking_dt}
    ensemble_res_dict = {}

    for ens_name,ens_clf in ensemble_algorithms.items():
        ensemble_res = {}
        if "hard" in ens_name:
            for timepoint in feature_dict.keys():
                ensemble_res[timepoint] = model_result_holdout(
                    df_train_eval,
                    df_holdout,
                    feature_columns_selected=feature_dict[timepoint],
                    target_name="Asthma_Diagnosis_5yCLA",
                    estimator=ens_clf,
                    scalar=scalar,
                    voting="hard",
                )
            ensemble_res_dict[ens_name] = ensemble_res
        else:
            for timepoint in feature_dict.keys():
                ensemble_res[timepoint] = model_result_holdout(
                    df_train_eval,
                    df_holdout,
                    feature_columns_selected=feature_dict[timepoint],
                    target_name="Asthma_Diagnosis_5yCLA",
                    estimator=ens_clf,
                    scalar=scalar,
                    voting=None,
                )
            ensemble_res_dict[ens_name] = ensemble_res

    ensemble_res_perf = {}
    model_multiindex = []
    model_metrics = [
        "Precision",
        "Recall",
        "F1",
        "Average_precision",
        "Roc_auc",
        "Support",
    ]
    for ensemble_name in ensemble_res_dict.keys():
        for timepoint in ensemble_res_dict[ensemble_name].keys():
            if "hard" in ensemble_name:
                ensemble_res_perf[ensemble_name + "-" + timepoint] = [
                    ensemble_res_dict[ensemble_name][timepoint][0][
                        "precision_recall_f1_support"
                    ][:, 1][0],
                    ensemble_res_dict[ensemble_name][timepoint][0][
                        "precision_recall_f1_support"
                    ][:, 1][1],
                    ensemble_res_dict[ensemble_name][timepoint][0][
                        "precision_recall_f1_support"
                    ][:, 1][2],
                    np.nan,
                    np.nan,
                    ensemble_res_dict[ensemble_name][timepoint][0][
                        "precision_recall_f1_support"
                    ][:, 1][3],
                ]

            else:
                ensemble_res_perf[ensemble_name + "-" + timepoint] = [
                    ensemble_res_dict[ensemble_name][timepoint][0][
                        "precision_recall_f1_support"
                    ][:, 1][0],
                    ensemble_res_dict[ensemble_name][timepoint][0][
                        "precision_recall_f1_support"
                    ][:, 1][1],
                    ensemble_res_dict[ensemble_name][timepoint][0][
                        "precision_recall_f1_support"
                    ][:, 1][2],
                    ensemble_res_dict[ensemble_name][timepoint][0][
                        "average_precision_score"
                    ],
                    ensemble_res_dict[ensemble_name][timepoint][0]["roc_auc_score"],
                    ensemble_res_dict[ensemble_name][timepoint][0][
                        "precision_recall_f1_support"
                    ][:, 1][3],
                ]

            model_multiindex.append(
                (
                    ensemble_name.replace("_", " ").capitalize(),
                    timepoint.replace("_", " ").title(),
                )
            )

    ensemble_model_performance = pd.DataFrame(ensemble_res_perf, index=model_metrics).T
    ensemble_model_performance.index = pd.MultiIndex.from_tuples(
        model_multiindex, names=["Model", "Time Point"]
    )

    return ensemble_model_performance


# Visualize the individual model performance & feature importance table at different time points
def ml_individual_performance(df_train_eval,
        df_holdout,
        ml_merged_features,
        threshold_for_selection=0.1,
        target_name="Asthma_Diagnosis_5yCLA",
        scalar=MinMaxScaler(),
        scoring_func=average_precision_score,
        importance_scoring="average_precision"):

    # Features collection extracted using merged feature table at different time points
    feature_dict = {}
    for i in ml_merged_features.columns: # Columns are time points
        feature_dict[i] = list(ml_merged_features[i][ml_merged_features[i] > threshold_for_selection].index)

    timepoint_res_dict = {}

    # Obtain the model performance
    for timepoint in feature_dict.keys():
        timepoint_res_dict[timepoint] = df_ml_run(
            df_train_eval,
            df_holdout,
            feature_columns=feature_dict[timepoint],
            target_name=target_name,
            scalar=scalar,
            scoring_func=scoring_func,
            importance_scoring=importance_scoring,
        )

    # Visualize the performance and feature importance
    for timepoint in feature_dict.keys():
        print("Model performance and Feature Importance", timepoint.upper())
        timepoint_res_dict[timepoint][0].style.background_gradient(cmap="Greens")
        timepoint_res_dict[timepoint][2].sort_values(
            by="Weighted_Importance", ascending=False, inplace=True
        )
        timepoint_res_dict[timepoint][2].style.background_gradient(cmap="Greens")

    return timepoint_res_dict

# View the highest performed features for various machine learning models
def ml_feature_selection(
        X,
        y,
        scalar=MinMaxScaler(),
        cv=StratifiedKFold(n_splits=3, random_state=3, shuffle=True),
        priori_k=25,
        scoring="average_precision",
        is_floating=True,
        fixed_features=None,
        precision_inspection_range=0.02,
        test_model_number=None, #  0 represents only specific ML model will be used, None represent all model in function
        clf=(None,"model_name"), # The first element is classifier, the second element is the name to be referred
):
    """
    Generate feature subset performance dataframe for minimal feature selection.
    ***Note: For Cross-validation, the index should be reset to default (0 to sample size-1) before cv can be put to use.

    :param X: feature matrix of train, evaluation with target variable dropped
    :param y: target label
    :param cv: PredefinedHoldoutSplit using test_index, and StratifiedKFold for multiple validation for feature selection
    :param priori_k: the number of feature to include for feature subset selection
    :param scoring: what metrics to use for feature selection
    :param is_floating: allow retrospective inspection for each added feature
    :param fixed_features: any feature to include persistently based on interest
    :param precision_inspection_range: allow minor difference of performance to check the minimal subset of features
    :param test_model_number: number of estimators to try (quicker stop)
    :return: birth_fs_df, birth_features_frequency, res_dict
    """

    # Untuned Estimator to obtain features with maximal performance
    svc = SVC(class_weight="balanced")
    knn = KNeighborsClassifier(n_neighbors=5)
    lr = LogisticRegression(class_weight="balanced")
    rf = RandomForestClassifier(class_weight="balanced", random_state=2021)
    xgb = XGBClassifier(class_weight="balanced")
    dt = DecisionTreeClassifier(
        criterion="entropy", random_state=0, max_depth=6, class_weight="balanced"
    )

    # Scale of X
    X = pd.DataFrame(scalar.fit_transform(X), columns=X.columns, index=X.index)

    # Records all the result
    res_dict = {}
    if test_model_number == 0:
        print(
            "The specific classifier where feature is being automatically selected is:", clf[1]
        )
        sfs = SFS(
            clf[0],
            k_features=priori_k,
            forward=True,
            floating=is_floating,
            verbose=2,
            scoring=scoring,
            cv=cv,
            fixed_features=fixed_features,
        )

        sfs = sfs.fit(X, y)

        res_dict[clf[1]] = pd.DataFrame.from_dict(
            sfs.get_metric_dict()
        ).T.sort_values(by="avg_score", ascending=False)

    else:
        for model, model_name in zip(
                [lr, rf, dt, knn, xgb, svc][:test_model_number], ["lr", "rf", "dt", "knn", "xgb", "svc"][:test_model_number]
        ):
            print(
                "Current classifier where feature selection is being tested is:", model_name
            )
            sfs = SFS(
                model,
                k_features=priori_k,
                forward=True,
                floating=is_floating,
                verbose=2,
                scoring=scoring,
                cv=cv,
                fixed_features=fixed_features,
            )

            sfs = sfs.fit(X, y)

            res_dict[model_name] = pd.DataFrame.from_dict(
                sfs.get_metric_dict()
            ).T.sort_values(by="avg_score", ascending=False)

    # Extract only the best performed subset
    birth_fs_df = pd.DataFrame()
    for i, j in zip(res_dict.values(), res_dict.keys()):
        # Add more features to model feature inspection dataframe
        i["clf"] = j
        i["number_of_features"] = i["feature_idx"].apply(len)

        rank = 1
        rank_list = []
        highest_number = i["avg_score"].iloc[0]

        # Calculate the ranking according to precision inspection range
        for score in i["avg_score"]:
            if score >= highest_number - precision_inspection_range:
                rank_list.append(rank)
            else:
                highest_number = score
                rank = rank + 1
                rank_list.append(rank)

        # Retrieve the minimal feature of the precision rank - top rank is 1
        i["feature_precision_rank"] = rank_list

        # Rank 1-N: "1" represent highest precision, sorted by ascending order
        # Number of features: sorted by ascending order as we are more interested in the minimal number of subset
        # Avg_score: sorted by descending order as we would like to keep the highest number in the top
        i.sort_values(
            by=["feature_precision_rank", "number_of_features", "avg_score"],
            ascending=[True, True, False],
            inplace=True,
        )

        # Add top row of features performance for each estimator to create a new dataframe
        birth_fs_df = birth_fs_df.append(i[:1], ignore_index=True)

    # After birth fs is created, sort first by 'avg_score'
    birth_fs_df = birth_fs_df.sort_values(by="avg_score", ascending=False).reset_index(
        drop=True
    )

    # Recreate the feature precision rank for the entire selected cohorts of models
    rank = 1
    rank_list = []
    highest_number = birth_fs_df["avg_score"].iloc[0]

    for score in birth_fs_df["avg_score"]:
        if score >= highest_number - precision_inspection_range:
            rank_list.append(rank)
        else:
            highest_number = score
            rank = rank + 1
            rank_list.append(rank)

    # Retrieve the minimal feature of the precision rank - top rank is 1
    birth_fs_df["feature_precision_rank"] = rank_list

    # The minimal number of features for rank 1
    birth_fs_df.sort_values(
        by=["feature_precision_rank", "number_of_features", "avg_score"],
        ascending=[True, True, False],
        inplace=True,
    )

    # k to create feature repetition frequency for different models
    k = []
    for i in [list(i) for i in birth_fs_df["feature_names"].values]:
        k = k + i

    # Keep record of the feature repetition times
    birth_features_frequency = pd.DataFrame(
        {i: k.count(i) for i in set(k)}, index=["Frequency"]
    ).T.sort_values(by="Frequency", ascending=False)

    return birth_fs_df, birth_features_frequency, res_dict


# Final Model Performance - Holdout Result View
def model_result_holdout(
    df_train_eval,
    df_holdout,
    feature_columns_selected,
    target_name,
    random_state_for_eval_split=123,
    eval_positive_number=30,
    eval_negative_number=150,
    train_eval_separation_to_fit=False,
    estimator=LogisticRegression(class_weight="balanced"),
    scalar=MinMaxScaler(),
    voting=None,
):
    """
    Perform model performance test on holdout dataset with trained model with given feature columns. Dataframe
    need to be scaled first - transformed with the previously fitted scalar.
    X_holdout, y_holdout will be determined using feature_columns and target
    :param df_train_eval: train_evaluation dataset that has been engineered and imputed
    :param df_holdout: holdout dataset that has been engineered and imputed
    :param feature_columns_selected: array-like
    :param target_name: str
    :param random_state_for_eval_split=123,
    :param eval_positive_number=30,
    :param eval_negative_number=150,
    :param train_eval_separation_to_fit: boolean
    :param estimator: classifier to be tested
    :param scalar: scalar to be selected
    :return: dictionary containing information for y_true_holdout, y_predicted_holdout, y_probability_holdout
    """
    # Reset index to start from 0
    df_train_eval.reset_index(drop=True, inplace=True)
    df_holdout.reset_index(drop=True, inplace=True)

    # Fit the model with only the train data during feature selection
    # If train_eval_separation_for_train == True,  random_state_for_eval_split, eval_positive_number=30,
    # eval_negative_number=150
    if train_eval_separation_to_fit:
        eval_positive_index = np.random.RandomState(
            random_state_for_eval_split
        ).permutation(df_train_eval[df_train_eval[target_name] == 1].index)[
            :eval_positive_number
        ]
        eval_negative_index = np.random.RandomState(
            random_state_for_eval_split
        ).permutation(df_train_eval[df_train_eval[target_name] == 0].index)[
            :eval_negative_number
        ]

        eval_index = set(list(eval_positive_index) + list(eval_negative_index))
        whole_index = set(df_train_eval.index)

        X_fit = df_train_eval[feature_columns_selected].loc[whole_index - eval_index]
        y_fit = df_train_eval[target_name].loc[whole_index - eval_index]

    else:
        X_fit = df_train_eval[feature_columns_selected]
        y_fit = df_train_eval[target_name]

    X_holdout = df_holdout[feature_columns_selected]
    y_holdout = df_holdout[target_name]

    # Scale first before fit the model
    # For the reproducibility of the result of feature selection and stratified multiple cross-validation,
    # we fit the entire train_eval subset for scalar
    scalar.fit(df_train_eval[feature_columns_selected])
    # scalar.fit(X_fit)

    X_fit = pd.DataFrame(
        scalar.transform(X_fit), columns=X_fit.columns, index=X_fit.index
    )
    X_holdout = pd.DataFrame(
        scalar.transform(X_holdout), columns=X_holdout.columns, index=X_holdout.index
    )

    # Train the model
    estimator.fit(X_fit, y_fit)

    # Fill the result to be returned
    holdout_result = {}
    holdout_result["y_true_holdout"] = y_holdout
    holdout_result["y_predicted_holdout"] = estimator.predict(X_holdout)
    holdout_result["precision_recall_f1_support"] = np.array(
        precision_recall_fscore_support(
            y_holdout, holdout_result["y_predicted_holdout"], labels=[0, 1]
        )
    )
    if voting != "hard":
        holdout_result["y_predicted_prob_holdout"] = estimator.predict_proba(X_holdout)
        holdout_result["average_precision_score"] = average_precision_score(
            y_holdout, holdout_result["y_predicted_prob_holdout"][:, 1]
        )
        holdout_result["roc_auc_score"] = roc_auc_score(
            y_holdout, holdout_result["y_predicted_prob_holdout"][:, 1]
        )
        holdout_result["precision_recall_threshold"] = precision_recall_curve(
            y_holdout, holdout_result["y_predicted_prob_holdout"][:, 1]
        )
        p, r, threshold = precision_recall_curve(
            y_holdout, holdout_result["y_predicted_prob_holdout"][:, 1]
        )
        minimal_divider_buffer = 0.00000001
        f_score = 2 * (p * r) / (p + r + minimal_divider_buffer)
        holdout_result["highest_f1_and_threshold"] = (
            f_score[f_score.argsort()[-1]],
            threshold[f_score.argsort()[-1]],
        )
        holdout_result["y_predicted_holdout_altered_threshold"] = np.where(
            holdout_result["y_predicted_prob_holdout"][:, 1]
            >= threshold[f_score.argsort()[-1]],
            1,
            0,
        )

        # Plotting
        display = PrecisionRecallDisplay.from_predictions(
            y_holdout, holdout_result["y_predicted_prob_holdout"][:, 1]
        )
        _ = display.ax_.set_title(f"{estimator}")

        display = RocCurveDisplay.from_predictions(
            y_holdout, holdout_result["y_predicted_prob_holdout"][:, 1]
        )
        _ = display.ax_.set_title(f"{estimator}")
        plt.plot([0, 1], [0, 1], color="red", lw=2, linestyle="--")

    # Print out result
    print(classification_report(y_holdout, holdout_result["y_predicted_holdout"]))

    display = ConfusionMatrixDisplay.from_predictions(
        y_holdout, holdout_result["y_predicted_holdout"]
    )
    _ = display.ax_.set_title(f"{estimator}")

    return holdout_result, estimator


# An drastic updating of the version of ml_run() which has less, cleaner code and more flexibility
# An drastic updating of the version of ml_run() which has less, cleaner code and more flexibility
def df_ml_run(
        df_train_eval,
        df_holdout,
        feature_columns,
        target_name,
        scalar=MinMaxScaler(),
        scoring_func=average_precision_score,
        importance_scoring="average_precision",
):
    """
    Fast run of a myriad of models and return three very informative dataframes (confusion matrix dataframe,  model performance dataframe
    and colored feature importance dataframe) and one decision tree for visualization

    :param scoring_func: a predefined function, default: f1_score
        model_scoring has to be one of accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
        since it must have the same name of sklearn.metrics functions

    :param importance_scoring: string, default: f1
        importance_scoring has to be one of 'accuracy', 'f1', 'precision', 'recall', 'roc_auc',"average_precision"
        since it must be recognizable as a parameter to be put into permutation importance

    :return: Confusion matrix dataframe, model performance, feature importance

    ------------------------
    Examples:
    ml_result_tuple = df_ml_run(X_train, X_test, y_train, y_test)
    """
    # Reset index to start from 0
    df_train_eval.reset_index(drop=True, inplace=True)
    df_holdout.reset_index(drop=True, inplace=True)

    # Generate trainable dataset and corresponding test dataset
    X_train = df_train_eval[feature_columns]
    y_train = df_train_eval[target_name]
    X_test = df_holdout[feature_columns]
    y_test = df_holdout[target_name]

    scalar.fit(X_train)

    X_train = pd.DataFrame(
        scalar.transform(X_train), columns=X_train.columns, index=X_train.index
    )

    X_test = pd.DataFrame(
        scalar.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    # A dictionary that will store the performance score of models to evaluate feature importance
    model_score = {}

    # A dictionary that will store the confusion matrix results as string to easy comparision
    model_cm = {}

    # A dictionary that will store the feature importance for different models
    model_feature = {}

    # Initializing the model using desired abbreviated names as model instance to train and test

    # xgb = XGBClassifier(learning_rate=0.01, n_estimators=250)
    # dt = DecisionTreeClassifier(criterion="entropy", random_state=0, max_depth=6)
    # rf = RandomForestClassifier(n_estimators=200, random_state=0, max_depth=5)
    # knn = KNeighborsClassifier(n_neighbors=5)
    # svc = SVC(kernel="rbf", C=6)
    # lr = LogisticRegression()
    # nb = GaussianNB()
    # svc = SVC(class_weight="balanced", probability=True)
    # knn = KNeighborsClassifier(n_neighbors=5)
    # lr = LogisticRegression(class_weight="balanced")
    # rf = RandomForestClassifier(class_weight="balanced", random_state=0)
    # xgb = XGBClassifier(class_weight="balanced")
    # dt = DecisionTreeClassifier(
    #     criterion="entropy", random_state=0, max_depth=6, class_weight="balanced"
    # )
    lr = LogisticRegression(C=0.02, solver="lbfgs", class_weight="balanced")

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        max_depth=3,
        max_features=5,
        random_state=2021,
    )

    # XGB
    xgb = XGBClassifier(
        max_depth=3,
        learning_rate=0.01,
        colsample_bytree=0.8,
        scale_pos_weight=15,
        subsample=0.8,
        random_state=2021,
    )
    # SVC
    svc = SVC(
        C=0.02,
        kernel="linear",
        class_weight="balanced",
        probability=True,
        random_state=2021,
    )
    # Decision Tree
    dt = DecisionTreeClassifier(
        criterion="gini",
        max_depth=6,  # Previous is None
        class_weight="balanced",
        random_state=2021,
    )
    # A dictionary that stores the full name of model
    model_name = {
        "knn": "K_Nearest_Neigbhors",
        "xgb": "eXtreme_Gradient_Boost",
        "svc": "Support_Vector_Machine",
        "dt": "Decision_Tree",
        "rf": "Random_Forest",
        "nb": "Naive_Bayes",
        "lr": "Logistic_Regression",
    }

    permutation_importance_list = [(svc, "svc")]
#    permutation_importance_list = [(knn, "knn"), (svc, "svc"), (nb, "nb")]
    feature_importance_list = [(dt, "dt"), (rf, "rf"), (xgb, "xgb")]
    coefficient_list = [(lr, "lr")]

    model_list = (
            permutation_importance_list + feature_importance_list + coefficient_list
    )

    # 1. Search for most effective predictive models and relevant feature importance
    for model, key in model_list:

        # Train and predict
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)

        # Store result
        if (importance_scoring == "average_precision") or (
                importance_scoring == "roc_auc"
        ):
            model_score[key] = scoring_func(y_test, model.predict_proba(X_test)[:, 1])
        else:
            model_score[key] = scoring_func(y_test, predicted)

        model_cm[key] = str(
            {
                k: dict(v)
                for k, v in dict(
                pd.DataFrame(
                    confusion_matrix(y_test, predicted),
                    columns=["Pr0", "Pr1"],
                    index=["Tr0", "Tr1"],
                )
            ).items()
            }
        )
        # Display model result
        print(f"confussion matrix: {confusion_matrix(y_test, predicted)}\n")
        plot_confusion_matrix(model, X_test, y_test)
        print(
            f"The {importance_scoring} score of {model_name[key]}: {model_score[key] * 100} \n"
        )
        print(classification_report(y_test, predicted))

        # Permutation importance
        if (model, key) in permutation_importance_list:

            # Evaluating Feature importance
            permutation_result = permutation_importance(
                model,
                X_train,
                y_train,
                n_repeats=10,
                random_state=1012,
                scoring=importance_scoring,
            )

            # Storing feature importance
            model_feature[key] = permutation_result.importances_mean.reshape(1, -1)[0]

            # Visualization of feature importance for permutation importance
            perm_sorted_idx = permutation_result.importances_mean.argsort()
            plt.figure(figsize=(20, 10))
            plt.title("Feature Importance for {}".format(model_name[key]))
            plt.barh(
                width=permutation_result.importances_mean[perm_sorted_idx].T,
                y=X_train.columns[perm_sorted_idx],
            )
        # Feature Importance for tree-based models:
        elif (model, key) in feature_importance_list:

            # Storing feature importance
            model_feature[key] = model.feature_importances_.reshape(1, -1)[0]

            # Plot visualization of feature importance for tree-based models
            imp_features = pd.DataFrame(
                data=model.feature_importances_,
                index=X_test.columns,
                columns=[model_name[key]],
            )
            imp_features.sort_values(model_name[key], ascending=False, inplace=True)
            plt.figure(figsize=(12, 8), dpi=200)
            sns.barplot(
                data=imp_features,
                y=imp_features.index,
                x=imp_features[model_name[key]],
            )

        # Feature importance for linear-based models:
        elif (model, key) in coefficient_list:

            # Storing feature importance
            model_feature[key] = model.coef_.reshape(1, -1)[0]

            # Plot visualization of feature importance for linear models
            imp_features = pd.DataFrame(
                data=model.coef_.reshape(1, -1)[0],
                index=X_test.columns,
                columns=[model_name[key]],
            )
            imp_features.sort_values(model_name[key], ascending=False, inplace=True)
            plt.figure(figsize=(12, 8), dpi=200)
            sns.barplot(
                data=imp_features,
                y=imp_features.index,
                x=imp_features[model_name[key]],
            )
        else:
            print(
                "{} is not found in the model_list, please check".format((model, key))
            )

    # 2. Put all metrics together for different models
    score_dict = defaultdict(list)
    # models = [lr, nb, rf, knn, dt, svc, xgb]
    # models_label = [
    #     "Logistic_Regression",
    #     "Naive_Bayes",
    #     "Random_Forest",
    #     "K_Nearest_Neigbhors",
    #     "Decision_Tree",
    #     "Support_Vector_Machine",
    #     "eXtreme_Gradient_Boost",
    # ]

    models = [lr, rf, xgb, svc,dt]
    models_label = [
        "Logistic_Regression",
        "Random_Forest",
        "eXtreme_Gradient_Boost",
        "Support_Vector_Machine",
        "Decision_Tree",
    ]

    measurements = [
        precision_score,
        recall_score,
        f1_score,
        average_precision_score,
        roc_auc_score,
        accuracy_score,
    ]
    measurement_names = [
        "Precision",
        "Recall",
        "F1",
        "Average_precision",
        "Roc_auc",
        "Accuracy",
    ]
    for i in range(len(models)):
        score_dict["Model"].append(models_label[i])
        for j in range(len(measurements)):
            if "_" in measurement_names[j]:
                score_dict[measurement_names[j]].append(
                    measurements[j](y_test, models[i].predict_proba(X_test)[:, 1])
                )
            else:
                score_dict[measurement_names[j]].append(
                    measurements[j](y_test, models[i].predict(X_test))
                )

    score_df = pd.DataFrame(score_dict).set_index("Model")

    # Extract highest performing scoring
    score_dict_highest = defaultdict(list)
    # models = [lr, nb, rf, knn, dt, svc, xgb]
    # models_label = [
    #     "Logistic_Regression",
    #     "Naive_Bayes",
    #     "Random_Forest",
    #     "K_Nearest_Neigbhors",
    #     "Decision_Tree",
    #     "Support_Vector_Machine",
    #     "eXtreme_Gradient_Boost",
    # ]
    measurements = [
        precision_score,
        recall_score,
        f1_score,
        average_precision_score,
        roc_auc_score,
        accuracy_score,
        "threshold_positioner",
    ]
    measurement_names = [
        "Precision",
        "Recall",
        "F1",
        "Average_precision",
        "Roc_auc",
        "Accuracy",
        "Threshold",
    ]
    for i in range(len(models)):
        score_dict_highest["Model"].append(models_label[i])

        # Retrieve Threshold and altered Prediction for the calculation of F1, Precision, Recall and Accuracy
        predicted_proba = models[i].predict_proba(X_test)[:, 1]
        p, r, threshold = precision_recall_curve(y_test, predicted_proba)
        f_score = 2 * (p * r) / (p + r + 0.0000001)
        predicted_highest = np.where(
            predicted_proba >= threshold[f_score.argsort()[-1]], 1, 0
        )
        threshold_for_highest = round(threshold[f_score.argsort()[-1]], 2)

        for j in range(len(measurements)):
            if measurement_names[j] == "Threshold":
                score_dict_highest[measurement_names[j]].append(
                    threshold[f_score.argsort()[-1]]
                )
            elif "_" in measurement_names[j]:
                score_dict_highest[measurement_names[j]].append(
                    measurements[j](y_test, predicted_proba)
                )
            else:
                score_dict_highest[measurement_names[j]].append(
                    measurements[j](y_test, predicted_highest)
                )

    score_dict_highest = pd.DataFrame(score_dict_highest).set_index("Model")

    # 3. Create feature importance table for various predictors

    # Prepare naming of models
    col_name = []
    for i in model_feature.keys():
        col_name.append(model_name[i])

    # Convert feature importances dictionary into dataframe, and tranpose to view feature importance on rows
    feature_res = pd.DataFrame(
        data=model_feature.values(), columns=X_test.columns, index=model_feature.keys()
    ).T

    # Standardize the columns while retaining signs of coefficients using dividing maximal value
    for col in feature_res.columns:
        feature_res[col] = feature_res[col] / feature_res[col].max()

    # Calculate weighted feature importance using a myriad of models based on their performance
    weighted_feature = sum(
        [feature_res[col].values * score for col, score in list(model_score.items())]
    )

    # Standardize the weighted_importance column
    feature_res["Weighted_Importance"] = weighted_feature / np.max(weighted_feature)
    col_name.append("Weighted_Importance")

    # Renaming the columns to the model's full name
    feature_res.columns = col_name
    #    feature_res_returned = feature_res.style.background_gradient(cmap="Greens")
    feature_res_returned = feature_res

    # 4. View the confusion matrix for details
    model_cm_df = pd.DataFrame(model_cm, index=["Confusion_Matrix_Summary"]).T

    # Add two more columns for easy observation
    model_cm_df["Pred_0"] = model_cm_df.Confusion_Matrix_Summary.apply(
        lambda x: eval(x)["Pr0"]
    )
    model_cm_df["Pred_1"] = model_cm_df.Confusion_Matrix_Summary.apply(
        lambda x: eval(x)["Pr1"]
    )

    index_name = []
    for i in model_cm_df.index:
        index_name.append(model_name[i])
    model_cm_df.index = index_name

    return score_df, score_dict_highest, feature_res_returned, model_cm_df, dt


# Identify the minimal features that generate highest performance using self-defined strategies.
def df_minimal_features(
        df,
        train_index,
        test_index,
        baseline_features=set(),
        strategy="forward",
        inspection_max=300,  # Times of iteration to search for minimal feature set
        estimator=LogisticRegression(class_weight="balanced"),
        scoring_func=average_precision_score,
):
    """Identify the minimal features that generate highest performance.
    Critical assumptions: if model performance didn't increase because of any one feature inclusion/exclusion, then the performance plateaued.
    This assumption could easily be flawed, if synergies of a combination of features (two or more) can be captured by the model.

    Parameters:
    -------------------
    df: Dataframe
        Processed dataframe where target is defined as "y".

    train_index: list
        Index list of the train dataset

    test_index: list
        Index list of the test dataset

    baseline_features: set, default, empty set
        Initial starting matrix

    strategy: string, default, "forward"
        "forward": start from features of any length and to include one extra feature with the purpose of increasing model performance.
        "backward": start from full features and to exclude one feature at a time in order to increase model performance.
        "bi_direction": first forward, then backward. Specifically, start from features of any length and begin to include one extra feature of maximal impact to increase model performance. If no increase can be achieved, then exclude one feature at a time to shrink the feature list.
        "retrospective": start from baseline features. For each step, two inspection will be performed. During inclusion inspection, we see whether an increase of performance can be achived. If ture, then add features with maximal impact. If none can increase the performance, an marker will be created to indicate such situation.
        Then comes exclusion inpection, where we drop features that hurt the model performance. Such retrospective one-step inspection will stop if neither inclusion nor exclusion can increase model performance. (Performance plateaued)

    Returns:
    --------------------
    A set of minimal features, and A dictionary of all calcuated combinations.

    """

    feature_combination_result = {}

    X_train = df.loc[train_index, :].drop(columns="y")
    y_train = df.loc[train_index, :]["y"]
    X_test = df.loc[test_index, :].drop(columns="y")
    y_test = df.loc[test_index, :]["y"]

    if baseline_features:
        estimator.fit(X_train[baseline_features], y_train)
        predicted = estimator.predict(X_test[baseline_features])
        feature_combination_result[str(baseline_features)] = scoring_func(
            y_test, predicted
        )
        print(
            f"Confussion Matrix for features {baseline_features} is: {confusion_matrix(y_test, predicted)}\n"
        )
        current_score = scoring_func(y_test, predicted)

    else:  # if baseline_feature is empty, then create a one-element baseline feature space
        for feature_name in X_train.columns:
            baseline_features.add(feature_name)
            estimator.fit(X_train[baseline_features], y_train)
            predicted = estimator.predict(X_test[baseline_features])
            feature_combination_result[str(baseline_features)] = scoring_func(
                y_test, predicted
            )
            print(
                f"Confussion Matrix for features {baseline_features} is: {confusion_matrix(y_test, predicted)}\n"
            )
            baseline_features.remove(feature_name)

        else:
            baseline_features = eval(
                max(feature_combination_result, key=feature_combination_result.get)
            )
            current_score = max(feature_combination_result.values())

    # Forward Search
    if strategy == "forward":
        for i in range(len(set(X_train.columns) - baseline_features)):
            for feature_name in set(X_train.columns) - baseline_features:
                baseline_features.add(feature_name)
                estimator.fit(X_train[baseline_features], y_train)
                predicted = estimator.predict(X_test[baseline_features])
                feature_combination_result[str(baseline_features)] = scoring_func(
                    y_test, predicted
                )
                print(
                    f"Confussion Matrix for features {baseline_features} is: {confusion_matrix(y_test, predicted)}\n"
                )
                baseline_features.remove(feature_name)

            else:  # After each step of inspection
                max_score = max(feature_combination_result.values())
                if max_score > current_score:
                    # Redefine baseline_feature through extracting the key with highest value
                    baseline_features = eval(
                        max(
                            feature_combination_result,
                            key=feature_combination_result.get,
                        )
                    )
                    current_score = max_score
                else:
                    return baseline_features, feature_combination_result
        else:  # If model performance keep increasing with each step of inclusion inspection
            return baseline_features, feature_combination_result

    # Backward search
    elif strategy == "backward":
        for i in range(len(set(baseline_features))):
            if (
                    len(set(baseline_features)) <= 1
            ):  # If there is only one element for the baseline_features
                return baseline_features, feature_combination_result
            else:
                for feature_name in baseline_features:  # Perform inspection
                    baseline_features.remove(feature_name)
                    estimator.fit(X_train[baseline_features], y_train)
                    predicted = estimator.predict(X_test[baseline_features])
                    feature_combination_result[str(baseline_features)] = scoring_func(
                        y_test, predicted
                    )
                    print(
                        f"Confussion Matrix for features {baseline_features} is: {confusion_matrix(y_test, predicted)}\n"
                    )
                    baseline_features.add(feature_name)
                else:  # After each round of inspection
                    max_score = max(feature_combination_result.values())
                    if max_score > current_score:
                        # Redefine baseline_feature through extracting the key with highest value
                        baseline_features = eval(
                            max(
                                feature_combination_result,
                                key=feature_combination_result.get,
                            )
                        )
                        current_score = max_score
                    else:
                        return baseline_features, feature_combination_result

    # Bi-direction search algorithm
    elif strategy == "bi_direction":
        for i in range(len(set(X_train.columns) - baseline_features)):
            for feature_name in set(X_train.columns) - baseline_features:
                baseline_features.add(feature_name)
                estimator.fit(X_train[baseline_features], y_train)
                predicted = estimator.predict(X_test[baseline_features])
                feature_combination_result[str(baseline_features)] = scoring_func(
                    y_test, predicted
                )
                print(
                    f"Confussion Matrix for features {baseline_features} is: {confusion_matrix(y_test, predicted)}\n"
                )
                baseline_features.remove(feature_name)

            else:  # After each step of inspection
                max_score = max(feature_combination_result.values())
                if max_score > current_score:
                    # Redefine baseline_feature through extracting the key with highest value
                    baseline_features = eval(
                        max(
                            feature_combination_result,
                            key=feature_combination_result.get,
                        )
                    )
                    current_score = max_score
                else:
                    # Begin backward search
                    for i in range(len(set(baseline_features))):
                        if (
                                len(set(baseline_features)) <= 1
                        ):  # If there is only one element for the baseline_features
                            return baseline_features, feature_combination_result
                        else:
                            for feature_name in baseline_features:  # Perform inspection
                                baseline_features.remove(feature_name)
                                estimator.fit(X_train[baseline_features], y_train)
                                predicted = estimator.predict(X_test[baseline_features])
                                feature_combination_result[
                                    str(baseline_features)
                                ] = scoring_func(y_test, predicted)
                                print(
                                    f"Confussion Matrix for features {baseline_features} is: {confusion_matrix(y_test, predicted)}\n"
                                )
                                baseline_features.add(feature_name)
                            else:  # After each round of inspection
                                max_score = max(feature_combination_result.values())
                                if max_score > current_score:
                                    # Redefine baseline_feature through extracting the key with highest value
                                    baseline_features = eval(
                                        max(
                                            feature_combination_result,
                                            key=feature_combination_result.get,
                                        )
                                    )
                                    current_score = max_score
                                else:
                                    return baseline_features, feature_combination_result

        else:  # If model performance keep increasing with any inclusion of features
            # Begin backward search
            for i in range(len(set(baseline_features))):
                if (
                        len(set(baseline_features)) <= 1
                ):  # If there is only one element for the baseline_features
                    return baseline_features, feature_combination_result
                else:
                    for feature_name in baseline_features:  # Perform inspection
                        baseline_features.remove(feature_name)
                        estimator.fit(X_train[baseline_features], y_train)
                        predicted = estimator.predict(X_test[baseline_features])
                        feature_combination_result[
                            str(baseline_features)
                        ] = scoring_func(y_test, predicted)
                        print(
                            f"Confussion Matrix for features {baseline_features} is: {confusion_matrix(y_test, predicted)}\n"
                        )
                        baseline_features.add(feature_name)
                    else:  # After each round of inspection
                        max_score = max(feature_combination_result.values())
                        if max_score > current_score:
                            # Redefine baseline_feature through extracting the key with highest value
                            baseline_features = eval(
                                max(
                                    feature_combination_result,
                                    key=feature_combination_result.get,
                                )
                            )
                            current_score = max_score
                        else:
                            return baseline_features, feature_combination_result

    # Retrospective one-step search algorithm
    # Two way inspection
    elif strategy == "retrospective":

        # Create a large number for the times of inspection for retrospective one-step inspection.
        print(
            f"Total number of retrospective bi-direction inspection is {inspection_max} "
        )

        # For signaling successfuly identified the minimal subset of features that render the highest performance.
        inclusion_marker = 0  # 1 means that adding of any one feature failed to increase model performance for the current baseline feature space, default is 0, suggesting the the model performance could be increased with an inclusion.
        exclusion_marker = 0  # 1 means exclusion of any one feature failed to improve model performance for the current subset of feature, default is 0, suggesting the the model performance could be increased with an exclusion.

        # Assumption: if one feature is found to hurt the model, then this feature might also hurt the model with its combination of different subset of features.
        exclusion_list = (
            set()
        )  # This assumption could possibly be flawed as it omits the synergies of the feature with other combinations of features. Nevertheless, it is still introduced with the purpose to decrease the computation repetition. And provide a way out of perpetuating loop.
        inpection_count = 0  # This number is used to see how many inspection have been conducted so far

        # Now begin to identify minimal subset of features
        for i in range(inspection_max):

            # Check whether any of the termination criterion is met
            if inclusion_marker * exclusion_marker:
                print(
                    f"Congrat! The minimal subset of feature has been identified given the baseline feature after {inpection_count} inspections!"
                )
                return baseline_features, feature_combination_result

            # Perform inclusion inspection
            for feature_name in (
                    set(X_train.columns) - baseline_features - exclusion_list
            ):
                baseline_features.add(feature_name)
                estimator.fit(X_train[baseline_features], y_train)
                predicted = estimator.predict(X_test[baseline_features])
                feature_combination_result[str(baseline_features)] = scoring_func(
                    y_test, predicted
                )
                print(
                    f"Confussion Matrix for features {baseline_features} is: {confusion_matrix(y_test, predicted)}\n"
                )
                baseline_features.remove(feature_name)

            else:  # After each step of inspection
                max_score = max(feature_combination_result.values())
                if max_score > current_score:
                    # Redefine baseline_feature through extracting the key with highest value
                    inclusion_marker = 0
                    baseline_features = eval(
                        max(
                            feature_combination_result,
                            key=feature_combination_result.get,
                        )
                    )
                    current_score = max_score
                else:
                    # Mark such situation where inclusion of any one feature failed to increase model performance
                    inclusion_marker = 1

            if (len(set(baseline_features)) <= 1) & (
                    inclusion_marker == 1
            ):  # If there is only one element for the baseline_features

                print("It seems that one feature can produce highest performance!")
                return baseline_features, feature_combination_result

            else:
                # Perform exclusion inspection

                # Check whether any of the termination criterion is met
                if inclusion_marker * exclusion_marker == 1:
                    print(
                        f"Congrat! The minimal subset of feature has been identified given the baseline feature after {inpection_count} inspections!"
                    )
                    return baseline_features, feature_combination_result

                for feature_name in baseline_features:  # Perform inspection
                    baseline_features.remove(feature_name)
                    estimator.fit(X_train[baseline_features], y_train)
                    predicted = estimator.predict(X_test[baseline_features])
                    feature_combination_result[str(baseline_features)] = scoring_func(
                        y_test, predicted
                    )
                    print(
                        f"Confussion Matrix for features {baseline_features} is: {confusion_matrix(y_test, predicted)}\n"
                    )
                    baseline_features.add(feature_name)
                else:  # After each round of inspection
                    max_score = max(feature_combination_result.values())
                    if max_score > current_score:
                        # Redefine baseline_feature through extracting the key with highest value
                        exclusion_marker = 0
                        baseline_features_one_less = eval(
                            max(
                                feature_combination_result,
                                key=feature_combination_result.get,
                            )
                        )
                        exclusion_list.update(
                            baseline_features - baseline_features_one_less
                        )
                        baseline_features = baseline_features_one_less
                        current_score = max_score
                    else:
                        # Mark such situation where exclusion of any one feature failed to increase model performance
                        exclusion_marker = 1

            inpection_count = inpection_count + 1

        else:  # If model performance keep increasing with any inclusion of features
            print(
                f"The optimal subset of feature after {inspection_max} times of inspection are generated!"
            )
            return baseline_features, feature_combination_result


# # Define your dataset based on target variable, which cannot be controlled in ML pipeline
# def target_selector(df, target_name="Asthma_Diagnosis_5yCLA", target_mapping={2: 1}, include_dust=False):
#     """Define your target variable, the mapping schemes, and whether to include dust sampling data for modelling.
#     Can only be used when it's the raw data with inclusive/complete feature columns
#
#     Parameters:
#     -----------------
#     df: DataFrame to choose target from
#
#     target_name : str,  default 'Asthma_Diagnosis_5yCLA'
#         A string that is selected from TARGET_VAR_REPO list to control the entire dataframe to be processed
#
#     target_mapping : dictionary, default {2:1} that is suggesting possible asthma(2) will be treated as asthma(0)
#         A dictionary used to specify how to map the "possible", or, or other variables
#
#     include_dust : boolean, default is False
#         A boolean for certain attribute selection that will drastically reduce sample size for analysis
#
#     Returns:
#     ----------------
#     Targeted df, in which target will be renamed to 'y'
#     """
#
#     # df_child variables sanity check
#     print(f"The dimension of original dataframe for process is {df.shape}")
#     print("Number of categorical features is: ", len(CAT_VARS_REPO))
#     print("Number of numeric features is: ", len(NUM_VARS_REPO))
#     print("Number of target variables is: ", len(TARGET_VAR_REPO))
#     print("Number of dropped features is: ", len(FEA_TO_DROP))
#     print("The difference of features of targeted and original dataframe is :{0}".format((set(df.columns.values) - set(
#         TARGET_VAR_REPO + NUM_VARS_REPO + CAT_VARS_REPO + FEA_TO_DROP))))
#     print(f"The number of missing value for target label is: {df[target_name].isna().sum()}")
#
#     # Print selectable target variables
#     print("------------------------------------------------------")
#     print("Note: Target variable can be one of: \n", TARGET_VAR_REPO)
#     print("------------------------------------------------------")
#     print("****Target variable will be renamed to y for easy access.**** \n")
#
#     y = df[target_name].copy().replace(target_mapping)
#
#     if include_dust:
#         X = df[NUM_VARS_REPO + CAT_VARS_REPO + ["Subject_Number"]].copy()
#         df_temp = pd.concat([X, y], axis=1)
#
#         df_Xy = df_temp.dropna(subset=SUB_NUM_HOMEDUST + [target_name]).reset_index(
#             drop=True
#         )
#
#         df_Xy.set_index(
#             "Subject_Number", inplace=True
#         )  # Use Index as an extra information
#
#         X_return = df_Xy.drop(columns=[target_name])
#         y_return = df_Xy[target_name]
#
#         return df_Xy.rename(columns={target_name: "y"}), X_return, y_return
#
#     else:
#         X = (
#             df[NUM_VARS_REPO + CAT_VARS_REPO + ["Subject_Number"]]
#                 .copy()
#                 .drop(columns=SUB_NUM_HOMEDUST)
#         )
#         df_temp = pd.concat([X, y], axis=1)
#
#         df_Xy = df_temp.dropna(subset=[target_name]).reset_index(drop=True)
#
#         df_Xy.set_index(
#             "Subject_Number", inplace=True
#         )  # Use Index as an extra information
#
#         return df_Xy.rename(columns={target_name: "y"})


# Define a new target selector
def df_target_selector(df, target_name="Asthma_Diagnosis_5yCLA", target_mapping={2: np.nan}):
    """Define your target variable, the mapping schemes.

    Parameters:
    -----------------
    df: DataFrame to choose target from

    target_name : str,  default 'Asthma_Diagnosis_5yCLA'
        A string that is selected from TARGET_VAR_REPO list to control the entire dataframe to be processed

    target_mapping : dictionary, default {2:1} that is suggesting possible asthma(2) will be treated as asthma(0)
        A dictionary used to specify how to map the "possible", or, or other variables

    Returns:
    ----------------
    Targeted df, in which target variable will be renamed to 'y'
    """

    # df_child variables sanity check
    print(f"The dimension of original dataframe for process is {df.shape}")
    print(f"The number of missing value for target label is: {df[target_name].isna().sum()}")

    # Print selectable target variables
    print("****Target variable will be renamed to y for easy access.**** ")

    df[target_name] = df[target_name].replace(target_mapping)

    df_Xy = df[df[target_name].notna()].rename(columns={target_name: "y"})

    if "Subject_Number" in df_Xy.columns:
        df_Xy.set_index("Subject_Number", inplace=True)

    return df_Xy


# Select a subset of samples with allowable missingness for further imputation and modelling
def sample_selector(df, max_nan_per_sample=10, differentiate=False):
    """
    Shrink the samples based on the number of missing values while perserve those samples with positive asthma outcome

    Parameters:
    ----------
    df: DataFrame
        Pandas DataFrame to be processed

    na_number_thresh: integer, default 10
        The maximal number of missing value you want to impute for each sample if they have a negative asthma diagnosis.

    differentiate: boolean, default True
        If True, then only those with negative outcome will be dropped from further exploration and modelling,
        suggesting you only take interest in further exploring those with a positive outcome (asthma). If False, then
        all of samples will be treated equally with the only criteria of missingness

    Returns:
    ---------
    Shrunk df, X, y, df_rows_dropped
    """
    df_samples = df.copy()  # The original one being passed won't be altered
    index_to_drop = []
    if differentiate:
        for i, j in df_samples.iterrows():
            if (j["y"] == 0) & (j.isna().sum() > max_nan_per_sample):
                index_to_drop.append(i)
    else:
        for i, j in df_samples.iterrows():
            if j.isna().sum() > max_nan_per_sample:
                index_to_drop.append(i)

    y_positive_num = df_samples[df_samples["y"] == 1].shape[0]

    df_samples_dropped = df_samples.loc[index_to_drop, :].copy()
    #     df_samples_dropped.reset_index(drop=True, inplace=True)

    y_positive_dropped_num = df_samples_dropped["y"][
        df_samples_dropped["y"] == 1
        ].shape[0]

    df_samples_kept = df_samples.drop(index=index_to_drop).copy()
    #     df_samples_kept.reset_index(drop=True, inplace=True)

    X = df_samples_kept.drop(columns="y")
    y = df_samples_kept["y"]

    print(
        f"A total of {y_positive_dropped_num} / {y_positive_num} ({round((y_positive_dropped_num / y_positive_num), 3) * 100}%) for asthma positive are dropped due to more than {max_nan_per_sample} missing value in the sample."
    )

    total_sample_dropped = len(df_samples_dropped.isna().sum(axis=1))

    print("")
    print(f"The total number of dropped samples is {total_sample_dropped}")
    print("************************************************************************")
    print("An excel file has been written to /output/unprocessed_targeted_raw_child.xlsx")
    df_samples_kept.to_excel("../output/unprocessed_targeted_raw_child.xlsx")

    return df_samples_kept, X, y, df_samples_dropped


# Generate Test Dataset for feature selection analysis and ML pipeline
def generate_trainable_dataset(X, y, add_indicator_threshold=30):
    """Generate one sample dataset using customized transformers that can be directly trained and save the sample dataset
    """
    # The transformers run will alter X inplace
    ApgarTransformer(engineer_type="Log1p").fit_transform(X)
    BirthTransformer().fit_transform(X)
    Log1pTransformer().fit_transform(X)
    RespiratoryTransformer(first_18m_divide=True).fit_transform(X)
    DiscretizePSS().fit_transform(X)
    ColumnFilter().fit_transform(X)
    ImputerStrategizer().fit_transform(X)
    CatNaNImputer().fit_transform(X)
    NumNaNimputer().fit_transform(X)
    CollinearRemover().fit_transform(X)

    # Generate test df dataset for feature selection pipeline
    df_for_ml = pd.concat([X, y], axis=1)

    print("A trainable dataset is created in the ..output/CHILD_ml_sample_dataset.xlsx!")

    df_for_ml.to_excel(CHILD_RAW_DIR + "CHILD_ml_sample_dataset.xlsx")

    return df_for_ml


# Quick overview of self-written functions and classes of imported libraries
def view_module_functions(module_alias):  # Examples: print(view_module_functions(UT))
    """
    Quick review of self-written functions and classes
    """
    return [
        i
        for i in dir(module_alias)
        if (not i.isupper()) & ~(i.startswith("_")) & (len(i) > 6)
    ]


# Summary of created dataframe for CHILD unique values, top percentage, etc
def df_summary(X):
    """
    Overview the number of unique values, Top percentage of value distribution, for each feature
    """

    cols_to_inspect = X.columns

    top_percentage = {}
    for i in X[cols_to_inspect].columns:
        top_percentage[i] = round(
            (X[i].value_counts(normalize=True).values[0] * 100), 2
        )
    per_ser = pd.Series(top_percentage)

    df_overview = (
        pd.concat(
            [
                X[cols_to_inspect].nunique(),
                X[cols_to_inspect].median(),
                per_ser,
                X[cols_to_inspect].max(),
                X[cols_to_inspect].sum(),
                X[cols_to_inspect].var(),
            ],
            axis=1,
        )
            .rename(
            columns={
                0: "Num_Unique_Values",
                1: "Median_Value",
                2: "Top_Percentage",
                3: "Max_Value",
                4: "Number_of_Binary_Positive",
                5: "Variance",
            }
        )
            .sort_values(by="Top_Percentage", ascending=False)
    )

    return df_overview


# Gadget to view the asthma proportions for different columns during feature selection
def view_y_proportions(df, columns_of_interest, thresh=0):
    """
    Gadget to view the asthma proportions for different columns during feature selection
    :param columns_of_interest = list of features
    :param thresh = 0
    :return dataframe of information
    """
    general_proportion = round((100 * df[df.y == 1].shape[0] / df.shape[0]), 2)
    print(
        "The proportion of definite asthma outcome for general cohorts is {}%".format(
            general_proportion
        )
    )
    asthma_for_col_number = []
    asthma_for_col_proportion = []
    col_with_thresh_number = []
    for i in columns_of_interest:

        total_of_subsamples = df[df[i] > thresh].shape[0]

        if 1 in df[df[i] > thresh].y.value_counts().sort_values(ascending=True).index:
            number_of_asthma = (
                df[df[i] > thresh].y.value_counts().sort_values(ascending=True).loc[1]
            )
        else:
            number_of_asthma = 0

        proportion = round(
            (number_of_asthma / (total_of_subsamples + 0.000001) * 100), 2
        )

        col_with_thresh_number.append(total_of_subsamples)
        asthma_for_col_proportion.append(proportion)
        asthma_for_col_number.append(number_of_asthma)
    #         print(
    #             "The proportion of asthma outcome when the value of {} is greater than {} is {}% or {} out of {}".format(
    #                 i, thresh, proportion, number_of_asthma, total_of_subsamples
    #             )
    #         )

    col_y_df = pd.DataFrame(
        [
            asthma_for_col_proportion,
            asthma_for_col_number,
            col_with_thresh_number,
            np.ones(len(col_with_thresh_number)) * thresh,
        ],
        index=[
            "Asthma_Proportion_over_thresh",
            "Asthma_Outcome_over_thresh",
            "Total_Number_over_thresh",
            "Thresh",
        ],
        columns=columns_of_interest,
    )

    general_ser = pd.Series(
        [general_proportion, df[df.y == 1].shape[0], df.shape[0], 0],
        name="*****GENERAL*****",
        index=[
            "Asthma_Proportion_over_thresh",
            "Asthma_Outcome_over_thresh",
            "Total_Number_over_thresh",
            "Thresh",
        ],
    )

    return col_y_df.T.append(general_ser).sort_values(
        by="Asthma_Proportion_over_thresh", ascending=False
    )


# View results of feature importance for a random subset of original dataframe
def randomsubset_permutation_importance(*, X=None, y=None, clf: object, percentile_of_features: float):
    """
    As when all features are included, none of the feature will have any importance, I therefore created this function
    to view the feature importance of a random subset of features.
    : para: percentile_of_features
    : para: clf: a classification algorithm
    : return: a visualization of feature importance for current subset of features
    """
    number_of_features = int(len(X.columns.values) * percentile_of_features)
    selected_columns = random.sample(list(X.columns.values), number_of_features)
    clf.fit(X[selected_columns], y)
    result = permutation_importance(
        clf, X[selected_columns], y, n_repeats=10, random_state=1012, scoring="roc_auc",
    )
    perm_sorted_idx = result.importances_mean.argsort()
    plt.figure(figsize=(20, 10))
    plt.barh(
        width=result.importances_mean[perm_sorted_idx].T, y=X.columns[perm_sorted_idx],
    )
    r = result
    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            print(
                f"{X.columns[i]:<8}"
                f"{r.importances_mean[i]:.3f}"
                f" +/- {r.importances_std[i]:.3f}"
            )


# Sample run various ML models and return a series of dictionary for viewing information
def ml_run(X_train, X_test, y_train, y_test, scoring_func=f1_score, importance_scoring='f1'):
    """
    Current model cohorts include:
        Linear model: Logistic Regression,
        Probability model: Gaussian Naive Bayes,
        Tree model: Decision Tree,
        Boundary based: Support Vector Machine,
        Distance based: K Nearest Neighbors,
        Ensemble bagging: Random Forest,
        Ensemble boosting: eXtreme Gradient Boost,
    Other models that could be added:
        Neural Network: Multi-layer Perceptron Classifier
        Ensemble voting: Soft voting based on probability/Hard voting
        Ensemble stacking: Use a mega-estimator and previous results as input to predict

    :param scoring_func: a predefined function, default: f1_score
        model_scoring has to be one of accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
        since it must have the same name of sklearn.metrics functions

    :param importance_scoring: string, default: f1
        importance_scoring has to be one of 'accuracy', 'f1', 'precision', 'recall', 'roc_auc',
        since it must be recognizable as a parameter to be put into permutation importance

    :return: Confusion matrix dataframe, model performance, feature importance

    ------------------------
    Examples:
    ml_tuple = ml_run(X_train, X_test, y_train, y_test)
    """
    # A dictionary that will store the performance score of models to evaluate feature importance
    model_score = {}

    # A dictionary that will store the confusion matrix results as string to easy comparision
    model_cm = {}

    # A dictionary that will store the feature importance for different models
    model_feature = {}

    # ---------------------------------------------------------------------------
    # Model 1: Logistic Regression
    # ---------------------------------------------------------------------------

    # Train & Predict
    model_lr = "Logistic_Regression"
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    predicted = lr.predict(X_test)

    # Record results
    model_score[model_lr] = scoring_func(y_test, predicted)
    model_cm[model_lr] = str(
        {
            k: dict(v)
            for k, v in dict(
            pd.DataFrame(
                confusion_matrix(y_test, predicted),
                columns=["Pred_0", "Pred_1"],
                index=["True_0", "True_1"],
            )
        ).items()
        }
    )
    model_feature[model_lr] = lr.coef_.reshape(1, -1)[0]

    # Display model info
    print(f"confussion matrix: {confusion_matrix(y_test, predicted)}\n")
    print(
        f"The performance score of Logistic Regression: {model_score[model_lr] * 100} \n"
    )
    print(classification_report(y_test, predicted))

    # Plot visualization of feature importance
    imp_features_lr = pd.DataFrame(
        data=lr.coef_.reshape((-1, 1)), index=X_test.columns, columns=[model_lr],
    )
    imp_features_lr.sort_values(model_lr, ascending=False, inplace=True)
    plt.figure(figsize=(12, 8), dpi=200)
    sns.barplot(
        data=imp_features_lr, y=imp_features_lr.index, x=imp_features_lr[model_lr],
    )

    # ---------------------------------------------------------------------------
    # Model 2: Gaussian Naive Bayes
    # ---------------------------------------------------------------------------

    # Train & Predict
    model_nb = "Naive_Bayes"
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    predicted = nb.predict(X_test)

    # Record results
    model_score[model_nb] = scoring_func(y_test, predicted)
    model_cm[model_nb] = str(
        {
            k: dict(v)
            for k, v in dict(
            pd.DataFrame(
                confusion_matrix(y_test, predicted),
                columns=["Pred_0", "Pred_1"],
                index=["True_0", "True_1"],
            )
        ).items()
        }
    )

    result = permutation_importance(
        nb,
        X_train,
        y_train,
        n_repeats=10,
        random_state=1012,
        scoring=importance_scoring,
    )

    model_feature[model_nb] = result.importances_mean.reshape(1, -1)[0]

    # Display model info
    print(f"confussion matrix: {confusion_matrix(y_test, predicted)}\n")
    print(
        f"The performance score of Gaussian Naive Bayes: {model_score[model_nb] * 100} \n"
    )
    print(classification_report(y_test, predicted))

    # Visualization of feature importance for permutation importance
    perm_sorted_idx = result.importances_mean.argsort()
    plt.figure(figsize=(20, 10))
    plt.title("Feature Importance for Gaussian Naive Bayes Modelling")
    plt.barh(
        width=result.importances_mean[perm_sorted_idx].T,
        y=X_train.columns[perm_sorted_idx],
    )

    # ---------------------------------------------------------------------------
    # Model 3: Random Forest Classifier
    # ---------------------------------------------------------------------------

    # Train & Predict
    model_rf = "Random_Forest"
    rf = RandomForestClassifier(n_estimators=20, random_state=12, max_depth=5)
    rf.fit(X_train, y_train)
    predicted = rf.predict(X_test)

    # Record results
    model_score[model_rf] = scoring_func(y_test, predicted)
    model_cm[model_rf] = str(
        {
            k: dict(v)
            for k, v in dict(
            pd.DataFrame(
                confusion_matrix(y_test, predicted),
                columns=["Pred_0", "Pred_1"],
                index=["True_0", "True_1"],
            )
        ).items()
        }
    )

    model_feature[model_rf] = rf.feature_importances_.reshape(1, -1)[0]

    # Display model info
    print(f"confussion matrix: {confusion_matrix(y_test, predicted)}\n")
    print(f"The performance score of Random Forest: {model_score[model_rf] * 100} \n")
    print(classification_report(y_test, predicted))

    # Plot visualization of feature importance
    imp_features_rf = pd.DataFrame(
        data=rf.feature_importances_, index=X_test.columns, columns=[model_rf],
    )
    imp_features_rf.sort_values(model_rf, ascending=False, inplace=True)
    plt.figure(figsize=(12, 8), dpi=200)
    sns.barplot(
        data=imp_features_rf, y=imp_features_rf.index, x=imp_features_rf[model_rf],
    )

    # ---------------------------------------------------------------------------
    # Model 4: K Nearest Neighbors
    # ---------------------------------------------------------------------------

    # Train & Predict
    model_knn = "K_Nearest_Neighbors"
    knn = KNeighborsClassifier(n_neighbors=12)
    knn.fit(X_train, y_train)
    predicted = knn.predict(X_test)

    # Record results
    model_score[model_knn] = scoring_func(y_test, predicted)
    model_cm[model_knn] = str(
        {
            k: dict(v)
            for k, v in dict(
            pd.DataFrame(
                confusion_matrix(y_test, predicted),
                columns=["Pred_0", "Pred_1"],
                index=["True_0", "True_1"],
            )
        ).items()
        }
    )

    result = permutation_importance(
        knn,
        X_train,
        y_train,
        n_repeats=10,
        random_state=1012,
        scoring=importance_scoring,
    )

    model_feature[model_knn] = result.importances_mean.reshape(1, -1)[0]

    # Display model info
    print(f"Confussion matrix: {confusion_matrix(y_test, predicted)}\n")
    print(
        f"The performance score of K Nearest Neighbors: {model_score[model_knn] * 100} \n"
    )
    print(classification_report(y_test, predicted))

    # Visualization of feature importance for permutation importance
    perm_sorted_idx = result.importances_mean.argsort()
    plt.figure(figsize=(12, 8), dpi=200)
    plt.title("Feature Importance for K Nearest Neighbors Modelling")
    plt.barh(
        width=result.importances_mean[perm_sorted_idx].T,
        y=X_train.columns[perm_sorted_idx],
    )

    # ---------------------------------------------------------------------------
    # Model 5: Decision Tree
    # ---------------------------------------------------------------------------

    # Train & Predict
    model_dt = "Decision_Tree"
    dt = DecisionTreeClassifier(criterion="entropy", random_state=0, max_depth=6)
    dt.fit(X_train, y_train)
    predicted = dt.predict(X_test)

    # Record results
    model_score[model_dt] = scoring_func(y_test, predicted)
    model_cm[model_dt] = str(
        {
            k: dict(v)
            for k, v in dict(
            pd.DataFrame(
                confusion_matrix(y_test, predicted),
                columns=["Pred_0", "Pred_1"],
                index=["True_0", "True_1"],
            )
        ).items()
        }
    )

    model_feature[model_dt] = dt.feature_importances_.reshape(1, -1)[0]

    # Display model info
    print(f"Confussion matrix: {confusion_matrix(y_test, predicted)}\n")
    print(f"The performance score of Random Forest: {model_score[model_dt] * 100} \n")
    print(classification_report(y_test, predicted))

    # Plot visualization of feature importance
    imp_features_dt = pd.DataFrame(
        data=dt.feature_importances_, index=X_test.columns, columns=[model_dt],
    )
    imp_features_dt.sort_values(model_dt, ascending=False, inplace=True)
    plt.figure(figsize=(12, 8), dpi=200)
    sns.barplot(
        data=imp_features_dt, y=imp_features_dt.index, x=imp_features_dt[model_dt],
    )

    # ---------------------------------------------------------------------------
    # Model 6: Support Vector Machine
    # ---------------------------------------------------------------------------

    # Train & Test
    model_svc = "Support Vector Classifier"
    svc = SVC(kernel="rbf", C=6)
    svc.fit(X_train, y_train)
    predicted = svc.predict(X_test)

    # Record results
    model_score[model_svc] = scoring_func(y_test, predicted)
    model_cm[model_svc] = str(
        {
            k: dict(v)
            for k, v in dict(
            pd.DataFrame(
                confusion_matrix(y_test, predicted),
                columns=["Pred_0", "Pred_1"],
                index=["True_0", "True_1"],
            )
        ).items()
        }
    )

    result = permutation_importance(
        svc,
        X_train,
        y_train,
        n_repeats=10,
        random_state=1012,
        scoring=importance_scoring,
    )

    model_feature[model_svc] = result.importances_mean.reshape(1, -1)[0]

    # Display model info
    print(f"Confussion matrix: {confusion_matrix(y_test, predicted)}\n")
    print(
        f"The performance score of K Nearest Neighbors: {model_score[model_svc] * 100} \n"
    )
    print(classification_report(y_test, predicted))

    # Visualization of feature importance for permutation importance
    perm_sorted_idx = result.importances_mean.argsort()
    plt.figure(figsize=(12, 8), dpi=200)
    plt.title("Feature Importance for Support Vector Machine Modelling")
    plt.barh(
        width=result.importances_mean[perm_sorted_idx].T,
        y=X_train.columns[perm_sorted_idx],
    )

    # ---------------------------------------------------------------------------
    # Model 7: Extreme Gradient Boost
    # ---------------------------------------------------------------------------

    # Train & Test
    model_xgb = "Extreme_Gradient_Boost"
    xgb = XGBClassifier(learning_rate=0.01, n_estimators=25)
    xgb.fit(X_train, y_train)
    predicted = xgb.predict(X_test)

    # Record results
    model_score[model_xgb] = scoring_func(y_test, predicted)
    model_cm[model_xgb] = str(
        {
            k: dict(v)
            for k, v in dict(
            pd.DataFrame(
                confusion_matrix(y_test, predicted),
                columns=["Pred_0", "Pred_1"],
                index=["True_0", "True_1"],
            )
        ).items()
        }
    )

    model_feature[model_xgb] = rf.feature_importances_.reshape(1, -1)[0]

    # Display model info
    print(f"confussion matrix: {confusion_matrix(y_test, predicted)}\n")
    print(f"The performance score of Random Forest: {model_score[model_xgb] * 100} \n")
    print(classification_report(y_test, predicted))

    # Plot visualization of feature importance
    imp_features_xgb = pd.DataFrame(
        data=xgb.feature_importances_, index=X_test.columns, columns=[model_xgb],
    )
    imp_features_xgb.sort_values(model_xgb, ascending=False, inplace=True)
    plt.figure(figsize=(12, 8), dpi=200)
    sns.barplot(
        data=imp_features_xgb, y=imp_features_xgb.index, x=imp_features_xgb[model_xgb],
    )

    # Put all metrics together
    score_dict = defaultdict(list)

    models = [lr, nb, rf, knn, dt, svc, xgb]
    model_names = [
        "Logistic Regression",
        "Naive Bayes",
        "Random Forest",
        "K-Nearest Neighbour",
        "Decision Tree",
        "Support Vector Machine",
        "Extreme Gradient Boost",
    ]
    measurements = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score]
    measurement_names = ["Accuracy", "F1 score", "Precision", "Recall", "AUC"]

    for i in range(len(models)):
        score_dict["Model"].append(model_names[i])
        for j in range(len(measurements)):
            score_dict[measurement_names[j]].append(
                measurements[j](y_test, models[i].predict(X_test))
            )

    score_df = pd.DataFrame(score_dict).set_index("Model")

    return score_df, model_score, model_cm, model_feature, dt


# Given df, X, y holdout, train, test split will be gained
def df_split(df, balancing='None', holdout_ratio=0.25,
             sampling_random_state=1, holdout_random_state=1,
             split_random_state=1, split_ratio=0.2, remove_duplicates=False):
    """Generate X_train, X_test, X_holdout, y_train, y_test, y_holdout for machine learning
    :param remove_duplicates: boolean
    :param df: DataFrame that has been engineered, imputed, and scaled
    :param balancing: string, 'None', 'Over', 'Under'
    :param holdout_ratio: float
    :param sampling_random_state: integer
    :param holdout_random_state: integer
    :param split_random_state: integer
    :param split_ratio: float
    :return: Split dataframe ready to be trained, tested

    --------------------------------------------------------
    Examples:
    X_train, X_test, X_holdout, y_train, y_test, y_holdout = df_split(df=df)
    """

    # Holdout for final model test
    X_all = df.drop(columns='y')
    y_all = df["y"]

    X, X_holdout, y, y_holdout = train_test_split(
        X_all, y_all, test_size=holdout_ratio, stratify=y_all, random_state=holdout_random_state
    )

    df_holdout = pd.concat([X_holdout, y_holdout], axis=1)
    df_rest = pd.concat([X, y], axis=1)

    # If the features (which you have selected) become lesser, duplicates would appear given the same sample size
    print("The number of repeated (redundant) samples in the holdout dataset given your selection of features for "
          "modelling are", df_holdout.shape[0], "-", df_holdout.drop_duplicates().shape[0])

    print("The number of repeated (redundant) samples in the train, test/eval dataset (rest) given your selection "
          "of features for modelling are", df_rest.shape[0], "-", df_rest.drop_duplicates().shape[0])

    print("You can drop duplicates with 'remove_duplicates=True' parameter")

    if remove_duplicates:
        df_holdout.drop_duplicates(inplace=True)
        df_rest.drop_duplicates(inplace=True)
        X_holdout = df_holdout.drop(columns='y')
        y_holdout = df_holdout['y']
        X = df_rest.drop(columns='y')
        y = df_rest['y']

    # Begin split
    if balancing == 'None':
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=split_ratio, stratify=y, random_state=split_random_state
        )
    elif balancing == 'Under':
        undersampler = RandomUnderSampler(sampling_strategy="majority", random_state=sampling_random_state)
        X_under, y_under = undersampler.fit_resample(X, y)
        X_train, X_test, y_train, y_test = train_test_split(
            X_under, y_under, test_size=split_ratio, stratify=y_under, random_state=split_random_state
        )
    elif balancing == 'Over':
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=split_ratio, stratify=y, random_state=split_random_state
        )
        oversampler = RandomOverSampler(sampling_strategy='minority', random_state=sampling_random_state)
        X_train, y_train = oversampler.fit_resample(X_train, y_train)
        X_test, y_test = oversampler.fit_resample(X_test, y_test)

    else:
        print("Wrong input of 'balancing', please see docstring")
        return

    return X_train, X_test, X_holdout, y_train, y_test, y_holdout


# Simplified uni-variate imputation method with basic common sense and minmaxscaler for easy operation
def df_simpleimputer_scaled(df):
    """ A new df will be returned with NaN filled with median value for each column
    :param df: Dataframe to be imputed
    :return: df Dataframe that has been imputed
    """
    df_new = pd.DataFrame()
    for col in df.columns[~df.columns.str.contains("^y$")]:
        df_new[col] = df[col].fillna(df[col].median())
    df_new = pd.DataFrame(
        MinMaxScaler().fit_transform(df_new), columns=df_new.columns, index=df_new.index,
    )
    df_new = pd.concat([df_new, df["y"]], axis=1)
    return df_new


# Perform statistical testing of all existing features using Chi-square and T-test for two populations (asthma,
# no asthma)
def df_feature_stats(df_child, target='y'):
    """df_child is cleaned dataframe with y as target of testing, categorical feature is defined as no more than 10 unqiue values (frequencies), numeric features are the rest.
    """
    numeric_col = []
    categorical_col = []
    for i in df_child.columns:
        if df_child[i].nunique() > 10:
            numeric_col.append(i)
        elif i != target:
            categorical_col.append(i)

    # Create numeric stats dataframe
    numeric_dict = {}
    print("diff represents the mean value of non-asthma group minus the mean value of asthma group.")
    for i in numeric_col:
        # Calculate difference of mean value - [Asthma Group - No_Asthma Group]
        diff_mean = np.mean(df_child[df_child[target] == 0][i]) - np.mean(df_child[df_child[target] == 1][i])

        # Perform independent t-test for two populations
        temp = pg.ttest(
            df_child[df_child[target] == 0][i],
            df_child[df_child[target] == 1][i],
            paired=False,
            alternative="two-sided",
            correction="auto",
        ).rename(index={"T-test": i})

        # Insert extra information
        temp.insert(4, "diff", diff_mean)

        # Store the result
        numeric_dict[i] = temp

    numeric_feature_stats = pd.concat([v for k, v in numeric_dict.items()], axis=0).sort_values('p-val')

    # Create categorical stats dataframe
    categorical_dict = {}
    for i in categorical_col:
        # Perform Chi Square Test for categorical features for two populations (with or without asthma)
        expected, observed, stats = pg.chi2_independence(
            df_child,
            x=i,
            y=target,
            correction=False,
        )
        stats.rename(index={0: i}, inplace=True)

        # Store 'Pearson' Chi-Square Result
        categorical_dict[i] = stats[:1]

    categorical_feature_stats = pd.concat([v for k, v in categorical_dict.items()], axis=0).sort_values('pval')

    return numeric_feature_stats, categorical_feature_stats


# Perform alluvial analysis for target variables (visualization)
def target_alluvial_analysis(df_child,
                             target_list=["Respiratory_Problems_Birth", "Recurrent_Wheeze_1y", "Asthma_Diagnosis_3yCLA",
                                          "Asthma_Diagnosis_5yCLA"], node_label=[
            "No Respiratory Problems at Birth",
            "Respiratory Problems at Birth",
            "No Wheeze Report at 1y",
            "Wheeze Report at 1y",
            "No Asthma at 3y",
            "Asthma at 3y",
            "Possible Asthma at 3y",
            "No Asthma at 5y",
            "Asthma at 5y",
            "Possible Asthma at 5y",
        ], flow_to_display=3):
    """
    Perform alluvial analysis for asthma targets. Available target include:['Triggered_Asthma_5yCLA', 'Triggered_Asthma_3yCLA', 'Wheeze_3yCLA', 'Wheeze_5yCLA', 'Asthma_Diagnosis_5yCLA ', 'Asthma_Diagnosis_3yCLA ',
    'Respiratory_Problems_Birth', 'Wheeze_3m', 'Noncold_Wheeze_3m', 'Wheeze_1y', 'Wheeze_6m', 'Recurrent_Wheeze_1y', 'Recurrent_Wheeze_3y', 'Recurrent_Wheeze_5y', 'Wheeze_Traj_Type']

    Currently, the supported target_list must be 4 potential target variables, with the first two in binary format (the constraint is due to manual coloring of flow and position)
    with the later two are three year diagnosis and five year diagnosis.

    Must "import plotly.graph_objects as go" Before apply this function

    Parameters:
    --------------
    target_list: list, must be in time order. with the earliest being the first and latest being the last

    Returns:
    --------------
    Three steps of flowing as DataFrame between four time points

    """
    # Add one column for easy counting
    df_child["Count"] = 1

    # Flow One: Birth - 1 y
    # Calculate aggregate of the combinations of two
    flow_1 = (
        df_child.groupby(target_list[:2])
            .sum()
            .reset_index()[target_list[:2] + ["Count"]]
    )

    # Calcuate Percentage
    flow_1["Percentage"] = round(
        flow_1.groupby([target_list[0]]).apply(lambda x: x / x.sum()).Count * 100, 1,
    )

    # Re-label according to plotly accepted format
    flow_1[target_list[1]] = flow_1[target_list[1]] + flow_1[target_list[0]].nunique()

    # Flow Two:
    # Aggregate
    flow_2 = (
        df_child.groupby(target_list[1:3])
            .sum()
            .reset_index()[target_list[1:3] + ["Count"]]
    )

    # Relabel
    flow_2[target_list[1]] = flow_2[target_list[1]] + flow_1[target_list[0]].nunique()

    flow_2[target_list[2]] = (
            flow_2[target_list[2]]
            + flow_1[target_list[0]].nunique()
            + flow_2[target_list[1]].nunique()
    )

    # Percentage
    flow_2["Percentage"] = round(
        flow_2.groupby([target_list[1]]).apply(lambda x: x / x.sum()).Count * 100, 1
    )

    # Flow three:
    flow_3 = (
        df_child.groupby(target_list[2:4])
            .sum()
            .reset_index()[target_list[2:4] + ["Count"]]
    )

    # Relabel
    flow_3[target_list[2]] = (
            flow_3[target_list[2]]
            + flow_1[target_list[0]].nunique()
            + flow_2[target_list[1]].nunique()
    )

    flow_3[target_list[3]] = (
            flow_3[target_list[3]]
            + flow_1[target_list[0]].nunique()
            + flow_2[target_list[1]].nunique()
            + flow_3[target_list[2]].nunique()
    )

    # Percentage
    flow_3["Percentage"] = round(
        flow_3.groupby([target_list[2]]).apply(lambda x: x / x.sum()).Count * 100, 1,
    )

    df_child.drop(columns='Count', inplace=True)

    # Artist
    link_color = [
        "rgba(242, 116, 32, 1)",
        "rgba(242, 116, 32, 1)",
        "rgba(73, 148, 206, 1)",
        "rgba(73, 148, 206, 1)",
        "rgba(250, 188, 19, 0.5)",
        "rgba(250, 188, 19, 0.5)",
        "rgba(250, 188, 19, 0.5)",
        "rgba(127, 194, 65, 0.5)",
        "rgba(127, 194, 65, 0.5)",
        "rgba(127, 194, 65, 0.5)",
        "rgba(253, 227, 202, 20.5)",
        "rgba(253, 227, 202, 20.5)",
        "rgba(253, 227, 202, 20.5)",
        "rgba(127, 94, 165, 20)",
        "rgba(127, 94, 165, 20)",
        "rgba(127, 94, 165, 20)",
        "rgba(21, 211, 211, 0.5)",
        "rgba(21, 211, 211, 0.5)",
        "rgba(21, 211, 211, 0.5)",
    ]

    # node_label = [
    #     "No Respiratory Problems at Birth",
    #     "Respiratory Problems at Birth",
    #     "No Wheeze Report at 1y",
    #     "Wheeze Report at 1y",
    #     "No Asthma at 3y",
    #     "Asthma at 3y",
    #     "Possible Asthma at 3y",
    #     "No Asthma at 5y",
    #     "Asthma at 5y",
    #     "Possible Asthma at 5y",
    # ]

    node_color = [
        "#F27420",
        "#4994CE",
        "#FABC13",
        "#7FC241",
        "#D3D3D3",
        "#8A5988",
        "#449E9E",
        "#A0693D",
        "#AD9A24",
        "#E8F5C0",
    ]

    # Drawing data
    if flow_to_display == 3:
        source = pd.concat(
            [flow_1.iloc[:, 0], flow_2.iloc[:, 0], flow_3.iloc[:, 0]]
        ).reset_index(drop=True)

        target = pd.concat(
            [flow_1.iloc[:, 1], flow_2.iloc[:, 1], flow_3.iloc[:, 1]]
        ).reset_index(drop=True)

        volumn = pd.concat(
            [flow_1.iloc[:, 2], flow_2.iloc[:, 2], flow_3.iloc[:, 2]]
        ).reset_index(drop=True)

        label_number = pd.concat(
            [flow_1.iloc[:, 3], flow_2.iloc[:, 3], flow_3.iloc[:, 3]]
        ).reset_index(drop=True)


    elif flow_to_display == 2:
        source = pd.concat(
            [flow_1.iloc[:, 0], flow_2.iloc[:, 0]]
        ).reset_index(drop=True)

        target = pd.concat(
            [flow_1.iloc[:, 1], flow_2.iloc[:, 1]]
        ).reset_index(drop=True)

        volumn = pd.concat(
            [flow_1.iloc[:, 2], flow_2.iloc[:, 2]]
        ).reset_index(drop=True)

        label_number = pd.concat(
            [flow_1.iloc[:, 3], flow_2.iloc[:, 3]]
        ).reset_index(drop=True)

        link_color = link_color[:-9]

        node_color = node_color[:-3]


    elif flow_to_display == 1:
        source = flow_1.iloc[:, 0].reset_index(drop=True)

        target = flow_1.iloc[:, 1].reset_index(drop=True)

        volumn = flow_1.iloc[:, 2].reset_index(drop=True)

        label_number = flow_1.iloc[:, 3].reset_index(drop=True)

        link_color = link_color[:-15]

        node_color = node_color[:-6]

    else:
        print("Unacceptable flow number, only 1-3 are accepted for now")

    label = [
        "The percentage that flows to target is "
        + str(j)
        + "% with the number of "
        + str(i)
        for i, j in zip(volumn, label_number)
    ]

    # Plotting
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=10,
                    line=dict(color="black", width=0.5),
                    label=node_label,
                    color=node_color,
                ),
                link=dict(
                    source=source,  # indices correspond to labels, eg A1, A2, A1, B1, ...
                    target=target,
                    value=volumn,
                    label=label,
                    color=link_color,
                ),
            )
        ]
    )

    layout = dict(
        title="CHILD Study Asthma Progression", height=772, font=dict(size=12),
    )

    fig.update_layout(layout)

    return fig
