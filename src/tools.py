__author__ = 'Stan He@Sickkids.ca'
__date__ = ['2021-10-21', '2021-10-26', '2021-10-29', '2021-11-01', '2021-11-08']

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
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    #    roc_curve,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    plot_confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.inspection import permutation_importance


# Load more variables to existing xlsx
def load_child_with_more(
        child_data_path=CHILD_DATA_PATH,
        child_breastfeeding_path=CHILD_BREASTFEEDING_PATH,
        child_wheezephenotype_path=CHILD_WHEEZEPHENOTYPE_PATH,
        child_ethnicity_path=CHILD_ETHNICITY_PATH,
):
    """Load CHILD data with add-on features from additional files
    """

    print(
        f"Loading {child_ethnicity_path, child_data_path, child_breastfeeding_path, child_wheezephenotype_path}, and merging")
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

    df_child = pd.merge(
        pd.merge(df_child, df_bf, on="Subject_Number", how="left"),
        df_wp,
        on="Subject_Number",
        how="left",
    )

    df_child = pd.merge(df_child, df_ethnicity, on="Subject_Number", how="left")
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
        "12m": "12m|1y",
        "18m": "18m|BF_implied",
        "24m": "24m|2y",
        "36m": "36m|3y",
        "48m": "48m|4y",
        "60m": "60m|5y|Traj_Type",
        "1_9m_2hy": "1m$|9m|2hy"
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


# An drastic updating of the version of ml_run() which has less, cleaner code and more flexibility
def df_ml_run(
        X_train, X_test, y_train, y_test, scoring_func=f1_score, importance_scoring="f1"
):
    """
    Fast run of a myriad of models and return three very informative dataframes (confusion matrix dataframe,  model performance dataframe
    and colored feature importance dataframe) and one decision tree for visualization

    :param scoring_func: a predefined function, default: f1_score
        model_scoring has to be one of accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
        since it must have the same name of sklearn.metrics functions

    :param importance_scoring: string, default: f1
        importance_scoring has to be one of 'accuracy', 'f1', 'precision', 'recall', 'roc_auc',
        since it must be recognizable as a parameter to be put into permutation importance

    :return: Confusion matrix dataframe, model performance, feature importance

    ------------------------
    Examples:
    ml_result_tuple = df_ml_run(X_train, X_test, y_train, y_test)
    """
    # A dictionary that will store the performance score of models to evaluate feature importance
    model_score = {}

    # A dictionary that will store the confusion matrix results as string to easy comparision
    model_cm = {}

    # A dictionary that will store the feature importance for different models
    model_feature = {}

    # Initializing the model using desired abbreviated names as model instance to train and test
    xgb = XGBClassifier(learning_rate=0.01, n_estimators=250)
    dt = DecisionTreeClassifier(criterion="entropy", random_state=0, max_depth=6)
    rf = RandomForestClassifier(n_estimators=200, random_state=0, max_depth=5)
    knn = KNeighborsClassifier(n_neighbors=5)
    svc = SVC(kernel="rbf", C=6)
    nb = GaussianNB()
    lr = LogisticRegression()

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

    permutation_importance_list = [(knn, "knn"), (svc, "svc"), (nb, "nb")]
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
            f"The performance score of {model_name[key]}: {model_score[key] * 100} \n"
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
    models = [lr, nb, rf, knn, dt, svc, xgb]
    models_label = [
        "Logistic Regression",
        "Naive Bayes",
        "Random Forest",
        "K-Nearest Neighbours",
        "Decision Tree",
        "Support Vector Machine",
        "Extreme Gradient Boost",
    ]
    measurements = [
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    ]
    measurement_names = ["Accuracy", "F1 score", "Precision", "Recall", "AUC"]
    for i in range(len(models)):
        score_dict["Model"].append(models_label[i])
        for j in range(len(measurements)):
            score_dict[measurement_names[j]].append(
                measurements[j](y_test, models[i].predict(X_test))
            )
    score_df = pd.DataFrame(score_dict).set_index("Model")

    #    model_score_returned = score_df.style.background_gradient(cmap="Greens")
    model_score_returned = score_df
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

    return model_score_returned, feature_res_returned, model_cm_df, dt


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
        elif i != "y":
            categorical_col.append(i)

    # Create numeric stats dataframe
    numeric_dict = {}
    for i in numeric_col:
        # Calculate difference of mean value - [Asthma Group - No_Asthma Group]
        diff_mean = np.mean(df_child[df_child[target] == 1][i]) - np.mean(df_child[df_child[target] == 0][i])

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
def target_alluvial_analysis(df_child,target_list=["Respiratory_Problems_Birth", "Recurrent_Wheeze_1y", "Asthma_Diagnosis_3yCLA",
                     "Asthma_Diagnosis_5yCLA"]):
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

    # Drawing data
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
    label = [
        "The percentage that flows to target is "
        + str(j)
        + "% with the number of "
        + str(i)
        for i, j in zip(volumn, label_number)
    ]

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

    node_label = [
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
    ]

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
