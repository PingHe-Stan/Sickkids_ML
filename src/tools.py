__author__ = 'Stan He@Sickkids.ca'
__date__ = '2021-10-21'
"""Gadgets for various tasks 
"""
import numpy as np
import pandas as pd
import random
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

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
        f"Loading {child_ethnicity_path, child_data_path, child_breastfeeding_path, child_ethnicity_path}, and merging")
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
    print(f"The dataframe merged with more information is saved to {CHILD_RAW_DIR} with name of CHILD_with_addon.xlsx")

    return df_child


# Define your dataset based on target variable, which cannot be controlled in ML pipeline
def target_selector(df, target_name="Asthma_Diagnosis_5yCLA", target_mapping={2: 1}, include_dust=False):
    """Define your target variable, the mapping schemes, and whether to include dust sampling data for modelling.

    Parameters:
    -----------------
    df: DataFrame to choose target from

    target_name : str,  default 'Asthma_Diagnosis_5yCLA'
        A string that is selected from TARGET_VAR_REPO list to control the entire dataframe to be processed

    target_mapping : dictionary, default {2:1} that is suggesting possible asthma(2) will be treated as asthma(0)
        A dictionary used to specify how to map the "possible", or, or other variables

    include_dust : boolean, default is False
        A boolean for certain attribute selection that will drastically reduce sample size for analysis

    Returns:
    ----------------
    Targeted df, in which target will be renamed to 'y'
    """

    # df_child variables sanity check
    print(f"The dimension of original dataframe for process is {df.shape}")
    print("Number of categorical features is: ", len(CAT_VARS_REPO))
    print("Number of numeric features is: ", len(NUM_VARS_REPO))
    print("Number of target variables is: ", len(TARGET_VAR_REPO))
    print("Number of dropped features is: ", len(FEA_TO_DROP))
    print("The difference of features of targeted and original dataframe is :{0}".format((set(df.columns.values) - set(
        TARGET_VAR_REPO + NUM_VARS_REPO + CAT_VARS_REPO + FEA_TO_DROP))))
    print(f"The number of missing value for target label is: {df[target_name].isna().sum()}")

    # Print selectable target variables
    print("------------------------------------------------------")
    print("Note: Target variable can be one of: \n", TARGET_VAR_REPO)
    print("------------------------------------------------------")
    print("****Target variable will be renamed to y for easy access.**** \n")

    y = df[target_name].copy().replace(target_mapping)

    if include_dust:
        X = df[NUM_VARS_REPO + CAT_VARS_REPO + ["Subject_Number"]].copy()
        df_temp = pd.concat([X, y], axis=1)

        df_Xy = df_temp.dropna(subset=SUB_NUM_HOMEDUST + [target_name]).reset_index(
            drop=True
        )

        df_Xy.set_index(
            "Subject_Number", inplace=True
        )  # Use Index as an extra information

        X_return = df_Xy.drop(columns=[target_name])
        y_return = df_Xy[target_name]

        return df_Xy.rename(columns={target_name: "y"}), X_return, y_return

    else:
        X = (
            df[NUM_VARS_REPO + CAT_VARS_REPO + ["Subject_Number"]]
                .copy()
                .drop(columns=SUB_NUM_HOMEDUST)
        )
        df_temp = pd.concat([X, y], axis=1)

        df_Xy = df_temp.dropna(subset=[target_name]).reset_index(drop=True)

        df_Xy.set_index(
            "Subject_Number", inplace=True
        )  # Use Index as an extra information

        return df_Xy.rename(columns={target_name: "y"})


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
                X[cols_to_inspect].mean(),
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
                1: "Mean_Value",
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
