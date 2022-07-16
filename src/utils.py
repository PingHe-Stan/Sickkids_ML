__author__ = 'Stan He@Sickkids.ca'
__contact__ = 'stan.he@sickkids.ca'
__date__ = ['2021-10-15', '2022-02-04', '2022-07-11']
"""Preprocessing pipeline components
"""

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
import sklearn.neighbors._base
import sys

from IPython.display import display as dispdf

sys.modules["sklearn.neighbors.base"] = (
    sklearn.neighbors._base)  # To conquer the version naming conflict for use of missingpy
from sklearn.impute import SimpleImputer, KNNImputer
from missingpy import MissForest  # For Imputation using Random Forest

from conf import *  # Import Global Variables


# Transformers - Apgar Score
class ApgarTransformer(BaseEstimator, TransformerMixin):
    """
    Perform transformer to enhance the meaningful extraction of subjective ranking system.

    Parameters:
    --------------
    engineer_type: string, default = 'Ordinal'
        (1) "Ordinal" Categorize Apgar Score into Three groups and be treated as ordinal categories, with missing value with np.nan
            - 'Critical_Low': - 0,1,2,3
            - 'Below_Normal': - 4,5,6
            - 'Normal': - 7+
            - np.nan: - np.nan
        (2) Numeric - "Log1p", note that there are more at higher (7,8,9) for apgar score, such operation modified(compressed) the distribution scale.
        (3) Numeric - "None", as it is.

    Returns:
    ---------------
    Transformed X with the specified engineering type

    """

    def __init__(self, engineer_type="None"):
        super().__init__()
        self.engineer_type = engineer_type

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if "Apgar_Score_1min" in X.columns:
            if self.engineer_type == "Ordinal":
                X["Apgar_Score_1min"] = pd.cut(
                    X["Apgar_Score_1min"],
                    [-1, 3.5, 6.5, 11],
                    labels=["Critical_Low", "Below_Normal", "Normal"],
                )
            elif self.engineer_type == "Log1p":
                X["Apgar_Score_1min"] = np.log1p(X["Apgar_Score_1min"])
            elif self.engineer_type == "None":
                pass
            else:
                print(
                    "The engineer_type you've entered is not valid. No engineering will be performed"
                )
                print("Valid inputs include 'Ordinal','Log1p','None'")
        if "Apgar_Score_5min" in X.columns:
            if self.engineer_type == "Ordinal":
                X["Apgar_Score_5min"] = pd.cut(
                    X["Apgar_Score_5min"],
                    [-1, 3.5, 6.5, 11],
                    labels=["Critical_Low", "Below_Normal", "Normal"],
                )
            elif self.engineer_type == "Log1p":
                X["Apgar_Score_5min"] = np.log1p(X["Apgar_Score_5min"])
            elif self.engineer_type == "None":
                pass
            else:
                print(
                    "The engineer_type you've entered is not valid. No engineering will be performed"
                )
                print("Valid inputs include 'Ordinal','Log1p','None'")
        return X


# Feature Engineering on features of Birth Profile
class BirthTransformer(BaseEstimator, TransformerMixin):
    """
    Categorize Mode_of_delivery into Two or Original groups
    - 'birth_mode_delivery': - Binarized, Triplefied, or original categorical value
    Devise numeric feature to signal the severity of the pregnancy condition and birth situations
    - 'Prenatal_Mother_Condition': - Numeric, 0: 'Nausea','Bleeding', 'None', 1: Existing OR 0,1(of little consequences),2(of impact)
    - 'First_10min_Measure': - Numeric, 0: 'None', 1: 'Suction, Oxyg, Ventilation', 2: 'Mask', 4:'Intubation' OR 2: Intubation, 1: Mask, 0: Others

    Parameters:
    -----------------------
    birth_mode_delivery: int, or string, default 2
        Whether binarize the birth mode into Csection or not, or triple them into vaginal, c-with labor, c-without labor.
        If the input is not 2 or 3, then detailed delivery mode will be used

    binary_pregnancy_conditions: boolean, default False
        If true, use 0 to represent common pregnancy conditions: Bleeding,Nausea, such as use 1 to represent other medical conditions
        If false, use 1 to represent "non-impactful" conditions (such as Infections, Hypertension, Hypotension), use 2 to represent impactful conditions such as diabetes, cardiac disorder(based on reading), and 1.5 to represent others

    signal_suction: boolean, default True
        If true, Suction as 0.5, Ventilation, and Oxygen are represented with 1, mask as 2, intubation as 4
        If false, they will be represented as 0, Mask as 1, intubation as 2

    Return
    --------------------------
    Transformed X with engineered birth features that are ready to be processed as categorical or numeric values

    """

    def __init__(
            self,
            birth_mode_delivery=2,
            binary_pregnancy_conditions=False,
            signal_suction=False,
    ):
        super().__init__()
        self.birth_mode_delivery = birth_mode_delivery
        self.binary_pregnancy_conditions = binary_pregnancy_conditions
        self.signal_suction = signal_suction

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        bimode_birth_dict = {
            1.0: "Vaginal",
            2.0: "Vaginal",
            3.0: "Vaginal",
            4.0: "Vaginal",
            6.0: "Caesarean",
            7.0: "Caesarean",
            8.0: "Caesarean",
            9.0: "Caesarean",
            11.0: "Vaginal",
        }

        trimode_birth_dict = {
            1.0: "Vaginal",
            2.0: "Vaginal",
            3.0: "Vaginal",
            4.0: "Vaginal",
            6.0: "Caesarean_without_Labor",
            7.0: "Caesarean_with_Labor",
            8.0: "Caesarean_with_Labor",
            9.0: "Caesarean_without_Labor",
            11.0: "Vaginal",
        }

        multimode_birth_dict = {
            1.0: "Vaginal_Unassisted",
            2.0: "Vaginal_Assisted",
            3.0: "Vaginal_Vacuum_Extracted",
            4.0: "Vaginal_Breech",
            6.0: "Elective_Caesarean",
            7.0: "Caesarean_during_Labor",
            8.0: "Emergency_Caesarean_with_Labor",
            9.0: "Emergency_Caesarean_without_Labor",
            11.0: "Vaginal_with_Episiotomy",
        }

        binary_conditions_dict = {
            "Gestational Diabetes": 1,  # No. of "Gestational Diabetes": 107
            "Cardiac Disorder": 1,  # No. of Cardiac Disorder 9,
            "Other": 1,  # Total No. of other 699
            "Hypertension": 1,  # No. of all hypertension: 130
            "Hypotension": 1,  # No. = 5
            "Infections": 1,  # No. = 84
            "Bleeding": 0,  # No. = 174
            "Nausea": 0,  # No. = 551
            "None": 0,  # No. 1206
        }

        triple_conditions_dict = {
            "Gestational Diabetes": 2,  # No. of "Gestational Diabetes": 107
            "Cardiac Disorder": 2,  # No. of Cardiac Disorder 9,
            "Other": 1,  # Total No. of other 699
            "Hypertension": 1,  # No. of all hypertension: 130
            "Hypotension": 1,  # No. = 5
            "Infections": 1,  # No. = 84
            "Bleeding": 0,  # No. = 174
            "Nausea": 0,  # No. = 551
            "None": 0,  # No. 1206
        }

        first10min_suction_dict = {
            "Intubation": 4,  # No. of intubation is 39
            "Mask": 2,  # No. of Mask is 65
            "Positive Pressure Ventilation": 1,  # No. of Positive Ventilation 81
            "Free Flow Oxygen": 1,  # No. of Free Flow Oxygen 87
            "Perineum suction": 1,  # No. of Perineum suction 30
            "Suction": 0.5,  # No. of Suction 415
            "None": 0,
        }

        first10min_nosuction_dict = {
            "Intubation": 2,  # No. of intubation is 39
            "Mask": 1,  # No. of Mask is 65
            "Positive Pressure Ventilation": 0,  # No. of Positive Ventilation 81
            "Free Flow Oxygen": 0,  # No. of Free Flow Oxygen 87
            "Perineum suction": 0,  # No. of Perineum suction 30
            "Suction": 0,  # No. of Suction 415
            "None": 0,
        }

        birth_mode_engioverview = pd.concat(
            [
                pd.DataFrame.from_dict(
                    multimode_birth_dict, orient="index", columns=["Multimode_birth_engi"]
                ),
                pd.DataFrame.from_dict(
                    bimode_birth_dict, orient="index", columns=["Bimode_birth_engi"]
                ),
                pd.DataFrame.from_dict(
                    trimode_birth_dict, orient="index", columns=["Triple_mode_birth_engi"]
                ),
            ],
            axis=1,
        )

        f10min_measure_engioverview = pd.concat(
            [
                pd.DataFrame.from_dict(
                    first10min_suction_dict, orient="index", columns=["First10min_suction_engi"]
                ),
                pd.DataFrame.from_dict(
                    first10min_nosuction_dict,
                    orient="index",
                    columns=["First10min_nosuction_engi"],
                ),
            ],
            axis=1,
        )

        pregn_conditions_engioverview = pd.concat(
            [
                pd.DataFrame.from_dict(
                    binary_conditions_dict, orient="index", columns=["Binary_conditions_engi"]
                ),
                pd.DataFrame.from_dict(
                    triple_conditions_dict, orient="index", columns=["Triple_conditions_engi"]
                ),
            ],
            axis=1,
        )

        print("Please see the following dataframe for engineer details:")

        dispdf(birth_mode_engioverview)
        dispdf(f10min_measure_engioverview)
        dispdf(pregn_conditions_engioverview)

        if 'Mode_of_delivery' in X.columns:
            if self.birth_mode_delivery == 2:
                X["Mode_of_delivery"] = X["Mode_of_delivery"].replace(bimode_birth_dict)
            elif self.birth_mode_delivery == 3:
                X["Mode_of_delivery"] = X["Mode_of_delivery"].replace(trimode_birth_dict)
            else:
                X["Mode_of_delivery"] = X["Mode_of_delivery"].replace(multimode_birth_dict)

        if "Prenatal_Mother_Condition" in X.columns:
            if self.binary_pregnancy_conditions:
                X["Prenatal_Mother_Condition"] = X["Prenatal_Mother_Condition"].replace(
                    binary_conditions_dict,
                    regex=True,
                    # Regex is True: Replace happens if there is a match (str.contains). False: Only exact match.
                )

            else:
                X["Prenatal_Mother_Condition"] = X["Prenatal_Mother_Condition"].replace(
                    triple_conditions_dict,
                    regex=True,
                    # Note: Regex is True: Replace happens if there is a match (str.contains). False: Only exact match.
                )

        if "First_10min_Measure" in X.columns:
            if self.signal_suction:
                X["First_10min_Measure"] = X["First_10min_Measure"].replace(
                    first10min_suction_dict,
                    regex=True,
                    # Regex is True: Replace happens if there is a match (str.contains). False: Only exact match.
                )
            else:
                X["First_10min_Measure"] = X["First_10min_Measure"].replace(
                    first10min_nosuction_dict,
                    regex=True,
                    # Regex is True: Replace happens if there is a match (str.contains). False: Only exact match.
                )

        return X


# Feature Engineering -  natural logarithm of one plus the input array to compress the distribution of SUB_NUM_LOG1P.
class Log1pTransformer(BaseEstimator, TransformerMixin):
    """
    Apply natural logarithm of one plus the input array to compress the distribution of SUB_NUM_LOG1P.

    Parameters for Log1pTransformer:
    ----------
    cols: list of columns to be log casted/mapped, default SUB_NUM_LOG1P

    Return:
    ---------
    Logified Numeric Values For Specified Columns
    """

    def __init__(self, cols=[
        "No_of_Pregnancy",
        "Stay_Duration_Hospital",
        "BF_Implied_Duration",
    ]):
        super().__init__()
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if len(self.cols) != 0:
            for coln in self.cols:
                X[coln] = np.log1p(X[coln])
        return X


# Feature Engineering the Respiratory History (trajectory) to be machine digestable format
class RespiratoryTransformer(BaseEstimator, TransformerMixin):
    """
    Convert Respiratory Information to be machine digestable format

    Parameters for RespiratoryTransformer:
    ----------
    first_18m_divide: boolean, default False
        Decide whether or not to cut the timeline into half so that the Respiratory_Report_Months could be used

    minimal_value_presence: integer, default 2
        When converting to numeric format, due to the NaN presence, a threshold of value presence will be decided

    Return:
    ---------
    Machine Digestable Respiratory Information
    """

    def __init__(self, first_18m_divide=False, minimal_value_presence=1):
        super().__init__()
        self.first_18m_divide = first_18m_divide
        self.minimal_value_presence = minimal_value_presence

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        # Parameters that could be later added to pipeline in the initialization
        thresh_point = 24  # For 18m, an extra of half a year is allowed for the subject to fill in the questionnaire

        # 'Mild':0.1,'Moderate':0.3,'Severe':0.6
        severity_dict = {
            "Unknown": 0,
            "No": 0,
            "Mild": 0.1,
            "Moderate": 0.3,
            "Severe": 0.6,
        }

        if self.first_18m_divide:

            # ---------------------------------------------------------------------
            # Divide the Respiratory_Infections & Severity_of_Respiratoryinfections into two respectively
            # ---------------------------------------------------------------------
            nan = (
                -1
            )  # nan in list needs to be defined first as a number for number comparison to locate the position between months

            position_list = (
                []
            )  # Record the index position for the earlier, later division

            frequency_earlier_report = (
                []
            )  # Record the ealier part of the "Respiratory_Infections" columns
            frequency_later_report = []
            severity_earlier_report = []
            severity_later_report = []
            for i, j, k in zip(
                    X["Respiratory_Report_Months"],
                    X["Respiratory_Infections"],
                    X["Severity_of_Respiratoryinfections"],
            ):
                for m, n in enumerate(eval(i)):
                    if n > thresh_point:
                        position_list.append(m)
                        nan = "Unknown"  # Unknown will be used to replace nan for the severity mapping to work
                        frequency_earlier_report.append(eval(j)[:m])
                        frequency_later_report.append(eval(j)[m:])
                        severity_earlier_report.append(eval(k)[:m])
                        severity_later_report.append(eval(k)[m:])
                        nan = -1  # Restored for month calculation
                        break

                else:
                    position_list.append(
                        9
                    )  # If all reported months are less then thresh, then all will be labelled as earlier report
                    nan = "Unknown"  # See above
                    frequency_earlier_report.append(eval(j)[:9])
                    frequency_later_report.append(eval(j)[9:])
                    severity_earlier_report.append(eval(k)[:9])
                    severity_later_report.append(eval(k)[9:])
                    nan = -1  # See above

            X["RIfrequency_earlier_12m"] = pd.Series(
                frequency_earlier_report, index=X.index
            )
            X["RIfrequency_later_36m"] = pd.Series(
                frequency_later_report, index=X.index
            )
            X["RIseverity_earlier_12m"] = pd.Series(
                severity_earlier_report, index=X.index
            )
            X["RIseverity_later_36m"] = pd.Series(
                severity_later_report, index=X.index
            )

            # ---------------------------------------------------------------------
            # Four new columns have been created, Following will be the conversion.
            # ---------------------------------------------------------------------

            # Empty the predefined set of lists for numeric conversion
            frequency_earlier_report = []
            frequency_later_report = []
            severity_earlier_report = []
            severity_later_report = []

            for i, j in zip(
                    X["RIfrequency_earlier_12m"], X["RIseverity_earlier_12m"]
            ):
                if (
                        eval(str(i)).count("Unknown")
                        > len(eval(str(i))) - self.minimal_value_presence
                ):
                    frequency_earlier_report.append(np.nan)
                    severity_earlier_report.append(np.nan)

                else:
                    frequency_earlier_report.append(eval(str(i)).count("LRTI"))
                    severity_earlier_report.append(
                        round(pd.Series(eval(str(j))).map(severity_dict).sum(), 2)
                    )

            for i, j in zip(
                    X["RIfrequency_later_36m"], X["RIseverity_later_36m"]
            ):
                if len(eval(str(i))) < self.minimal_value_presence:
                    frequency_later_report.append(np.nan)
                    severity_later_report.append(np.nan)
                else:
                    frequency_later_report.append(eval(str(i)).count("LRTI"))
                    severity_later_report.append(
                        round(
                            pd.Series(eval(str(j))).map(severity_dict).sum(), 2
                        )  # Round to make the exact match, otherwise, float type might interfere
                    )

            # Replaced with processed data
            #             X["RIfrequency_earlier_12m"] = pd.Series(frequency_earlier_report)
            #             X["RIseverity_earlier_12m"] = pd.Series(severity_earlier_report)
            #             X["RIfrequency_later_36m"] = pd.Series(frequency_later_report)
            #             X["RIseverity_later_36m"] = pd.Series(severity_later_report)

            X["RIfrequency_earlier_12m"] = pd.Series(
                frequency_earlier_report, index=X.index
            )
            X["RIfrequency_later_36m"] = pd.Series(
                frequency_later_report, index=X.index
            )
            X["RIseverity_earlier_12m"] = pd.Series(
                severity_earlier_report, index=X.index
            )
            X["RIseverity_later_36m"] = pd.Series(
                severity_later_report, index=X.index
            )

            # The original SUB_CAT_RESP will be dropped and four new columns will be created
            print(
                "The output columns will be: RIfrequency_earlier_12m, RIseverity_earlier_12m, RIfrequency_later_36m, RIseverity_later_36m"
            )

            X.drop(
                columns=[
                    "Respiratory_Report_Months",
                    "Respiratory_Infections",
                    "Severity_of_Respiratoryinfections",
                ],
                inplace=True,
            )

            return X

        else:
            nan = "Unknown"

            # Engineering Schemes for Respirotary Infections
            resp_presence_list = []
            resp_severity_list = []  # Create a new list to store mapping result

            # Two new columns to record the time of severe, and moderate frequence
            resp_severe_list = []
            resp_moderate_list = []

            # ['Respiratory_Infections'] will be replaced by the number of occurence of LRTI to indicate the presence of LRTI
            for i in X["Respiratory_Infections"]:
                if eval(i).count("Unknown") > 9 - self.minimal_value_presence:
                    resp_presence_list.append(np.nan)
                else:
                    resp_presence_list.append(eval(i).count("LRTI"))

            #
            for i in range(len(X["Severity_of_Respiratoryinfections"])):
                if (
                        eval(X["Severity_of_Respiratoryinfections"][i]).count("Unknown")
                        > 9 - self.minimal_value_presence
                ):
                    resp_presence_list.append(np.nan)
                else:
                    resp_severity_list.append(
                        round(
                            pd.Series(eval(X["Severity_of_Respiratoryinfections"][i])).map(severity_dict).sum(),
                            2,
                        )
                    )

            for i in X.Severity_of_Respiratoryinfections:
                resp_severe_list.append(eval(i).count("Severe"))
                resp_moderate_list.append(eval(i).count("Moderate"))

            X["Respiratory_Infections"] = pd.Series(resp_presence_list, index=X.index)
            X["Severity_of_Respiratoryinfections"] = pd.Series(
                resp_severity_list, index=X.index
            )

            # Add two more columns to indicate Severe or Moderate Occurence
            X["Respiratory_Ever_Severe"] = pd.Series(resp_severe_list, index=X.index)
            X["Respiratory_Ever_Moderate"] = pd.Series(
                resp_moderate_list, index=X.index
            )

            # Two more columns will be created
            print(
                "The output columns will be: Respiratory_Infections, Severity_of_Respiratoryinfections, Respiratory_Ever_Severe, Respiratory_Ever_Moderate "
            )

            X.drop(columns=["Respiratory_Report_Months"], inplace=True)

            return X


# Feature selection based on missingness and collinearity
class ColumnFilter(BaseEstimator, TransformerMixin):
    """
    Drop the repetitious columns (columns that have a 0.95 correlation factor with any other columns ) with greater missingness and higher correlation with other columns to reduce the dimensionality of feature matrix.

    Parameters for Log1pTransformer:
    ----------

    repetition_standard: float, default: 0.95
        Represent the a very high degree of collinearity with other features, need to be removed to help stablizing modeling and interpretability.

    corr_threshold: float, default: 0.6
        0.9 very strong, 0.8, fairly strong, 0.6 moderate positive. It will only be used in combination of feature missingness when dropping them.

    feature_missingness: float, default 0.05 (as percentage 5%)
        A very small number of missingness wouldn't hurt the model to much, and is relatively more robust when being KNN, RandomForest imputed.

    corr_and_missing: boolean, default True
        If set to false, number of columns will be drastically reduced, as either corr or missing will lead to columns to be removed

    Return:
    ---------
    Transformed X will have the repeated columns removed, and its features that will normally have a reduced level of collinarity and missingness.
    """

    def __init__(
            self,
            repetition_standard=0.95,
            corr_threshold=0.7,
            feature_missingness=0.05,
            corr_and_missing=True,
    ):
        super().__init__()
        self.repetition_standard = repetition_standard
        self.corr_threshold = corr_threshold
        self.feature_missingness = feature_missingness
        self.corr_and_missing = corr_and_missing

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        corr_X_df = X.corr()
        repetition_drop = []  # Feature will be dropped because of high correlation
        repetition_dict = {}

        for i, col in enumerate(corr_X_df.columns):
            for i_name in corr_X_df[col].index[i + 1:]:
                if corr_X_df[col][i_name] > self.repetition_standard:
                    repetition_drop.extend([i_name])
                    repetition_dict[
                        " <> ".join([i_name, corr_X_df[col].name])
                    ] = corr_X_df[col][i_name]

        corr_list_high = (
            []
        )  # For features with certain level of missingness, if they also have a moderate level of correlation (>0.6/0.7), they will be dropped.
        corr_dict_high = {}

        for i, col in enumerate(corr_X_df.columns):
            for i_name in corr_X_df[col].index[i + 1:]:
                if corr_X_df[col][i_name] > self.corr_threshold:
                    corr_list_high.extend([i_name])
                    corr_dict_high[
                        " <> ".join([i_name, corr_X_df[col].name])
                    ] = corr_X_df[col][i_name]

        feature_missing_high = []
        feature_missing_dict = {}

        for col in X.columns:
            if X[col].isna().mean() > self.feature_missingness:
                feature_missing_high.append(col)
                feature_missing_dict[col] = X[col].isna().mean()

        if self.corr_and_missing:
            two_factor_drop = set(corr_list_high) & set(feature_missing_high)
        else:
            two_factor_drop = set(corr_list_high) | set(feature_missing_high)

        final_drop = two_factor_drop | set(repetition_drop)

        # Print out important information regarding columns selection:

        print(
            "---------------------------------------------------------------------------------------------------"
        )
        print(
            f"Given the correlation threshhold of {self.repetition_standard}, the columns that will be removed are:{list(set(repetition_drop))}. Please see the following correlation:{repetition_dict}"
        )

        print(
            "---------------------------------------------------------------------------------------------------"
        )
        print(
            f"Given the correlation threshhold of {self.corr_threshold}, the columns that will be considered to be dropped are:{list(set(corr_list_high))}. Please see the following correlation:{corr_dict_high}"
        )
        print(
            "---------------------------------------------------------------------------------------------------"
        )
        print(
            f"Given the missingness threshhold of {self.feature_missingness}, the columns that will be considered to be dropped are:  {feature_missing_high}. Please see the following missingness:{feature_missing_dict}"
        )
        print(
            "---------------------------------------------------------------------------------------------------"
        )
        print(
            f"The finalized dropped columns are {sorted(final_drop)} with two factor dropped columns {two_factor_drop} and repetition dropped {repetition_drop}"
        )

        X.drop(
            columns=final_drop, inplace=True,
        )

        return X


# Feature Engineering maternal Stress level
class DiscretizePSS(BaseEstimator, TransformerMixin):
    """
    Discretize PSS or CESD score into ordinal categories to suppress the numeric linearity between scores.

    Parameters:
    ----------
    discretize: string, default "Ordinal".
        Other selections include: "Log1p", "None"

    pss_mapping: dict, defautl {'Low_Stress':13.5,'Moderate_Stress':26.5,'High_Stress':41}
        See ref: https://das.nh.gov/wellness/docs/percieved%20stress%20scale.pdf

    cesd_cutoff: dict, default {'Low_Depression':19.5, 'High_Depression':60}
        See ref: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0155431#sec022

    Return:
    ---------
    Transformed X with discretized or log-fied PSS or CESD score
    """

    def __init__(
            self,
            discretize="None",
            pss_mapping={"Low_Stress": 13.5, "Moderate_Stress": 26.5, "High_Stress": 41},
            cesd_cutoff={"Low_Depression": 19.5, "High_Depression": 60},
    ):
        super().__init__()
        self.discretize = discretize
        self.pss_mapping = pss_mapping
        self.cesd_cutoff = cesd_cutoff

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.discretize == "Ordinal":
            for coln in X.columns[X.columns.str.contains("PSS_")]:
                X[coln] = pd.cut(
                    X[coln],
                    [
                        -1,
                        self.pss_mapping["Low_Stress"],
                        self.pss_mapping["Moderate_Stress"],
                        self.pss_mapping["High_Stress"],
                    ],
                    labels=["Low_Stress", "Moderate_Stress", "High_Stress"],
                )

            for coln in X.columns[X.columns.str.contains("CESD_")]:
                X[coln] = pd.cut(
                    X[coln],
                    [
                        -1,
                        self.cesd_cutoff["Low_Depression"],
                        self.cesd_cutoff["High_Depression"],
                    ],
                    labels=["Low_Depression", "High_Depression"],
                )
        elif self.discretize == "Log1p":
            for coln in X.columns[X.columns.str.contains("PSS_|CESD_")]:
                X[coln] = np.log1p(X[coln])

        return X


# Feature selection based on the missingness level for the following ML modelling
class ImputerStrategizer(BaseEstimator, TransformerMixin):
    """
    Control the overall imputation strategies: "No imputation", "Very selective imputation", "All imputation acceptable".

    Parameters:
    ----------
    strategy: string, default: "no_imputation"
        Available strategies include: "no_imputation", "selective_imputation", "all_imputation"

    selective_threshold: integer, default: 5
        No. of the missing value for existing features allowed, based on the missingness: 5, 20/25, 50/60, 100, 140/160, 300, all (highest 389)
        Only take effects in combination with strategy of "selective_imputation", or will be ignored.

    Return:
    ---------
    Transformed X with imputation level selected.
    """

    def __init__(self, strategy="all_imputation", selective_threshold=5):
        super().__init__()
        self.strategy = strategy
        self.selective_threshold = selective_threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.strategy == "all_imputation":
            imputation_series = (
                X.isna().sum().sort_values()[X.isna().sum().sort_values() > 0]
            )
            print(
                "All those with missing value will be imputated, the features with number of missing values are: ",
                (list(imputation_series.index), list(imputation_series.values)),
            )
            print(
                "In which columns with categorical values are: ",
                X.select_dtypes(include="object").columns.to_list(),
            )
            return X

        elif self.strategy == "no_imputation":
            print(
                "The features for direct modelling are: ",
                X.dropna(axis=1).columns.to_list(),
            )
            print(
                "In which columns with categorical values are: ",
                X.dropna(axis=1).select_dtypes(include="object").columns.to_list(),
            )
            X.dropna(axis=1, inplace=True)
            return X

        elif self.strategy == "selective_imputation":

            dropped_ser = X.isna().sum()[X.isna().sum() > self.selective_threshold]
            dropped_columns = (
                X.isna().sum()[X.isna().sum() > self.selective_threshold].index
            )

            imputation_ser = X.isna().sum()[
                (X.isna().sum() <= self.selective_threshold) & (X.isna().sum() > 0)
                ]
            imputation_columns = (
                X.isna()
                    .sum()[
                    (X.isna().sum() <= self.selective_threshold) & (X.isna().sum() > 0)
                    ]
                    .index
            )

            print("The features that will be imputed are: ", imputation_ser)
            print(
                "The features for direct modelling are: ",
                X.dropna(axis=1).columns.to_list(),
            )
            print("The features that will be dropped are: ", dropped_ser)

            X.drop(columns=dropped_columns, inplace=True)
            return X

        else:
            msg = (
                "Wrong input for parameter `strategy`. Expected "
                "'all_imputation', 'no_imputation', 'selective_imputation', got {}"
            )
            print(msg)
            raise ValueError(msg.format(type(self.strategy)))


# Imputation on categorical features in combination of one-hot-encoding
class CatNaNImputer(BaseEstimator, TransformerMixin):
    """
    The unknown in the categorical columns will either be imputed, ignored, or be treated as an additional category.
    Then one-hot-encoding will be applied to prepare the data for machine learning.
    For one-hot-encoding and drop first will only be applied to binary categories, for easy intepretation of coefficients and feature importance.

    Parameters for:
    ----------
    NaN_signal_thresh: integer, Default: 30
        If the missingness for a categorical feature is greater than 30, then the missingness will be used as an additional information for model building.
        The adopted missingness name is called "Missing"
        The number should be high enough so that a statistic significant understanding can be established.

    NaN_imputation_strategy: string, Default: "ignore"
        Availabe strategy include: "mode" or "ignore". When "mode" is chosen, the replacement will be the category with highest frequency.
        "mode" is only recommended when there exist an evident majority (> 80%) category.

    Return:
    ---------
    Transformed X where all "categorical, object" columns will first be imputed, then one-hot-encoded.
    """

    def __init__(self, NaN_signal_thresh=30, NaN_imputation_strategy="ignore"):
        super().__init__()
        self.NaN_signal_thresh = NaN_signal_thresh
        self.NaN_imputation_strategy = NaN_imputation_strategy
        self.mode_value_ = {}

    def fit(self, X, y=None):

        cols_to_ohe = X.select_dtypes(include=["object", "category"]).columns

        for i in cols_to_ohe:
            self.mode_value_[i] = (
                X[i].value_counts().index[0]
            )  # Note: This should be included in the fit procedure. And fill in during the transform phase.

        return self

    def transform(self, X, y=None):

        cols_to_ohe = X.select_dtypes(include=["object", "category"]).columns
        for i in cols_to_ohe:
            # Cast all columns into one type for unified operations
            X[i] = X[i].astype("object")  # NaN will be kept as NaN

            if X[i].isna().sum() >= self.NaN_signal_thresh:
                X[i].fillna(value="Missing", inplace=True)
                addon_naming = pd.get_dummies(X[[i]], dummy_na=False).columns.values
                X[addon_naming] = pd.get_dummies(X[[i]], dummy_na=False)
                X.drop(columns=i, inplace=True)

            else:

                if self.NaN_imputation_strategy == "ignore":

                    # One Hot Encoding - dependent on whether its binary
                    if X[i].nunique() == 2:
                        addon_naming = pd.get_dummies(
                            X[[i]], dummy_na=False, drop_first=True
                        ).columns.values
                        X[addon_naming] = pd.get_dummies(
                            X[[i]], dummy_na=False, drop_first=True
                        )
                        X.drop(columns=i, inplace=True)
                    else:  # If categorical feature are non-binary, all dummied columns will be kept
                        addon_naming = pd.get_dummies(
                            X[[i]], dummy_na=False, drop_first=False
                        ).columns.values
                        X[addon_naming] = pd.get_dummies(
                            X[[i]], dummy_na=False, drop_first=False
                        )
                        X.drop(columns=i, inplace=True)

                elif self.NaN_imputation_strategy == "mode":

                    X[i].fillna(value=self.mode_value_[i], inplace=True)

                    # One Hot Encoding - dependent on whether its binary
                    if X[i].nunique() == 2:
                        addon_naming = pd.get_dummies(
                            X[[i]], dummy_na=False, drop_first=True
                        ).columns.values
                        X[addon_naming] = pd.get_dummies(
                            X[[i]], dummy_na=False, drop_first=True
                        )
                        X.drop(columns=i, inplace=True)
                    else:  # If categorical feature are non-binary, all dummied columns will be kept
                        addon_naming = pd.get_dummies(
                            X[[i]], dummy_na=False, drop_first=False
                        ).columns.values
                        X[addon_naming] = pd.get_dummies(
                            X[[i]], dummy_na=False, drop_first=False
                        )
                        X.drop(columns=i, inplace=True)
                else:
                    msg = (
                        "Wrong input for parameter `NaN_imputation_strategy`. Expected "
                        "'ignore', 'mode', got {}"
                    )
                    print(msg)
                    raise ValueError(msg.format(type(self.strategy)))

        return X


# Imputation on Numeric features in combination with imputation techniques (Simple, KNN, Missforest, etc)
class NumNaNimputer(BaseEstimator, TransformerMixin):
    """
    Impute Numeric columns based on the characteristics of the each feature
    -- number of unique values, number of nullness, and feature groups

    :param add_indicator_threshold, default 30
        Only those with a statistically meaningful intepretation will be add a column
    :param imputing_corrected_subset, string defautl "KNN"
        Either KNN or Random Forest will be used for imputating subset of groups that has certain degree of correlation.

    :return Imputed X that could be directly put into ML models
    """

    def __init__(self, add_indicator_threshold=30, imputing_correlated_subset="KNN", random_state=2021):
        super().__init__()
        self.add_indicator_threshold = add_indicator_threshold
        self.imputing_correlated_subset = imputing_correlated_subset
        self.random_state = random_state
        self.num_strategies = (
            {}
        )  # Dictionary to store columns for different kind of groups for imputing
        self.num_strategies_df_ = (
            pd.DataFrame()
        )  # DataFrame for overviewing the missingness for different columns
        # Imputer Initializing
        if self.imputing_correlated_subset == "KNN":
            self.BF_imputer = KNNImputer()
            self.Weight_imputer = KNNImputer()
            self.RESP_imputer = KNNImputer()
        else:
            self.BF_imputer = MissForest(random_state=random_state)
            self.Weight_imputer = MissForest(random_state=random_state)
            self.RESP_imputer = MissForest(random_state=random_state)
        self.Binary_ignore_missing = SimpleImputer(
            add_indicator=False, strategy="most_frequent"
        )
        self.Binary_indicator_missing = SimpleImputer(
            add_indicator=True, strategy="most_frequent"
        )
        self.Nonbinary_ignore_missing = SimpleImputer(
            add_indicator=False, strategy="median"
        )
        self.Nonbinary_indicator_missing = SimpleImputer(
            add_indicator=True, strategy="median"
        )

    def fit(self, X, y=None):
        """
        Generate numeric columns that will be processed differently based on the X characteristics
            Only those with a statistically significant missingness will be accompanied with a column of missingness indicator
        Fitted_outcome:
            1. Dictionary of different subsets of columns for different imputation schemes that will be referred during transformation.
                Valid keys include: BF_columns, Weight_columns, Resp_columns, Uni_binary_columns_indicator, Uni_binary_columns_ignore,
                Uni_nonbinary_columns_indicator_median, Uni_nonbinary_columns_ignore_median
            2. fitted imputer - using fit method
            3. Attribute: num_strategies_df_: DataFrame for overviewing the missingness for different columns
        """
        # Step 1: Strategize imputation techniques based on X feature characteristics
        cols_with_na = X.isna().sum()[X.isna().sum() > 0].index

        top_percentage = {}
        for i in X[cols_with_na].columns:
            top_percentage[i] = round(
                (X[i].value_counts(normalize=True).values[0] * 100), 2
            )
        per_ser = pd.Series(top_percentage)

        self.num_strategies_df_ = pd.concat(
            [X[cols_with_na].nunique(), X[cols_with_na].isna().sum(), per_ser], axis=1
        ).rename(
            columns={
                0: "Num_Unique_Values",
                1: "Num_Missing_Values",
                2: "Top_Percentage",
            }
        )

        self.num_strategies_df_.sort_values(
            by="Top_Percentage", ascending=False, inplace=True
        )

        # 1.1 For multi-variate imputation based on subset correlation: (KNN, Miss)

        # Based on pairwise correlation of columns,
        # KNN imputer for following subset of columns, the rest is more likely to be noise, rather than signal
        # BF_ was removed because there is only one feature that was not binary - Or Imputation could be done in
        # TODO: Numeric Imputation Upgrade - Next Step
        #  Three Stages: (1) binary variables imputation (2) grouped variables imputation
        #  (3) Rest numeric imputation if any For now the BF_ will be removed as there is only one non-binary feature
        # BF_columns = set(X.columns[X.columns.str.contains("BF_")])
        Weight_columns = set(X.columns[X.columns.str.contains("Weight_")]) | {
            "Gest_Days"
        }
        # Remove binary imputation from previous Resp_columns
        # Resp_columns = set(X.columns[X.columns.str.contains("Wheeze_|RI|Respiratory")])
        Resp_columns = {
            i
            for i in set(
                X.columns[
                    X.columns.str.contains("Wheeze_|RI|Respiratory")
                ]
            )
            if X[i].nunique() >= 3
        }

        # self.num_strategies["BF_columns"] = BF_columns
        self.num_strategies["Weight_columns"] = Weight_columns
        self.num_strategies["Resp_columns"] = Resp_columns

        # 1.2 For Uni-Variate imputation

        # Simple for the rest columns with na values (Univariate)
        # Ungrouped_columns = set(cols_with_na) - (
        #     BF_columns | Weight_columns | Resp_columns
        # )
        Ungrouped_columns = set(cols_with_na) - (
                Weight_columns | Resp_columns
        )
        Uni_binary_columns_indicator = set()
        Uni_binary_columns_ignore = set()
        Uni_nonbinary_columns_indicator_median = set()
        Uni_nonbinary_columns_ignore_median = set()

        for col in Ungrouped_columns:
            # Only mode - can be biased when there is no indicator, however, curse of dimensionality is another concern
            if (self.num_strategies_df_.loc[col]["Num_Unique_Values"] == 2) & (
                    self.num_strategies_df_.loc[col]["Num_Missing_Values"]
                    < self.add_indicator_threshold
            ):
                Uni_binary_columns_ignore.add(col)
            elif (self.num_strategies_df_.loc[col]["Num_Unique_Values"] == 2) & (
                    self.num_strategies_df_.loc[col]["Num_Missing_Values"]
                    >= self.add_indicator_threshold
            ):
                Uni_binary_columns_indicator.add(col)
            # Middle number will be accepted.
            elif (self.num_strategies_df_.loc[col]["Num_Unique_Values"] != 2) & (
                    self.num_strategies_df_.loc[col]["Num_Missing_Values"]
                    < self.add_indicator_threshold
            ):
                Uni_nonbinary_columns_ignore_median.add(col)
            else:
                Uni_nonbinary_columns_indicator_median.add(col)

        self.num_strategies[
            "Uni_binary_columns_indicator"
        ] = Uni_binary_columns_indicator
        self.num_strategies["Uni_binary_columns_ignore"] = Uni_binary_columns_ignore
        self.num_strategies[
            "Uni_nonbinary_columns_indicator_median"
        ] = Uni_nonbinary_columns_indicator_median
        self.num_strategies[
            "Uni_nonbinary_columns_ignore_median"
        ] = Uni_nonbinary_columns_ignore_median

        # Step 2: Begin fitting using different techniques

        # Multi-variate fit:
        # if len(self.num_strategies["BF_columns"]):
        #     self.BF_imputer.fit(X[self.num_strategies["BF_columns"]])
        if len(self.num_strategies["Weight_columns"]):
            self.Weight_imputer.fit(X[self.num_strategies["Weight_columns"]])
        if len(self.num_strategies["Resp_columns"]):
            self.RESP_imputer.fit(X[self.num_strategies["Resp_columns"]])

        # Uni-variate fit:
        if len(self.num_strategies["Uni_binary_columns_indicator"]):
            self.Binary_indicator_missing.fit(
                X[self.num_strategies["Uni_binary_columns_indicator"]]
            )

        if len(self.num_strategies["Uni_nonbinary_columns_indicator_median"]):
            self.Nonbinary_indicator_missing.fit(
                X[self.num_strategies["Uni_nonbinary_columns_indicator_median"]]
            )

        if len(self.num_strategies["Uni_binary_columns_ignore"]):
            self.Binary_ignore_missing.fit(
                X[self.num_strategies["Uni_binary_columns_ignore"]]
            )

        if len(self.num_strategies["Uni_nonbinary_columns_ignore_median"]):
            self.Nonbinary_ignore_missing.fit(
                X[self.num_strategies["Uni_nonbinary_columns_ignore_median"]]
            )

        return self

    def transform(self, X, y=None):

        # Multi-variate transform
        # if len(self.num_strategies["BF_columns"]):
        #     X.loc[:, self.num_strategies["BF_columns"]] = self.BF_imputer.transform(
        #         X[self.num_strategies["BF_columns"]]
        #     )
        if len(self.num_strategies["Weight_columns"]):
            X.loc[:, self.num_strategies["Weight_columns"]] = self.Weight_imputer.transform(
                X[self.num_strategies["Weight_columns"]]
            )
        if len(self.num_strategies["Resp_columns"]):
            X.loc[:, self.num_strategies["Resp_columns"]] = self.RESP_imputer.transform(
                X[self.num_strategies["Resp_columns"]]
            )

        # Uni-variate transform
        if len(self.num_strategies["Uni_binary_columns_indicator"]):
            added_columns_binary = [
                i + "_Missing" for i in self.num_strategies["Uni_binary_columns_indicator"]
            ]
            binary_columns_with_indicator = (
                    list(self.num_strategies["Uni_binary_columns_indicator"])
                    + added_columns_binary
            )
            X[binary_columns_with_indicator] = self.Binary_indicator_missing.transform(
                X[self.num_strategies["Uni_binary_columns_indicator"]]
            )

        if len(self.num_strategies["Uni_nonbinary_columns_indicator_median"]):
            added_columns_nonbinary = [
                i + "_Missing"
                for i in self.num_strategies["Uni_nonbinary_columns_indicator_median"]
            ]
            nonbinary_columns_with_indicator = (
                    list(self.num_strategies["Uni_nonbinary_columns_indicator_median"])
                    + added_columns_nonbinary
            )
            X[
                nonbinary_columns_with_indicator
            ] = self.Nonbinary_indicator_missing.transform(
                X[self.num_strategies["Uni_nonbinary_columns_indicator_median"]]
            )

        if len(self.num_strategies["Uni_binary_columns_ignore"]):
            X[
                list(self.num_strategies["Uni_binary_columns_ignore"])
            ] = self.Binary_ignore_missing.transform(
                X[self.num_strategies["Uni_binary_columns_ignore"]]
            )

        if len(self.num_strategies["Uni_nonbinary_columns_ignore_median"]):
            X[
                list(self.num_strategies["Uni_nonbinary_columns_ignore_median"])
            ] = self.Nonbinary_ignore_missing.transform(
                X[self.num_strategies["Uni_nonbinary_columns_ignore_median"]]
            )

        return X


# Numeric Imputation strategies based on different numeric features (subset of KNN/RF needs to be added later on)
# def numeric_imputation_selector(X, add_indicator_threshold=30):
#     """Generate numeric columns that will be processed differently.
#
#     Parameters:
#     ----------------
#     X: DataFrame, Default None
#         Transformed numeric X where columns will be divided into subgroups for different imputation.
#
#     add_indicator_threshold: integer, default 30
#         Only those with a statistically meaning intepretation will be add a column
#
#     Returns:
#     ----------------
#     1. Dictionary of different subsets of columns for different imputation schemes.
#         Valid keys include: BF_columns, Weight_columns, Resp_columns, Uni_binary_columns_indicator, Uni_binary_columns_ignore,
#         Uni_nonbinary_columns_indicator_median, Uni_nonbinary_columns_ignore_median
#     2. DataFrame for overviewing the missingness for different columns
#
#     """
#
#     NumStraties = (
#         {}
#     )  # Dictionary to store columns for different kind of groups for imputing
#     NumStraties_df = (
#         pd.DataFrame()
#     )  # DataFrame for overviewing the missingness for different columns
#
#     cols_with_na = X.isna().sum()[X.isna().sum() > 0].index
#
#     top_percentage = {}
#     for i in X[cols_with_na].columns:
#         top_percentage[i] = round(
#             (X[i].value_counts(normalize=True).values[0] * 100), 2
#         )
#     per_ser = pd.Series(top_percentage)
#
#     NumStraties_df = pd.concat(
#         [X[cols_with_na].nunique(), X[cols_with_na].isna().sum(), per_ser], axis=1
#     ).rename(
#         columns={0: "Num_Unique_Values", 1: "Num_Missing_Values", 2: "Top_Percentage"}
#     )
#
#     NumStraties_df.sort_values(by="Top_Percentage", ascending=False, inplace=True)
#
#     # For multi-variate imputation: (KNN, Miss)
#
#     # Based on pairwise correlation of columns,
#     # KNN imputer for following subset of columns, the rest is more likely to be noise, rather than signal
#     BF_columns = set(X.columns[X.columns.str.contains("BF_")])
#     Weight_columns = set(X.columns[X.columns.str.contains("Weight_")]) | {"Gest_Days"}
#     Resp_columns = set(X.columns[X.columns.str.contains("Wheeze_|RI|Respiratory")])
#
#     NumStraties["BF_columns"] = BF_columns
#     NumStraties["Weight_columns"] = Weight_columns
#     NumStraties["Resp_columns"] = Resp_columns
#
#     # For Uni-Variate imputation
#
#     # Simple for the rest columns with na values (Univariate)
#     Ungrouped_columns = set(cols_with_na) - (BF_columns | Weight_columns | Resp_columns)
#
#     Uni_binary_columns_indicator = set()
#     Uni_binary_columns_ignore = set()
#     Uni_nonbinary_columns_indicator_median = set()
#     Uni_nonbinary_columns_ignore_median = set()
#
#     for col in Ungrouped_columns:
#         # Only mode - can be biased when there is no indicator, however, curse of dimensionality is another concern
#         if (NumStraties_df.loc[col]["Num_Unique_Values"] == 2) & (
#                 NumStraties_df.loc[col]["Num_Missing_Values"] < add_indicator_threshold
#         ):
#             Uni_binary_columns_ignore.add(col)
#         elif (NumStraties_df.loc[col]["Num_Unique_Values"] == 2) & (
#                 NumStraties_df.loc[col]["Num_Missing_Values"] >= add_indicator_threshold
#         ):
#             Uni_binary_columns_indicator.add(col)
#         # Middle number will be accepted.
#         elif (NumStraties_df.loc[col]["Num_Unique_Values"] != 2) & (
#                 NumStraties_df.loc[col]["Num_Missing_Values"] < add_indicator_threshold
#         ):
#             Uni_nonbinary_columns_ignore_median.add(col)
#         else:
#             Uni_nonbinary_columns_indicator_median.add(col)
#
#     NumStraties["Uni_binary_columns_indicator"] = Uni_binary_columns_indicator
#     NumStraties["Uni_binary_columns_ignore"] = Uni_binary_columns_ignore
#     NumStraties[
#         "Uni_nonbinary_columns_indicator_median"
#     ] = Uni_nonbinary_columns_indicator_median
#     NumStraties[
#         "Uni_nonbinary_columns_ignore_median"
#     ] = Uni_nonbinary_columns_ignore_median
#
#     return NumStraties, NumStraties_df


# Numeric Imputation Process using KNN or Random Forest
# def numeric_imputation(X, imputation_dict, imputing_correlated_subset="KNN"):
#     """Imputing numeric columns based on Dictionary passed from NumImputingSelector().
#
#     Parameters:
#     ----------------
#     X: DataFrame
#         Transformed numeric X where missing value will be imputed according to the design of NumImputingSelector()
#
#
#     imputation_dict: Dictionary, default dictionary (first element) from NumImputingSelector()
#         Valid keys include: BF_columns, Weight_columns, Resp_columns, Uni_binary_columns_indicator, Uni_binary_columns_ignore,
#         Uni_nonbinary_columns_indicator_median, Uni_nonbinary_columns_ignore_median
#
#
#     imputing_correlated_subset: string, default 'KNN'
#         Either KNN or Random Forest will be used for imputating subset of groups that has certain degree of correlation.
#
#
#     Returns:
#     ----------------
#     Transformed X with no missing value, which is ready used to be used for further standard process.
#     """
#
#     if imputing_correlated_subset == "KNN":
#         imputer = KNNImputer()
#     else:
#         imputer = MissForest()
#
#     # For intercorrelated subset of numeric variables, multivariate imputation will be applied.
#
#     # BF Subset
#     # SideNote: You need to change the set to list in order for the following to work, or use the loc function
#     # However, X[imputation_dict["BF_columns"]] will not work properly during value assignment
#     # Option A: X[list(imputation_dict["BF_columns"])] = imputer.fit_transform(X[imputation_dict["BF_columns"]])
#     # Option B:
#     X.loc[:, imputation_dict["BF_columns"]] = imputer.fit_transform(
#         X[imputation_dict["BF_columns"]]
#     )  # fit_transform return a numpy array of numeric numbers with no column names
#     # Weight Subset
#     X.loc[:, imputation_dict["Weight_columns"]] = imputer.fit_transform(
#         X[imputation_dict["Weight_columns"]]
#     )
#     # Resp Subset
#     X.loc[:, imputation_dict["Resp_columns"]] = imputer.fit_transform(
#         X[imputation_dict["Resp_columns"]]
#     )
#
#     added_columns_binary = [
#         i + "_Missing" for i in imputation_dict["Uni_binary_columns_indicator"]
#     ]
#     added_columns_nonbinary = [
#         i + "_Missing"
#         for i in imputation_dict["Uni_nonbinary_columns_indicator_median"]
#     ]
#
#     binary_columns_with_indicator = (
#             list(imputation_dict["Uni_binary_columns_indicator"]) + added_columns_binary
#     )
#     nonbinary_columns_with_indicator = (
#             list(imputation_dict["Uni_nonbinary_columns_indicator_median"])
#             + added_columns_nonbinary
#     )
#
#     # For other numeric variables, univariate imputation will be applied.
#
#     if len(imputation_dict["Uni_binary_columns_indicator"]):
#         X[binary_columns_with_indicator] = SimpleImputer(
#             add_indicator=True, strategy="most_frequent"
#         ).fit_transform(X[imputation_dict["Uni_binary_columns_indicator"]])
#
#     if len(imputation_dict["Uni_nonbinary_columns_indicator_median"]):
#         X[nonbinary_columns_with_indicator] = SimpleImputer(
#             add_indicator=True, strategy="median"
#         ).fit_transform(X[imputation_dict["Uni_nonbinary_columns_indicator_median"]])
#
#     if len(imputation_dict["Uni_binary_columns_ignore"]):
#         X[list(imputation_dict["Uni_binary_columns_ignore"])] = SimpleImputer(
#             add_indicator=False, strategy="most_frequent"
#         ).fit_transform(X[imputation_dict["Uni_binary_columns_ignore"]])
#
#     if len(imputation_dict["Uni_nonbinary_columns_ignore_median"]):
#         X[list(imputation_dict["Uni_nonbinary_columns_ignore_median"])] = SimpleImputer(
#             add_indicator=False, strategy="median"
#         ).fit_transform(X[imputation_dict["Uni_nonbinary_columns_ignore_median"]])
#
#     return X


# Recheck collinearity after imputation with missingness indicator
class CollinearRemover(BaseEstimator, TransformerMixin):
    """
    Remove one of pair columns that has a very high correlation with another feature, over 0.95
    :param: collinear_level = 0.95
    :return: transformed X with no repetitious features
    """

    def __init__(self, collinear_level=0.95):
        super().__init__()
        self.collinear_level = collinear_level

    def fit(self, X, y=None):
        """Find out which features to drop due to very high collinearity
        """
        corr_X_df = X.corr()
        self.repetition_drop = []  # Feature will be dropped because of high correlation
        self.repetition_dict = {}

        for i, col in enumerate(corr_X_df.columns):
            for i_name in corr_X_df[col].index[i + 1:]:
                if corr_X_df[col][i_name] > self.collinear_level:
                    self.repetition_drop.extend([i_name])
                    self.repetition_dict[" <> ".join([i_name, corr_X_df[col].name])] = corr_X_df[
                        col
                    ][i_name]

        return self

    def transform(self, X, y=None):
        print(
            "********************************************************************************"
        )
        print(
            f"Given the correlation threshhold of {self.collinear_level}, the columns that will be removed are:{list(set(self.repetition_drop))}. Please see the following correlation:{self.repetition_dict}"
        )
        return X.drop(columns=list(set(self.repetition_drop)), inplace=True)
