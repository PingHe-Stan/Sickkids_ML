__author__ = 'Stan He@Sickkids.ca'
__date__ = '2021-10-13'
"""Generate xlsx in the output directory for ML pipeline
"""

import os
import numpy as np
import pandas as pd
from functools import reduce  # For multiple dataframes merge

# Data source
path = '../data/'


# Prepare dataframe for overviewing existing dataset
def data_repo_review(data_path='../data/'):
    """
    Function: Prepare a dataframe to review the existing the dataset in an organized way
    Input: path of directory for all existing data
    Output: dataframe and one image called "Current_Repo_Overview.png"
    Date: 2021-09-17 By Stan He@Sickkids
    """
    data_received = os.listdir(data_path)

    # Overview the files in different time points
    files = pd.DataFrame(data_received, columns=['Files'])
    datafiles = files[files.Files.str.contains('xlsx')].reset_index(drop=True)

    # Extract Time Point from file names
    index_values = []
    for i in datafiles.index.values:
        if datafiles.Files[i].startswith(tuple([str(i) for i in np.arange(10)])):
            index_values.append(' '.join(datafiles.Files[i].lower().split()[:2]))
        elif datafiles.Files[i].startswith(('Prenatal', 'prenatal')):
            index_values.append('Prenatal')
        elif datafiles.Files[i].startswith('Birth'):
            index_values.append('Birth')
        else:
            index_values.append('Other')
    datafiles['Time_Point'] = index_values

    # Prepare for dataset overview plot
    datafiles.sort_values(by='Time_Point').reset_index(drop=True, inplace=True)
    data_repo = datafiles.groupby('Time_Point').agg(lambda x: list(x))

    return data_repo


# Overview the selected file for modelling
def data_selection(data_path='../data/'):
    """
    Function: Make a selection for the questionnaires info to be considered
    Input: Path to the repo directory
    Output: Dictionary of DataFrame to preprocess, Dictionary of Variables to Select and Rename
    Date: 2021-09-17 By Stan He@Sickkids
    """

    # Specify Feature Dataset

    # Birth Profiling
    features_of_birth = []
    features_of_birth.append(data_path + 'Birth Q107CBIRTHCD.xlsx')
    features_of_birth.append('Birth Profiling')

    # Early Anthropometrics & Disease Record
    features_of_anthrop = []
    features_of_anthrop.append(data_path + 'Anthropometrics and sex.xlsx')
    features_of_anthrop.append('Early Anthropometrics & Disease Record')

    features_of_wheeze = []
    features_of_wheeze.append(data_path + 'Wheeze derived variables.xlsx')
    features_of_wheeze.append('Early Anthropometrics & Disease Record')

    features_of_respiratory = []
    features_of_respiratory.append(data_path + 'Respiratory infections type and severity.xlsx')
    features_of_respiratory.append('Early Anthropometrics & Disease Record')

    # Genetic

    features_of_parental_ethnicity = [0]
    features_of_parental_ethnicity[0] = data_path + "Prenatal Q91PRNMH18WK.xlsx"
    features_of_parental_ethnicity.append('Genetic')


    features_of_parental_spt = [0]
    features_of_parental_spt[0] = data_path + 'Parental SPT.xlsx'
    features_of_parental_spt.append('Genetic')

    features_of_parental_asthma = [0]
    features_of_parental_asthma[0] = data_path + 'Parental history Asthma.xlsx'
    features_of_parental_asthma.append('Genetic')

    features_of_parental_asthma = [0]
    features_of_parental_asthma[0] = data_path + 'Parental history of allergies.xlsx'
    features_of_parental_asthma.append('Genetic')

    # Environmental
    features_of_breastfeeding = [0]
    features_of_breastfeeding[0] = data_path + 'Breastfeeding variables.xlsx'
    features_of_breastfeeding.append('Environmental')

    features_of_maternal_stress = [0]
    features_of_maternal_stress[0] = data_path + 'Maternal PSS CESD scores.xlsx'
    features_of_maternal_stress.append('Environmental')

    features_of_antibiotic = [0]
    features_of_antibiotic[0] = data_path + 'antibiotics first year of life.xlsx'
    features_of_antibiotic.append('Environmental')

    features_of_smoke = [0]
    features_of_smoke[0] = data_path + 'Prenatal smoke exposure.xlsx'
    features_of_smoke.append('Environmental')

    features_of_home = [0]
    features_of_home[0] = data_path + '6 Month Q165HENV6M.xlsx'
    features_of_home.append('Environmental')

    features_of_dust = [0]
    features_of_dust[0] = data_path + 'Dust Pthalates 3m.xlsx'
    features_of_dust.append('Environmental')

    # Specify Target Dataset

    # Consistent Variable, Inconsistent Variable
    target_of_5y = [0]
    target_of_5y[0] = data_path + '5 year Q454CHCLA5Y.xlsx'
    target_of_5y.append('Target at 5y')

    target_of_3y = [0]
    target_of_3y[0] = data_path + '3 year Q378CHCLA3Y.xlsx'
    target_of_3y.append('Target at 3y')

    # Recurrent Wheeze at 3y and 5y
    target_of_recwheeze = []
    target_of_recwheeze.append(data_path + 'Wheeze derived variables.xlsx' + 'wheeze phenotypes.xlsx')
    target_of_recwheeze.append('Target of curated wheeze')


    # Define dataset dataframe from different perspectives
    data_dict = dict()
    for i in dir():
        if (i.find('features_of_') != -1) or (i.find('target_of_') != -1):
            data_dict[i] = eval(i)

    df_dataset = pd.DataFrame.from_dict(data_dict, orient='index', columns=['Files', 'Aspects']).sort_values(
        by='Aspects')

    return df_dataset[['Aspects', 'Files']]


# Merge all data into one dataframe with selected features
def feature_selection(data_path='../data/'):
    """
    Function:
    (1) Make a selection for the variables from all the dataset selected above info to be considered
    (2) Rename the original questionnaire columns to corresponding descriptive names
    (3) Merge all datasets of different excels into one large dataframe
    Input: Path to the repo directory
    Output:
    (1) Collection of Individual dataframe to be preprocessed,
    (2) Dataframe displaying selected features from all available features from dataset
    (3) Merged dataframe ready to be processed
    Date: 2021-09-20 By Stan He@Sickkids
    """
    # Create a new xlsx due to the unique format of 'Respiratory infections type and severity.xlsx'
    pd.read_excel(data_path + 'Respiratory infections type and severity.xlsx') \
        .groupby('subjectnumber').agg(lambda x: list(x)).reset_index(drop=False) \
        .to_excel(data_path + 'Respiratory infections type and severity1.xlsx', index=False)

    # Create a new xlsx to remove those who has already withdraw from study at birth
    temp = pd.read_excel(data_path + 'Birth Q107CBIRTHCD.xlsx')
    temp[temp.StudyStatus != 'Withdrawn'].to_excel(data_path + 'Birth Q107CBIRTHCD no withdraw.xlsx', index=False)

    # Specify Feature Dataset

    # Birth Profiling
    features_of_birth = data_path + 'Birth Q107CBIRTHCD no withdraw.xlsx'
    variables_of_birth = {'SubjectNumber': 'Subject_Number', 'CBIRTHCDQ2': 'Sex', 'CBIRTHCDQ3': 'No_of_Pregnancy',
                          'CBIRTHCDQ9': 'Anesthetic_delivery', 'CBIRTHCDQ10': 'Analgesics_usage_delivery',
                          'CBIRTHCDQ12': 'Mode_of_delivery', 'CBIRTHCDQ24': 'Apgar_Score_1min',
                          'CBIRTHCDQ25': 'Apgar_Score_5min',
                          'CBIRTHCDQ35': 'Respiratory_Problems_Birth', 'CBIRTHCDQ36': 'Jaundice_Birth',
                          'CBIRTHCDQ38': 'Complications_Birth',
                          'CBIRTHCDQ39': 'Stay_Duration_Hospital', 'StudyCenter': 'Study_Center',
                          'CBIRTHCDQ8a': 'Prenatal_Bleeding', 'CBIRTHCDQ8b': 'Prenatal_Nausea',
                          'CBIRTHCDQ8c': 'Prenatal_Infections',
                          'CBIRTHCDQ8d': 'Prenatal_Induced_Hypertension',
                          'CBIRTHCDQ8e': 'Prenatal_Gestational_Diabetes',
                          'CBIRTHCDQ8f': 'Prenatal_Cardiac_Disorder', 'CBIRTHCDQ8g': 'Prenatal_Hypertension',
                          'CBIRTHCDQ8h': 'Prenatal_Hypotension', 'CBIRTHCDQ8j': 'Prenatal_None_Conditions',
                          'CBIRTHCDQ8k': 'Prenatal_Other_Conditions',
                          'CBIRTHCDQ31a': 'F10min_Intubation', 'CBIRTHCDQ31b': 'F10min_Mask_Ventilation',
                          'CBIRTHCDQ31c': 'F10min_Free_Flow_Oxygen',
                          'CBIRTHCDQ31d': 'F10min_Oxygen_Mask', 'CBIRTHCDQ31e': 'F10min_Positive_Pressure_Ventilation',
                          'CBIRTHCDQ31f': 'F10min_Perineum_suction', 'CBIRTHCDQ31g': 'F10min_Suction',
                          'CBIRTHCDQ31i': 'F10min_No_Measure_Needed'}

    # Early Disease/Anthrop

    features_of_wheeze = data_path + 'Wheeze derived variables.xlsx'
    variables_of_wheeze = {'subjectnumber': 'Subject_Number',
                           'wheeze3m': 'Wheeze_3m', 'nocoldwheeze3m': 'Noncold_Wheeze_3m',
                           'nocoldwheezeepi3m': 'Epi_Noncold_Wheeze_3m',
                           'wheeze6m': 'Wheeze_6m', 'nocoldwheeze6m': 'Noncold_Wheeze_6m',
                           'nocoldwheezeepi6m': 'Epi_Noncold_Wheeze_6m',
                           'wheeze1y': 'Wheeze_1y', 'nocoldwheeze1y': 'Noncold_Wheeze_1y',
                           'nocoldwheezeepi1y': 'Epi_Noncold_Wheeze_1y',
                           'wheeze18m': 'Wheeze_18m', 'nocoldwheeze18m': 'Noncold_Wheeze_18m',
                           'nocoldwheezeepi18m': 'Epi_Noncold_Wheeze_18m',
                           'wheeze2y': 'Wheeze_2y', 'nocoldwheeze2y': 'Noncold_Wheeze_2y',
                           'nocoldwheezeepi2y': 'Epi_Noncold_Wheeze_2y',
                           'wheeze2hy': 'Wheeze_2yh', 'nocoldwheeze2hy': 'Noncold_Wheeze_2hy',
                           'nocoldwheezeepi2hy': 'Epi_Noncold_Wheeze_2hy',
                           'wheeze3y': 'Wheeze_3y', 'nocoldwheeze3y': 'Noncold_Wheeze_3y',
                           'nocoldwheezeepi3y': 'Epi_Noncold_Wheeze_3y',
                           'wheeze4y': 'Wheeze_4y', 'nocoldwheeze4y': 'Noncold_Wheeze_4y',
                           'nocoldwheezeepi4y': 'Epi_Noncold_Wheeze_4y',
                           'wheeze5y': 'Wheeze_5y', 'nocoldwheeze5y': 'Noncold_Wheeze_5y',
                           'nocoldwheezeepi5y': 'Epi_Noncold_Wheeze_5y',
                           'recwheeze3y': 'Recurrent_Wheeze_3y', 'recwheeze1y': 'Recurrent_Wheeze_1y',
                           'recwheeze5y': 'Recurrent_Wheeze_5y',
                           'cumepi3m': 'Cumulative_Wheeze_3m', 'cumepi6m': 'Cumulative_Wheeze_6m',
                           'cumepi12m': 'Cumulative_Wheeze_12m',
                           'cumepi18m': 'Cumulative_Wheeze_18m', 'cumepi24m': 'Cumulative_Wheeze_24m',
                           'cumepi30m': 'Cumulative_Wheeze_30m',
                           'cumepi36m': 'Cumulative_Wheeze_36m', 'cumepi48m': 'Cumulative_Wheeze_48m',
                           'cumepi60m': 'Cumulative_Wheeze_60m'}

    features_of_respiratory = data_path + 'Respiratory infections type and severity1.xlsx'
    variables_of_respiratory = {'subjectnumber': 'Subject_Number', 'actual_age': 'Respiratory_Report_Months',
                                'ri_type': 'Respiratory_Infections', 'ri_sev': 'Severity_of_Respiratoryinfections'}

    features_of_anthrop = data_path + 'Anthropometrics and sex.xlsx'
    variables_of_anthrop = {'subjectnumber': 'Subject_Number', 'Sex': 'Gender', 'gest_days': 'Gest_Days',
                            'weight_0': 'Weight_0m',
                            'weight_3': 'Weight_3m', 'weight_12': 'Weight_12m', 'weight_36': 'Weight_36m',
                            'weight_60': 'Weight_60m',
                            'z_weight_age_0': 'Weight_for_age_0m', 'z_weight_age_3': 'Weight_for_age_3m',
                            'z_weight_age_12': 'Weight_for_age_12m',
                            'z_weight_age_36': 'Weight_for_age_36m', 'z_weight_age_60': 'Weight_for_age_60m'}

    # Genetic
    features_of_parental_spt = data_path + 'Parental SPT.xlsx'
    variables_of_parental_spt = {'SubjectNumber': 'Subject_Number', 'dad_atopy': 'Dad_Atopy', 'dad_food': 'Dad_Food',
                                 'dad_inhalant': 'Dad_Inhalant',
                                 'mom_atopy': 'Mom_Atopy', 'mom_food': 'Mom_Food', 'mom_inhalant': 'Mom_Inhalant'}

    features_of_parental_asthma = data_path + 'Parental history Asthma.xlsx'
    variables_of_parental_asthma = {'subjectnumber': 'Subject_Number', 'Mother_Asthma': 'Mother_Asthma',
                                    'Father_Asthma': 'Father_Asthma',
                                    'Parental_Asthma': 'Parental_Asthma'}

    # TODO: Need to update previous two functions & global variables & missingness mapping functions - config as well
    features_of_parental_allergy = data_path + 'Parental history of allergies.xlsx'
    variables_of_parental_allergy = {
        "SubjectNumber": "Subject_Number",
        "WheezeMother": "Wheeze_Mother",
        "AsthmaMother": "Asthma_Mother",
        "HayFeverMother": "Hayfever_Mother",
        "ADMother": "AD_Mother",
        "Pollen_tress_Mother": "Pollentress_Mother",
        "FAllergiesMother": "FAllergies_Mother",
        "WheezeFather": "Wheeze_Father",
        "AsthmaFather": "Asthma_Father",
        "HayFeverFather": "Hayfever_Father",
        "ADFather": "AD_Father",
        "Pollen_tress_Father": "Pollentress_Father",
        "FAllergiesFather": "FAllergies_Father",
    }

    # Environmental
    features_of_breastfeeding = data_path + 'Breastfeeding variables.xlsx'
    variables_of_breastfeeding = {'SubjectNumber': 'Subject_Number', 'BF_duration_imp': 'BF_Implied_Duration',
                                  'BF_1m': 'BF_1m',
                                  'BF_3m': 'BF_3m', 'BF_6m': 'BF_6m', 'BF_9m': 'BF_9m', 'BF_12m': 'BF_12m',
                                  'BF_18m': 'BF_18m', 'BF_24m': 'BF_24m'}

    features_of_maternal_stress = data_path + 'Maternal PSS CESD scores.xlsx'
    variables_of_maternal_stress = {'SubjectNumber': 'Subject_Number',
                                    'psssumr_pre36wk': 'PSS_36week', 'csedsumr_pre36wk': 'CSED_36week',
                                    'psssumr_pre18wk': 'PSS_18week', 'csedsumr_pre18wk': 'CSED_18week',
                                    'psssumr_6m': 'PSS_6m', 'csedsumr_6m': 'CSED_6m', 'psssumr_12m': 'PSS_12m',
                                    'csedsumr_12m': 'CSED_12m',
                                    'psssumr_18m': 'PSS_18m', 'csedsumr_18m': 'CSED_18m',
                                    'psssumr_24m': 'PSS_24m', 'csedsumr_24m': 'CSED_24m'}

    features_of_antibiotic = data_path + 'antibiotics first year of life.xlsx'
    variables_of_antibiotic = {'subjectnumber': 'Subject_Number', 'atbx_n': 'Number_of_AntibioticsCourse_12m',
                               'atbx_time': 'Time_of_AntibioticsUsage_12m', 'atbx': 'Antibiotics_Usage_12m'}

    features_of_smoke = data_path + 'Prenatal smoke exposure.xlsx'
    variables_of_smoke = {'Subjectnumber': 'Subject_Number', 'prenatal_second_hand': 'Smoke_Prenatal_Secondhand',
                          'prenatal_mat_smoke': 'Smoke_Prenatal_Maternal'}

    features_of_home = data_path + '6 Month Q165HENV6M.xlsx'  # Request for more home environmental questionnaires
    variables_of_home = {'SubjectNumber': 'Subject_Number', 'HENV6MQ8': 'Home_Furry_Pets_6m',
                         'HENV6MQ7': 'Home_New_Furnitures_6m',
                         'HENV6MQ20': 'Home_Presence_Smoke_6m'}

    features_of_dust = data_path + 'Dust Pthalates 3m.xlsx'
    variables_of_dust = {'subjectnumber': 'Subject_Number', 'dep_bc_rec': 'Home_DEP_3m', 'dibp_bc_rec': 'Home_DiBP_3m',
                         'dnbp_bc_rec': 'Home_DNBP_3m', 'bzbp_bc_rec': 'Home_BzBP_3m', 'dehp_bc_rec': 'Home_DEHP_3m'}

    # Specify Target Dataset

    # Consistent Variable, Inconsistent Variable
    target_of_5y = data_path + '5 year Q454CHCLA5Y.xlsx'
    yvariables_of_5y = {'SubjectNumber': 'Subject_Number', 'CHCLA5YQ9': 'Wheeze_5yCLA',
                        'CHCLA5YQ10': 'Wheeze_withoutcold_5yCLA', 'CHCLA5YQ12': 'Medicine_for_Wheeze_5yCLA',
                        'CHCLA5YQ12_1': 'Regular_Controller_5yCLA', 'CHCLA5YQ12_2': 'Intermittent_Controller_5yCLA',
                        'CHCLA5YQ12_3': 'Reliever_5yCLA', 'CHCLA5YQ14': 'Frequency_Oral_Steroid_5yCLA',
                        'CHCLA5YQ11': 'Wheeze_Frequency_5yCLA', 'CHCLA5YQ41': 'Asthma_Diagnosis_5yCLA',
                        'CHCLA5YQ41_1': 'Viral_Asthma_5yCLA', 'CHCLA5YQ41_2': 'Triggered_Asthma_5yCLA',
                        'CHCLA5YQ46': 'Medical_Conditions_5yCLA', 'CHCLA5YQ4_1a': 'Systolic_BP_5yCLA',
                        'CHCLA5YQ4_1b': 'Diastolic_BP_5yCLA', 'CHCLA5YQ4_2': 'Pulse_Rate_5yCLA'}

    target_of_3y = data_path + '3 year Q378CHCLA3Y.xlsx'
    yvariables_of_3y = {'SubjectNumber': 'Subject_Number', 'CHCLA3YQ4_1a': 'Systolic_BP_3yCLA',
                        'CHCLA3YQ4_1b': 'Diastolic_BP_3yCLA', 'CHCLA3YQ4_2': 'Pulse_Rate_3yCLA',
                        'CHCLA3YQ8': 'Wheeze_3yCLA', 'CHCLA3YQ31': 'Asthma_Diagnosis_3yCLA',
                        'CHCLA3YQ31_1': 'Viral_Asthma_3yCLA', 'CHCLA3YQ31_2': 'Triggered_Asthma_3yCLA'}

    # TODO: Need to update previous two functions & global variables - config as well
    target_of_1y = data_path + '1 year q192CHCLA1Y.xlsx'
    yvariables_of_1y = {'SubjectNumber': 'Subject_Number', 'CHCLA1YQ11_1': 'Prolonged_Expiration_1yCLA',
                        'CHCLA1YQ11_2': 'Crackles_1yCLA', 'CHCLA1YQ11_3': 'Wheeze_1yCLA'}

    # Recurrent Wheeze at 3y and 5y
    target_of_recwheeze = data_path + 'Wheeze derived variables.xlsx'
    yvariables_of_recwheeze = {'subjectnumber': 'Subject_Number', 'recwheeze3y': 'Recurrent_Wheeze_3y',
                               'recwheeze1y': 'Recurrent_Wheeze_1y', 'recwheeze5y': 'Recurrent_Wheeze_5y'}

    # Define a dataframe dictionary for features and targets of CHILD dataset
    df_dict = dict()
    for i in dir():
        if (i.find('features_of_') != -1) or (i.find('target_of_') != -1):
            df_dict[i] = pd.read_excel(eval(i))

    variables_dict = dict()
    for i in dir():
        if (i.find('variables_of_') != -1) or (i.find('yvariables_of_') != -1):
            variables_dict[i] = list(eval(i).values())

    # Create feature selection criteria dataframe for all dataframes loaded into dataframe dictionary
    df_columns = []
    df_list = list(df_dict.keys())

    for i in range(len(df_list)):
        df_columns.append(list(df_dict[df_list[i]].columns.values))

    file_list = []
    for i in df_dict.keys():
        file_list.append(eval(i).split('/')[-1])

    features_dataframe_display = pd.DataFrame(
        {'Files': file_list, 'Columns': df_columns, 'Features': variables_dict.values()})

    variables_dict = dict()
    for i in dir():
        if (i.find('variables_of_') != -1) or (i.find('yvariables_of_') != -1):
            variables_dict[i] = eval(i)

    features_dataframe = pd.DataFrame({'Files': file_list, 'Columns': df_columns, 'Features': variables_dict.values()})

    # Clear the df_list from name variables carrier to dataframe carrier
    df_list = []

    # Append renamed dataframe to dataframe list
    for i in range(len(df_dict.keys())):
        df_current = df_dict[list(df_dict.keys())[i]]
        coln = list(features_dataframe.Features[i].keys())
        rename_dict = features_dataframe.Features[i]
        df_list.append(df_current[coln].rename(columns=rename_dict))

    # Create a merged dataframe for further engineering
    df_merged = reduce(lambda left, right: pd.merge(left, right, on=['Subject_Number'],
                                                    how='left'), df_list)

    return df_dict, features_dataframe_display, df_merged


# Restore the true missingness for merged
def data_mapping(merged_df=None):
    """
    Function: Data Mapping to restore true NA values. dataframe from previously merged and renamed datasets.
    Input: dataframe created from 'feature_selection' function (as the last element of returned value)
    Output: All Dataframe value perserved with true NaN value
    Date: 2021-09-24 By Stan He@Sickkids
    """

    df_mapping = merged_df.copy()  # Must use copy to reserve the original dataframe in the tuple (same to list)

    # For Birth Profile (Drop Sex but Keep Gender)
    df_mapping.drop(columns=['Sex'], axis=1, inplace=True)  # Drop due to repetition
    df_mapping[['No_of_Pregnancy', 'Mode_of_delivery', 'Stay_Duration_Hospital']] = df_mapping[
        ['No_of_Pregnancy', 'Mode_of_delivery', 'Stay_Duration_Hospital']].replace(
        {999: np.nan, 88: np.nan, 99: np.nan})

    # For Categorical Value, Categorical_Unknown can be used as a dummied variable 
    df_mapping['Anesthetic_delivery'] = df_mapping['Anesthetic_delivery'].replace({9: np.nan, 999: np.nan})
    df_mapping['Analgesics_usage_delivery'] = df_mapping['Analgesics_usage_delivery'].replace({9: np.nan, 999: np.nan})
    df_mapping['Apgar_Score_1min'] = df_mapping['Apgar_Score_1min'].replace({99: np.nan, 999: np.nan})
    df_mapping['Apgar_Score_5min'] = df_mapping['Apgar_Score_5min'].replace({99: np.nan, 999: np.nan})
    df_mapping['Respiratory_Problems_Birth'] = df_mapping['Respiratory_Problems_Birth'].replace(
        {9: np.nan, 999: np.nan})
    df_mapping['Jaundice_Birth'] = df_mapping['Jaundice_Birth'].replace({9: np.nan, 999: np.nan})
    df_mapping['Complications_Birth'] = df_mapping['Complications_Birth'].replace({9: np.nan, 999: np.nan})

    # No need to engineer columns 
    birth_coln_cleaned = ['Prenatal_Bleeding', 'Prenatal_Nausea', 'Prenatal_Infections',
                          'Prenatal_Induced_Hypertension', 'Prenatal_Gestational_Diabetes',
                          'Prenatal_Cardiac_Disorder', 'Prenatal_Hypertension',
                          'Prenatal_Hypotension', 'Prenatal_None_Conditions',
                          'Prenatal_Other_Conditions', 'F10min_Intubation',
                          'F10min_Mask_Ventilation', 'F10min_Free_Flow_Oxygen',
                          'F10min_Oxygen_Mask', 'F10min_Positive_Pressure_Ventilation',
                          'F10min_Perineum_suction', 'F10min_Suction',
                          'F10min_No_Measure_Needed']

    # Antibiotics - all nan in Time and Number of Usage strictly corresponding to 0 Antibiotics_Usage
    df_mapping[['Number_of_AntibioticsCourse_12m', 'Time_of_AntibioticsUsage_12m', 'Antibiotics_Usage_12m']] = df_mapping[
        ['Number_of_AntibioticsCourse_12m', 'Time_of_AntibioticsUsage_12m', 'Antibiotics_Usage_12m']].fillna(0)

    # Anthropometrics
    anthrop_cleaned_columns = ['Gest_Days', 'Weight_0m', 'Weight_3m', 'Weight_12m', 'Weight_36m',
                               'Weight_60m', 'Weight_for_age_0m', 'Weight_for_age_3m', 'Weight_for_age_12m',
                               'Weight_for_age_36m', 'Weight_for_age_60m']

    # Breastfeeding 
    breastfeeding_cleaned = ['BF_Implied_Duration', 'BF_1m', 'BF_3m', 'BF_6m', 'BF_9m',
                             'BF_12m', 'BF_18m', 'BF_24m']

    # Dust Sample
    dust_cleaned = ['Home_DEP_3m', 'Home_DiBP_3m',
                    'Home_DNBP_3m', 'Home_BzBP_3m', 'Home_DEHP_3m']

    # Home Environment - 'Percentage_OutdoorsActivity_6m' will be dropped due to high prevalence of NaN, and unreasonable estimate of percentage
    df_mapping[['Home_Furry_Pets_6m', 'Home_New_Furnitures_6m', 'Home_Presence_Smoke_6m']] = df_mapping[
        ['Home_Furry_Pets_6m', 'Home_New_Furnitures_6m', 'Home_Presence_Smoke_6m']
    ].replace({8888: np.nan, 888: np.nan, 999: np.nan})

    # Maternal Mental
    df_mapping[['PSS_36week', 'CSED_36week', 'PSS_18week', 'CSED_18week', 'PSS_6m',
                'CSED_6m', 'PSS_12m', 'CSED_12m', 'PSS_18m', 'CSED_18m', 'PSS_24m',
                'CSED_24m']] = df_mapping[['PSS_36week', 'CSED_36week', 'PSS_18week', 'CSED_18week', 'PSS_6m',
                                           'CSED_6m', 'PSS_12m', 'CSED_12m', 'PSS_18m', 'CSED_18m', 'PSS_24m',
                                           'CSED_24m']].replace({8888.0: np.nan, 999.0: np.nan})

    # Parental allergy history
    df_mapping[['Wheeze_Mother',
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
                'FAllergies_Father']] = df_mapping[['Wheeze_Mother',
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
                'FAllergies_Father']].replace({8888.0: np.nan, 999.0: np.nan, 8: np.nan, 9: np.nan})


    # Target/Variables 1y
    df_mapping[['Prolonged_Expiration_1yCLA',
    'Crackles_1yCLA',
    'Wheeze_1yCLA']] = df_mapping[['Prolonged_Expiration_1yCLA',
    'Crackles_1yCLA',
    'Wheeze_1yCLA']].replace({8888.0: np.nan, 999.0: np.nan})

    # Target 3y
    df_mapping[['Systolic_BP_3yCLA', 'Diastolic_BP_3yCLA', 'Pulse_Rate_3yCLA',
                'Wheeze_3yCLA', 'Asthma_Diagnosis_3yCLA', 'Viral_Asthma_3yCLA',
                'Triggered_Asthma_3yCLA', ]] = df_mapping[
        ['Systolic_BP_3yCLA', 'Diastolic_BP_3yCLA', 'Pulse_Rate_3yCLA',
         'Wheeze_3yCLA', 'Asthma_Diagnosis_3yCLA', 'Viral_Asthma_3yCLA',
         'Triggered_Asthma_3yCLA', ]].replace({777: np.nan, 8888: np.nan, 999: np.nan, 888: 0})

    # Target 5y
    df_mapping[['Wheeze_5yCLA',
                'Wheeze_withoutcold_5yCLA', 'Medicine_for_Wheeze_5yCLA',
                'Regular_Controller_5yCLA', 'Intermittent_Controller_5yCLA',
                'Reliever_5yCLA', 'Frequency_Oral_Steroid_5yCLA',
                'Wheeze_Frequency_5yCLA', 'Asthma_Diagnosis_5yCLA',
                'Viral_Asthma_5yCLA', 'Triggered_Asthma_5yCLA',
                'Medical_Conditions_5yCLA', 'Systolic_BP_5yCLA',
                'Diastolic_BP_5yCLA', 'Pulse_Rate_5yCLA']] = df_mapping[['Wheeze_5yCLA',
                                                                         'Wheeze_withoutcold_5yCLA',
                                                                         'Medicine_for_Wheeze_5yCLA',
                                                                         'Regular_Controller_5yCLA',
                                                                         'Intermittent_Controller_5yCLA',
                                                                         'Reliever_5yCLA',
                                                                         'Frequency_Oral_Steroid_5yCLA',
                                                                         'Wheeze_Frequency_5yCLA',
                                                                         'Asthma_Diagnosis_5yCLA',
                                                                         'Viral_Asthma_5yCLA', 'Triggered_Asthma_5yCLA',
                                                                         'Medical_Conditions_5yCLA',
                                                                         'Systolic_BP_5yCLA',
                                                                         'Diastolic_BP_5yCLA',
                                                                         'Pulse_Rate_5yCLA']].replace(
        {777: np.nan, 8888: np.nan, 999: np.nan, 888: 0})

    df_mapping[['Viral_Asthma_5yCLA', 'Triggered_Asthma_5yCLA']] = df_mapping[
        ['Viral_Asthma_5yCLA', 'Triggered_Asthma_5yCLA']
    ].replace({8: 0, 8.0: 0})

    # Target Recurrent Wheeze    
    df_mapping.drop(columns=['Recurrent_Wheeze_3y_x', 'Recurrent_Wheeze_1y_x', 'Recurrent_Wheeze_5y_x'], axis=1,
                    inplace=True)  # Drop due to repetition
    df_mapping.rename(
        columns={'Recurrent_Wheeze_3y_y': 'Recurrent_Wheeze_3y', 'Recurrent_Wheeze_1y_y': 'Recurrent_Wheeze_1y',
                 'Recurrent_Wheeze_5y_y': 'Recurrent_Wheeze_5y'}, inplace=True)

    return df_mapping


# Reverse engineering for mother pregnancy medical condition and first 10 minutes of child birth
def dummy_reversed_features(feature_tuple=None):
    """
    Function: Add two features for the baby's birth profile (1) Prenatal Mother Conditions (2) First 10 minute measures
    Input: Feature Tuple created during feature_selection process
    Output: DataFrame containing the First 10 minutes measures & Mother Conditions
    """
    raw_birth_kept = feature_tuple[0]['features_of_birth'].copy()

    # Create a feature for the mother's medical condition summary before delivery
    temp = raw_birth_kept[['SubjectNumber', 'CBIRTHCDQ8a', 'CBIRTHCDQ8b', 'CBIRTHCDQ8c', 'CBIRTHCDQ8d', 'CBIRTHCDQ8e',
                           'CBIRTHCDQ8f', 'CBIRTHCDQ8g', 'CBIRTHCDQ8h', 'CBIRTHCDQ8i', 'CBIRTHCDQ8j',
                           'CBIRTHCDQ8k']].set_index('SubjectNumber')

    temp_2 = temp[temp == 1].stack().reset_index()
    temp_2.columns = ['SubjectNumber', 'Prenatal_Mother_Condition', 'Mother_Condition_number']
    temp_2['Prenatal_Mother_Condition'] = temp_2.Prenatal_Mother_Condition.replace(
        {'CBIRTHCDQ8a': 'Bleeding', 'CBIRTHCDQ8b': 'Nausea', 'CBIRTHCDQ8c': 'Infections',
         'CBIRTHCDQ8d': 'Pregnancy Induced Hypertension', 'CBIRTHCDQ8e': 'Gestational Diabetes',
         'CBIRTHCDQ8f': 'Cardiac Disorder', 'CBIRTHCDQ8g': 'Hypertension',
         'CBIRTHCDQ8h': 'Hypotension', 'CBIRTHCDQ8i': 'Unknown',
         'CBIRTHCDQ8j': 'None', 'CBIRTHCDQ8k': 'Other'})

    mother_condition_birth = temp_2.groupby('SubjectNumber').agg(
        {'Prenatal_Mother_Condition': ','.join, 'Mother_Condition_number': sum}).reset_index()

    # Create a feature for the adopted measures to restore baby's birth health situation of during first 10 minutes

    temp = raw_birth_kept[
        ['SubjectNumber', 'CBIRTHCDQ31a', 'CBIRTHCDQ31b', 'CBIRTHCDQ31c', 'CBIRTHCDQ31d', 'CBIRTHCDQ31e',
         'CBIRTHCDQ31f', 'CBIRTHCDQ31g', 'CBIRTHCDQ31h', 'CBIRTHCDQ31i']].set_index('SubjectNumber')
    temp_2 = temp[temp == 1].stack().reset_index()
    temp_2.columns = ['SubjectNumber', 'First_10min_Measure', 'F10m_number']
    temp_2['First_10min_Measure'] = temp_2.First_10min_Measure.replace(
        {'CBIRTHCDQ31a': 'Intubation', 'CBIRTHCDQ31b': 'Mask Ventilation', 'CBIRTHCDQ31c': 'Free Flow Oxygen',
         'CBIRTHCDQ31d': 'Oxygen Mask', 'CBIRTHCDQ31e': 'Positive Pressure Ventilation',
         'CBIRTHCDQ31f': 'Perineum suction', 'CBIRTHCDQ31g': 'Suction while in warmer',
         'CBIRTHCDQ31h': 'Unknown', 'CBIRTHCDQ31i': 'None'})

    reverse_dummy_10mins = temp_2.groupby('SubjectNumber').agg(
        {'First_10min_Measure': ','.join, 'F10m_number': sum}).reset_index()

    df_added_features_temp = feature_tuple[0]['features_of_birth'][['SubjectNumber']].merge(mother_condition_birth,
                                                                                            how='left',
                                                                                            on='SubjectNumber')
    df_added_features = pd.merge(df_added_features_temp, reverse_dummy_10mins, how='left', on='SubjectNumber')
    df_added_features.rename(columns={'SubjectNumber': 'Subject_Number'}, inplace=True)

    # # Mother condition 0, 1 0 represent: No previous condition with only or no bleeding or nausea situations
    # df_added_features['Mother_Condition_Encoded'] = df_added_features.Prenatal_Mother_Condition.str.replace(
    #     r'(^.*None.*$)', '0') \
    #     .replace(['Nausea', 'Bleeding', 'Bleeding,Nausea'], '0') \
    #     .str.replace(r'(^.*Hypertension.*$)', '1').str.replace(r'(^.*Other.*$)', '1') \
    #     .str.replace(r'(^.*Infections.*$)', '1').str.replace(r'(^.*Diabetes.*$)', '1') \
    #     .str.replace(r'(^.*Disorder.*$)', '1').str.replace(r'(^.*Hypotension.*$)', '1').replace({'Unknown': np.nan})
    #
    # # First 10 minutes 0,1,2,3 - 3 represent Intubation needed at birth, 0 represent no measure needed
    # df_added_features['First10min_Measure_Encoded'] = df_added_features.First_10min_Measure.str.replace(r'(
    # ^.*None.*$)', '0') \ .str.replace(r'(^.*Intubation.*$)', '3').str.replace(r'(^.*[M|m]ask.*$)',
    # '2') \ .str.replace(r'(^.*Suction.*$)', '1', case=False).str.replace(r'(^.*[Oxyg|Ventil].*n.*$)', '1').replace(
    # {'Unknown': np.nan})

    df_added_features = df_added_features.drop(columns=['F10m_number', 'Mother_Condition_number'])

    return df_added_features


# Generate Raw dataframe and CHILD_raw.xlsx for engineering and imputation
def generate_raw_xlsx(data_path='../data/', output_path='../output'):
    """Generate XLSX for CHILD analysis and modelling

    Parameters:
    Path to the CHILD dictionary

    Returns:
    xlsx for CHILD study analysis and ML
    """
    print(f'The path for all the CHILD data is located at {data_path}')
    print('Generating a merged DataFrame For Analyzing and Modelling...')
    feature_tuple = feature_selection(data_path)
    df_mapped = data_mapping(feature_tuple[2])
    reversed_feature = dummy_reversed_features(feature_tuple)
    df_child_raw = pd.merge(df_mapped, reversed_feature.iloc[:, :3].replace({'Unknown': np.nan}), how='left',
                            on='Subject_Number')  # The last two are engineered and therefore are left-out
    print(f'The DataFrame for ML is saved to {output_path} ')
    df_child_raw.to_excel(output_path + '/CHILD_raw.xlsx', index=False)

    return df_child_raw

