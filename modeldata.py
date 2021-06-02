import numpy as np
import pandas as pd
import csv
import re
import matplotlib.pyplot as plt
import math

# --------------------------------------Measuring Learning & #Geen match-----------------------------------------------

filename = "Modeldata_S6587_L165317_K165317.csv"

# make df_learning
def measure_learning(csvfile):
    df = pd.read_csv(csvfile)  # error_bad_lines=False

    # get only User ID and Model ID
    df_learning = df.iloc[:, [1, 2]]
    df_learning.dropna(how="all", inplace=True)

    # transpose df
    df = df.T

    lst_geen_match = []
    lst_correct = []
    lst_missing = []
    lst_wrong = []
    lst_score = []
    lst_only_score = []
    lst_correct_num = []
    lst_errors_num = []
    lst_missing_num = []
    lst_score_num = []
    i = 3
    j = 1
    while i <= len(df.columns):
        df2 = df[df.index.str.contains("geen match", case=False)]
        lst_geen_match.append(df2.iloc[:, i - 2].sum())

        # remove geen match rows
        indices = df[df.index.str.contains("geen match", case=False)].index
        df.drop(indices, inplace=True)
        df.iloc[:, i] = df.iloc[:, i].astype("string")
        df.iloc[:, i - 1] = df.iloc[:, i - 1].astype("string")
        df.iloc[:, i - 1] = df.iloc[:, i - 1].fillna("non")

        # make extra column merging the 1/0 value with the error name or non if non
        # will result in '1;non', '0;non', or '0;ERROR_NAME'
        df.insert(
            i + 1, "concat-col{}".format(j), (df.iloc[:, i] + ";" + df.iloc[:, i - 1])
        )
        # make dictionary counting each of the above values
        dct = df.iloc[1:, i + 1].value_counts().to_dict()

        # total number of ingridents (column minus User ID and Model ID)
        total = len(df.iloc[1:, i]) - 2

        ### get score for errors using difflib ###
        from difflib import SequenceMatcher

        # if i == 3:
        score_wrong = 0  # sum of partial grades
        n = 3
        while n < len(df.iloc[:, 0]):
            if type(df.iloc[n, i]) == str:
                if df.iloc[n, i] == "0":
                    # print("True")
                    if df.iloc[n, i - 1] != "non":
                        a = df.iloc[n, 0]
                        b = df.iloc[n, i - 2]
                        score_wrong += SequenceMatcher(None, a, b).ratio()
            n += 1
        ##########################################

        number_correct = 0
        number_missing = 0
        number_wrong = 0
        for key in dct:
            if key == "1;non":
                number_correct += dct[key]
            elif key == "0;non":
                number_missing += dct[key]
            else:  # i.e. if key == 0;Error_NAME
                number_wrong += dct[key]

        lst_correct.append((number_correct / total) * 100)
        lst_missing.append((number_missing / total) * 100)
        lst_wrong.append((number_wrong / total) * 100)
        lst_score.append(((number_correct + score_wrong) / total) * 100)
        lst_only_score.append(score_wrong)
        lst_correct_num.append(number_correct)
        lst_missing_num.append(number_missing)
        lst_errors_num.append(number_wrong)
        lst_score_num.append(number_correct + score_wrong)

        i += 4
        j += 1

    df_learning["%Correct"] = lst_correct
    df_learning["%Missing"] = lst_missing
    df_learning["%Wrong"] = lst_wrong  # wrong not missing
    df_learning["%Score"] = lst_score
    # %Correct + %Missing + %Wrong should = 100%
    df_learning["#Correct"] = lst_correct_num
    df_learning["#Missing"] = lst_missing_num
    df_learning["#Errors"] = lst_errors_num
    df_learning["#Score"] = lst_score_num
    df_learning["Partial grades"] = lst_only_score
    df_learning["#Geen match"] = lst_geen_match

    return df_learning


# Input: df_learning
# Output: if more than 1 model for same user, return df_learning leaving only highest scoring model
# If remove_list=True, return list of models that should be removed instead of df_learning (to remove in actionlog)
def remove_duplicates(df, remove_list=False):
    df2 = df.sort_values("%Correct").drop_duplicates("User ID", keep="last")
    if remove_list:
        removed_models = []
        for usernum, df_user in df.groupby("User ID"):
            df_user = df_user.sort_values("%Correct")
            removed_models += df_user.iloc[:-1]["Model ID"].tolist()
        return removed_models
    else:
        return df2


def errors_overall(df, plot=False):
    # most to least error made:
    print("Errors made overall in model:")
    df = df.T
    dct = {}
    i = 2
    while i < len(df.columns):
        column = df.iloc[:, i].tolist()
        lst = []
        for error in column:
            if type(error) == str:  # not math.isnan(error)
                if error.rfind("WRONG") != 0:
                    indices = [m.start() for m in re.finditer("WRONG", error)]
                    error_parts = [
                        error[i:j] for i, j in zip(indices, indices[1:] + [None])
                    ]
                    lst = lst + error_parts
                else:
                    lst.append(error)

        for error in lst:
            if error in dct:
                dct[error] += 1
            else:
                dct[error] = 1
        i += 3
    for key in dct:
        print(key, ": ", dct[key])

    if plot:
        plt.bar(range(len(dct)), list(dct.values()), align="center")
        plt.xticks(range(len(dct)), list(dct.keys()))
        plt.show()
    return sorted(dct.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)


def errors_per_stage2(df):
    # remove 'geen match' columns
    df = df.loc[:, ~df.columns.str.endswith("geen match")]

    df_errors = pd.DataFrame(columns=["Stage", "Error", "#Error"])
    columns = df.columns.tolist()
    columns.remove("User ID")
    columns.remove("Model ID")

    for column in columns:
        df_column = df[column].value_counts().to_frame(name="#Error")
        df_column["Error"] = df_column.index

        df_column["Error"] = df_column["Error"].astype("string")

        df_column = df_column[df_column["Error"].str.contains("WRONG")]
        df_column = df_column[df_column["Error"].str.count("WRONG_") == 1]

        column_name = column.split(".")[0]
        df_column["Stage"] = [column_name] * len(df_column)

        df_column.reset_index(inplace=True)
        df_column.drop(["index"], axis=1, inplace=True)
        df_column["Error"] = df_column["Error"].str.replace("WRONG_", "", regex=False)

        df_errors = pd.concat([df_errors, df_column])

    df_errors["Stage_Error"] = df_errors["Stage"] + "_" + df_errors["Error"]

    sum_df = df_errors["Stage_Error"].value_counts().to_frame(name="#Error")
    sum_df["Stage_Error"] = sum_df.index

    sum_df[["Stage", "Error"]] = sum_df["Stage_Error"].str.split("_", expand=True)

    sum_df.reset_index(inplace=True)
    sum_df.drop(["index", "Stage_Error"], axis=1, inplace=True)

    print("::::::::\n", sum_df)
    return sum_df


# errors frequent in start/mid/end -> cannot do with ActionLogs not containing errors
# define stages as Entity, quantity, proportionality +ve, -ve, etc.
# input action log too to get stages from Target type column
def errors_per_stage(df, df_actionlog, plot=False):
    stages = df_actionlog["Target type"].tolist()
    stages = list(set(stages))
    stages = [x.capitalize() for x in stages]
    print(";:::", stages)
    # remove 'geen match' columns
    df = df.loc[:, ~df.columns.str.endswith("geen match")]
    dct_stages = {}
    for stage in stages:
        df_columns = df.loc[:, df.columns.str.startswith(stage)]
        df_columns.fillna("0", inplace=True)

        df_rows = df_columns

        columns = df_rows.columns.tolist()
        dct = {}
        lst = []

        for column in columns:
            for error in df_rows[column]:
                if error != "0":
                    if error.rfind("WRONG") != 0:
                        indices = [m.start() for m in re.finditer("WRONG", error)]
                        error_parts = [
                            error[i:j] for i, j in zip(indices, indices[1:] + [None])
                        ]
                        lst = lst + error_parts
                    else:
                        lst.append(error)

            for error in lst:
                if error in dct:
                    dct[error] += 1
                else:
                    dct[error] = 1
        if plot:
            dct_stages[stage] = dct
        else:
            dct = sorted(dct.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
            dct_stages[stage] = dct

    if plot:
        graph_lst = []
        for key in dct_stages:
            for error in dct_stages[key]:
                graph_lst.append([key, error, dct_stages[key][error]])
        df = pd.DataFrame(graph_lst, columns=["Stage", "Error", "#Error"])
        df.pivot(index="Error", columns="Stage", values="#Error").plot(
            kind="bar", rot=0
        )
        plt.show()
        print(df)
        return df

    else:
        print("Errors per stage:")
        for key in dct_stages:
            print(key, ": ")
            print(dct_stages[key])
        return dct_stages


# To get errors for actionlog tasks_level2_1
def get_errors_target_type(df, modelnum, target_type):
    # remove spaces in columns name
    df.columns = df.columns.str.replace(" ", "_")

    df["Index"] = [x for x in range(0, len(df))]
    df.set_index("Index", inplace=True)

    if len(df[df.Model_ID == modelnum]) != 0:
        index = df[df.Model_ID == modelnum].index[0]
        df_model = df.iloc[[index, index + 1, index + 2]]
        columns = [
            col
            for col in df.columns
            if (target_type.lower() in col.lower() and not "geen" in col.lower())
        ]

        columns_geen_match = [
            col
            for col in df.columns
            if target_type.lower() in col.lower() and "geen" in col.lower()
        ]

        df_model["Index"] = [x for x in range(0, len(df_model))]
        df_model.set_index("Index", inplace=True)

        count = 0
        errors = []
        for column in columns:
            if df_model.at[2, column] == "0":
                count += 1
                if type(df_model.at[1, column]) == str:
                    errors.append(df_model.at[1, column])

        geen_match_count = 0
        for column in columns_geen_match:
            if type(df_model.at[0, column]) != np.nan:
                geen_match_count += df_model.at[0, column]

        return (count, geen_match_count, errors)

    else:
        return (np.nan, np.nan, [np.nan])


def get_errors_level3(df, modelnum, task_name):
    if type(task_name) == str:
        # remove spaces in columns name
        df.columns = df.columns.str.replace(" ", "_")

        df["Index"] = [x for x in range(0, len(df))]
        df.set_index("Index", inplace=True)

        if len(df[df.Model_ID == modelnum]) != 0:
            index = df[df.Model_ID == modelnum].index[0]
            df_model = df.iloc[[index, index + 1, index + 2]]

            df_model["Index"] = [x for x in range(0, len(df_model))]
            df_model.set_index("Index", inplace=True)

            col_name = 0
            for col in df_model.columns:
                if type(df_model.iloc[0][col]) == str:
                    if df_model.iloc[0][col].lower() == task_name.lower():
                        col_name = col
                        pass

            if col_name == 0:
                return (np.nan, np.nan)
            else:
                correctness = df_model[col_name].iloc[2]
                error_type = df_model[col_name].iloc[1]

                return (correctness, error_type)
        else:
            return (np.nan, np.nan)
    else:
        return (np.nan, np.nan)