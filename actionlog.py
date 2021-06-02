import numpy as np
import pandas as pd
import csv
import statistics
import random
import matplotlib.pyplot as plt
from matplotlib import pyplot
from datetime import datetime
import re
import math
from modeldata import get_errors_target_type, get_errors_level3


# ------------------Turn xlsx to csv (correctly because using Excel conversions doesn't work)-------------------------
# Input: xlsx file name (and path) and desired output csv file name
def xlsx_to_csv(file, output):
    read_file = pd.read_excel(file)
    read_file.replace(
        to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"],
        value=["", ""],
        regex=True,
        inplace=True,
    )
    read_file.to_csv(output, index=None, header=True)


# --------------------------------------Merging data sets (xlsx files)-------------------------------------
# Input: list of xlsx files, and desired output file name (.csv), and if ModelData not ActionLog input True
def merge_datasets(files_lst, output, modeldata=False):
    lst = []
    for index, file in enumerate(files_lst):
        xlsx_to_csv(file, "output{}.csv".format(index))
        if modeldata:
            if index != 0:
                lst.append(pd.read_csv("output{}.csv".format(index)).drop(0))
            else:
                lst.append(pd.read_csv("output{}.csv".format(index)))
        else:
            lst.append(pd.read_csv("output{}.csv".format(index)))

    merged_dfs = pd.concat(lst)
    merged_dfs.to_csv(output)


# --------------------------------------Cleaning/Preparing Data-----------------------------------------------------------------

# change columns' data types as appropriate
# Input: ActionLog csv file name, and (errors=)True if ActionLog Arguements contain errors
def change_dtypes(csvfile, errors=False):
    # read file and make Moment column a datetype
    df = pd.read_csv(csvfile, parse_dates=["Moment"])  # error_bad_lines=False
    # drop rows/columns that are all empty
    df.dropna(how="all", inplace=True)
    # make User ID, Model ID, Action, and Target type category columns
    # and add columns with category codes for each
    df["User ID"] = df["User ID"].astype("category")
    # df["User ID cat"] = df["User ID"].cat.codes
    df["Model ID"] = df["Model ID"].astype("category")
    # df["Model ID cat"] = df["Model ID"].cat.codes
    df["Action"] = df["Action"].astype("category")
    # df["Action cat"] = df["Action"].cat.codes
    df["Target type"] = df["Target type"].astype("category")
    # df["Target type cat"] = df["Target type"].cat.codes

    # make rest of column string type
    df["Target"] = df["Target"].astype("string")
    df["Arguments"] = df["Arguments"].astype("string")

    # the below will run only if ActionLog contains data about errors (Arguments column)
    # make Arguments Summary column containing only error type
    if errors:

        def error_summary(x):
            if "fouten" in x:
                return (
                    x.split("]")[0][13:].replace('"', "").replace("WRONG_", " ").strip()
                )

        df["Arguments Summary"] = df["Arguments"].apply(lambda x: error_summary(x))

        # extra columns only present in 'My Data'
        df["Domain"] = df["Domain"].astype("string")
        df["Project"] = df["Project"].astype("string")
        df["Login"] = df["Login"].astype("string")
        df["Display name"] = df["Display name"].astype("string")
        df["Model name"] = df["Model name"].astype("string")

    return df


# To see which category has which code in a column
# Input: ActionLog DataFrame (output of change_dtypes())
def print_cat_codes(df, column_name):
    print(df[column_name].cat.categories)


# return list of Model IDs (from actionlog or modeldata)
def models_list(df):
    return df["Model ID"].unique().tolist()


# --------------------------------------Adding Measures-----------------------------------------------------------------
# Input: actionlog df
# Output: Add "Time (s)" column for each step in each model and an index for each column (first step time = 0)
def add_time_col(df):
    df_output = pd.DataFrame(
        columns=[
            "User ID",
            "Model ID",
            "Moment",
            "Action",
            "Target",
            "Target type",
            "Arguments",
            "Index",
            "Time_s",
        ]
    )
    for modelnum, df_model in df.groupby("Model ID"):
        df_model["Index"] = [x for x in range(0, len(df_model))]
        df_model = df_model.set_index("Index")
        df_model["Index"] = [x for x in range(0, len(df_model))]

        def get_time(index):
            if index == 0:
                return 0
            else:
                time = (
                    df_model.at[index, "Moment"] - df_model.at[index - 1, "Moment"]
                ) / np.timedelta64(1, "s")
                return time

        if len(df_model) == 1:
            df_model["Time_s"] = [0]
        else:
            df_model["Time_s"] = df_model["Index"].apply(lambda x: get_time(x))

        df_output = df_output.append(df_model, ignore_index=True)
    # there is an extra "Unnamed" column for some reason. To delete:
    df_output = df_output.loc[:, ~df_output.columns.str.contains("^Unnamed")]

    return df_output


# Run elbow then silhouette to decide on number of clusters
# Then run cluster with the number of clusters decided on
def cluster_time(df, n_clusters=0, elbow=False, silho=False):
    from numpy import unique
    from numpy import where
    from sklearn.cluster import KMeans
    from matplotlib import pyplot
    from sklearn.metrics import silhouette_score

    # make dataset suitable for clustering. referece: https://stackoverflow.com/a/28020783
    df = df[["Time_s"]]
    df_array = df.values

    # k-means clustering on dataset
    def cluster(data, num_clusters):
        # define the model
        model = KMeans(n_clusters=num_clusters)
        # fit the model
        model.fit(data)
        # assign a cluster to each example
        yhat = model.predict(data)
        # retrieve unique clusters
        clusters = unique(yhat)
        # create scatter plot for samples from each cluster
        ones = []
        for i in range(len(data)):
            ones.append(1)
        for cluster in clusters:
            # get row indexes for samples with this cluster
            row_ix = where(yhat == cluster)
            # create scatter of these samples
            pyplot.scatter(data, ones)  # because data is 1D
        # show the plot
        print("::Cluster centers::")
        print(model.cluster_centers_)
        pyplot.show()

    def sse(data):
        # elbow method (look for elbow point):
        sse = []
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(data)
            sse.append(kmeans.inertia_)

        plt.style.use("fivethirtyeight")
        plt.plot(range(1, 11), sse)
        plt.xticks(range(1, 11))
        plt.xlabel("Number of Clusters")
        plt.ylabel("SSE")
        plt.show()

    def silhouette(data):
        # silhouette coefficient method (look for max point):
        silhouette_coefficients = []

        for k in range(2, 11):
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(data)
            score = silhouette_score(data, kmeans.labels_)
            silhouette_coefficients.append(score)

        plt.style.use("fivethirtyeight")
        plt.plot(range(2, 11), silhouette_coefficients)
        plt.xticks(range(2, 11))
        plt.xlabel("Number of Clusters")
        plt.ylabel("Silhouette Coefficient")
        plt.show()

    if elbow:
        sse(df_array)
    elif silho:
        silhouette(df_array)
    elif n_clusters == 0:
        print("Please enter the number of clusters")
    else:
        cluster(df_array, n_clusters)


# Input: action log (after "Time_s" is added (add_time_col))
# Output:
# if plot_scatter: returns scatter plot of Time_s against Action_Type (Action + Target type)
# if return_quartiles: returns dictionary with Action_Type as keys and list: [Q1, median, Q3, IQR, min_outliers, max_outliers]
# if IQR: uses IQR outliers method to find breaks. Breaks are marked by changing their Time_s to 0 and adding Break duration column
# if normal: check if Time_s distribution is normal, if yes, outliers are > 3*standard deviation
# if forest:uses Isolation Forest method to find anomilies
# if cutoff_time = some seconds: outliers will be anything above that (can be based on cluster value)
# if print: add informative print statements, otherwise just returns df
def find_breaks(
    df,
    plot_scatter=False,
    return_quartiles=False,
    IQR=False,
    IQR_all=False,
    forest=False,
    cutoff_time=None,
    normal=False,
    printed=False,
):
    df["Action"] = df["Action"].astype("string")
    df["Target type"] = df["Target type"].astype("string")
    df["Action_Type"] = df["Action"] + "_" + df["Target type"]

    df["Index"] = [x for x in range(0, len(df))]

    if plot_scatter:
        df = df[["Action_Type", "Time_s"]]
        df_array = df.values
        pyplot.scatter(df_array[:, 0], df_array[:, 1])
        plt.grid()
        plt.xticks(rotation=30, horizontalalignment="right", fontsize="xx-small")
        pyplot.show()
    elif IQR or return_quartiles:
        quartiles_dct = {}
        for action, df_action in df.groupby("Action_Type"):
            # quartiles = [Q1, median, Q3, IQR, min_outliers, max_outliers]
            quartiles = df_action.Time_s.quantile([0.25, 0.5, 0.75]).tolist()
            quartiles.append(quartiles[2] - quartiles[0])
            quartiles.append(quartiles[0] - 1.5 * quartiles[3])
            quartiles.append(quartiles[2] + 1.5 * quartiles[3])
            quartiles_dct[action] = quartiles

        def break_duration(index):
            max_time = quartiles_dct[df.at[index, "Action_Type"]][-1]
            if df.at[index, "Time_s"] > max_time:
                return df.at[index, "Time_s"]
            else:
                return 0

        times = []

        def check_break(index):
            max_time = quartiles_dct[df.at[index, "Action_Type"]][-1]
            if df.at[index, "Time_s"] > max_time:
                times.append((df.at[index, "Time_s"], index))
                return 0
            else:
                return df.at[index, "Time_s"]

        df["Break_Duration"] = df["Index"].apply(lambda x: break_duration(x))
        df["Time_s"] = df["Index"].apply(lambda x: check_break(x))

        if return_quartiles:
            for key in quartiles_dct:
                if printed:
                    print(key, "::")
                    print(quartiles_dct[key])
            return quartiles_dct
        else:
            if printed:
                times.sort()
                print("Times that are breaks and their index", times)
                print(df[df["Break_Duration"] != 0])
                print(":::Min time considered a break is: ", times[0][0], "seconds")
            return df
    elif IQR_all:
        # quartiles = [Q1, median, Q3, IQR, min_outliers, max_outliers]
        quartiles = df.Time_s.quantile([0.25, 0.5, 0.75]).tolist()
        quartiles.append(quartiles[2] - quartiles[0])
        quartiles.append(quartiles[0] - 1.5 * quartiles[3])
        quartiles.append(quartiles[2] + 1.5 * quartiles[3])

        def break_duration(index):
            max_time = quartiles[-1]
            if df.at[index, "Time_s"] > max_time:
                return df.at[index, "Time_s"]
            else:
                return 0

        times = []

        def check_break(index):
            max_time = quartiles[-1]
            if df.at[index, "Time_s"] > max_time:
                times.append((df.at[index, "Time_s"], index))
                return 0
            else:
                return df.at[index, "Time_s"]

        df["Break_Duration"] = df["Index"].apply(lambda x: break_duration(x))
        df["Time_s"] = df["Index"].apply(lambda x: check_break(x))

        if printed:
            times.sort()
            print("Times that are breaks and their index", times)
            print(df[df["Break_Duration"] != 0])
            print(":::Min time considered a break is: ", times[0][0], "seconds")
        return df
    elif forest:
        from sklearn.ensemble import IsolationForest

        df_array = df[["Time_s"]].values
        # clf = IsolationForest(max_samples=100, random_state=1, contamination="auto")
        clf = IsolationForest(random_state=0)
        # preds is an array of predictions for each data point
        # If the result is -1, it means that this specific data point is an outlier
        preds = clf.fit_predict(df_array)

        def break_duration(index):
            pred = preds[index]
            if pred == -1:
                return df.at[index, "Time_s"]
            else:
                return 0

        times = []

        def check_break(index):
            pred = preds[index]
            if pred == -1:
                times.append((df.at[index, "Time_s"], index))
                return 0
            else:
                return df.at[index, "Time_s"]

        df["Break_Duration"] = df["Index"].apply(lambda x: break_duration(x))
        df["Time_s"] = df["Index"].apply(lambda x: check_break(x))

        if printed:
            times.sort()
            print("Times that are breaks and their index", times)
            print(df[df["Break_Duration"] != 0])
            print(":::Min time considered a break is: ", times[0][0], "seconds")
        return df
    elif cutoff_time:

        def break_duration(index):
            max_time = cutoff_time
            if df.at[index, "Time_s"] > max_time:
                return df.at[index, "Time_s"]
            else:
                return 0

        times = []

        def check_break(index):
            max_time = cutoff_time
            if df.at[index, "Time_s"] > max_time:
                times.append((df.at[index, "Time_s"], index))
                return 0
            else:
                return df.at[index, "Time_s"]

        df["Break_Duration"] = df["Index"].apply(lambda x: break_duration(x))
        df["Time_s"] = df["Index"].apply(lambda x: check_break(x))

        if printed:
            times.sort()
            print("Times that are breaks and their index", times)
            print(df[df["Break_Duration"] != 0])
            print(":::Min time considered a break is: ", times[0][0], "seconds")
        return df
    elif normal:
        # first check if Time_s for each Action_Type is actually a normal distribution
        # Kolmogorov-Smirnov test:
        from scipy.stats import kstest, norm

        for action, df_action in df.groupby("Action_Type"):
            df_action = df_action[["Time_s"]]
            df_array = df_action.values

            ks_statistic, p_value = kstest(df_array, "norm")
            # print(ks_statistic, p_value)
            if p_value > 0.05:
                print(action, "::normal")
            else:
                print(action, ":: not normal")
        # if distribution is normal, any point that is more than 3 times the standard deviation
        # is very likely to be an outlier
        # https://towardsdatascience.com/5-ways-to-detect-outliers-that-every-data-scientist-should-know-python-code-70a54335a623
        # -> distribution is not normal


# Input: action log
# Output: tasks defined by Actions (create/modify/delete), their #Steps and time
# model=model number: return df of tasks for that model
def tasks_level1(df, model=0, printed=False):
    data = []
    for modelnum, df_model in df.groupby("Model ID"):
        if not df_model.empty:
            i = 0
            for action, df_action in df_model.groupby("Action"):
                row = []
                row.append(df_model["User ID"].iloc[0])
                row.append(modelnum)
                row.append(i)
                i += 1
                row.append(action)
                row.append(df_action["Time_s"].sum())
                row.append(len(df_action))

                data.append(row)

    df_tasks = pd.DataFrame(
        data,
        columns=[
            "User ID",
            "Model ID",
            "Task#",
            "Task",
            "Time_s",
            "#Steps",
        ],
    )

    # drop the create model one (it's not a task)
    df_tasks = df_tasks[df_tasks.Task != "newmodel"]

    if model == 0:
        if printed:
            print(df_tasks)
            return df_tasks
        else:
            return df_tasks
    else:
        if printed:
            print(df_tasks[df_tasks["Model ID"] == model])
        return df_tasks[df_tasks["Model ID"] == model]


# # Input: action log
# # Output: tasks defined by Actions and Target type (create entity/modify quantity/delete entity), their #Steps and time
# model=model number: return df of tasks for that model
def tasks_level2(df, model=0, printed=False):
    df["Action"] = df["Action"].astype("string")
    df["Target type"] = df["Target type"].astype("string")
    df["Action_Type"] = df["Action"] + "_" + df["Target type"]

    data = []
    for modelnum, df_model in df.groupby("Model ID"):
        if not df_model.empty:
            i = 0
            for action_type, df_action_type in df_model.groupby("Action_Type"):
                row = []
                row.append(df_model["User ID"].iloc[0])
                row.append(modelnum)
                row.append(i)
                i += 1
                row.append(action_type)
                row.append(df_action_type["Time_s"].sum())
                row.append(len(df_action_type))

                data.append(row)

    df_tasks = pd.DataFrame(
        data,
        columns=[
            "User ID",
            "Model ID",
            "Task#",
            "Task",
            "Time_s",
            "#Steps",
        ],
    )

    # drop the create model one (it's not a task)
    df_tasks = df_tasks[df_tasks.Task != "newmodel_model"]
    df_tasks = df_tasks[df_tasks.Task != "copymodel_model"]

    if model == 0:
        if printed:
            print(df_tasks)
            return df_tasks
        else:
            return df_tasks
    else:
        if printed:
            print(df_tasks[df_tasks["Model ID"] == model])
        return df_tasks[df_tasks["Model ID"] == model]


# # Input: action log, model data, modelnum if want one model details
# # Output: tasks defined by Target type (entity/quantity/configuration), their #Steps, time, #Errors, Error Types, #create, #modify, ...
def tasks_level2_1(df, df_modeldata, model=0, printed=False):
    actions = list(df["Action"].unique())
    # drop the 'newmodel' and 'copymodel'
    for action in actions:
        if "model" in action:
            actions.remove(action)

    # remove spaces in columns name
    df.columns = df.columns.str.replace(" ", "_")

    data = []
    for modelnum, df_model in df.groupby("Model_ID"):
        if not df_model.empty:
            i = 0
            for target_type, df_target_type in df_model.groupby("Target_type"):
                row = []
                row.append(df_model["User_ID"].iloc[0])
                row.append(modelnum)
                row.append(i)
                i += 1
                row.append(target_type)
                row.append(df_target_type["Time_s"].sum())
                row.append(len(df_target_type))

                # errors:
                (count, geen_match_count, errors) = get_errors_target_type(
                    df_modeldata, modelnum, target_type
                )

                row.append(count)

                if errors != []:
                    errors_dct = {x: errors.count(x) for x in errors}
                    errors_string = str(errors_dct)
                    errors_string = errors_string.strip("{")
                    errors_string = errors_string.strip("}")
                    row.append(errors_string)
                    missing = count - len(errors)
                    row.append(missing)
                else:
                    row.append(np.nan)
                    missing = count
                    row.append(missing)

                row.append(geen_match_count)

                # actions:
                for action in actions:
                    row.append((df_target_type.Action == action).sum())

                data.append(row)

    action_columns = []
    for action in actions:
        action_columns.append("#" + action)

    df_tasks = pd.DataFrame(
        data,
        columns=[
            "User ID",
            "Model ID",
            "Task#",
            "Task",
            "Time_s",
            "#Steps",
            "#Errors",
            "Error types",
            "#Missing_elements",
            "#Geen_match",
        ]
        + action_columns,
    )

    # drop the create model one (it's not a task)
    df_tasks = df_tasks[df_tasks.Task != "model"]

    # undo remove spaces in columns name
    df.columns = df.columns.str.replace("_", " ")
    df.rename(columns={"Time s": "Time_s"}, inplace=True)

    if model == 0:
        if printed:
            print(df_tasks)
            return df_tasks
        else:
            return df_tasks
    else:
        if printed:
            with pd.option_context(
                "display.max_rows", None, "display.max_columns", None
            ):
                print(df_tasks[df_tasks["Model ID"] == model])
        return df_tasks[df_tasks["Model ID"] == model]


# # Input: action log, model data
# # Output: tasks defined by Target (IDs) (entity:Auto/entity:Auto-quantity:Snelheid),
# and for proportionality +ve/-ve & configuration get from/to from Arguments
# their #Steps, time, Correct/Error (match error with last step in task), Error Type, #create, #modify, ...
def tasks_level3(df, df_modeldata, model=0, printed=False):
    actions = list(df["Action"].unique())
    # drop the 'newmodel' and 'copymodel'
    for action in actions:
        if "model" in action:
            actions.remove(action)

    target_types = list(df["Target type"].unique())

    def split(text):
        if "|" in text:
            lst = []
            lst0 = text.split("|")
            for item in lst0:
                lst += item.split(":")
            return lst
        elif ":" in text:
            return text.split(":")
        else:
            return text

    def return_ids(text, return_lst=False):
        lst = split(text)
        if return_lst:
            return lst
        else:
            r = re.compile("^[n][0-9]+")
            lst_ids = list(filter(r.match, lst))
            return "_".join(lst_ids)

    df["ID"] = df["Target"].apply(lambda x: return_ids(x))

    df["Index"] = [x for x in range(0, len(df))]
    df.set_index("Index", inplace=True)

    data = []
    for modelnum, df_model in df.groupby("Model ID"):
        if not df_model.empty:
            i = 0
            for id, df_id in df_model.groupby("ID"):
                row = []
                row.append(df_model["User ID"].iloc[0])
                row.append(modelnum)
                row.append(i)  # task number
                i += 1
                row.append(df_id.index.tolist())  # indices
                #### get task name ####
                last_target = df_id.iloc[-1]["Target"]
                target_lst = return_ids(last_target, return_lst=True)
                r = re.compile("^[n][0-9]+")
                lst_ids = list(filter(r.match, split(last_target)))
                target_minus_ids = [item for item in target_lst if item not in lst_ids]
                row.append(":".join(target_minus_ids))
                ######################
                row.append(df_id["Time_s"].sum())
                row.append(len(df_id))

                # errors:
                ### make task name format like in model data ###
                if "derivative_value" in last_target:
                    task_name = np.nan
                elif "entity" in last_target or "quantity" in last_target:
                    names_lst = [
                        item for item in target_minus_ids if item not in target_types
                    ]
                    task_name = ":".join(names_lst)
                elif "configuration" in last_target:
                    names_lst = [
                        item for item in target_minus_ids if item not in target_types
                    ]
                    task_name = names_lst[0]
                    last_argument = df_id.iloc[-1]["Arguments"]
                    splits = last_argument.split('"')
                    task_name += "("
                    task_name += splits[splits.index("_fromroute") + 2].split(":")[-1]
                    task_name += " - "
                    task_name += splits[splits.index("_toroute") + 2].split(":")[-1]
                    task_name += ")"
                elif "proportionality_positive" in last_target:
                    task_name = "proportionality_positive("
                    last_argument = df_id.iloc[-1]["Arguments"]
                    splits = last_argument.split('"')

                    task_name += (
                        splits[splits.index("_fromroute") + 2]
                        .split("|")[0]
                        .split(":")[-1]
                    )
                    task_name += ":"
                    task_name += (
                        splits[splits.index("_fromroute") + 2]
                        .split("|")[1]
                        .split(":")[-1]
                    )
                    task_name += " - "
                    task_name += (
                        splits[splits.index("_toroute") + 2]
                        .split("|")[0]
                        .split(":")[-1]
                    )
                    task_name += ":"
                    task_name += (
                        splits[splits.index("_toroute") + 2]
                        .split("|")[1]
                        .split(":")[-1]
                    )
                    task_name += ")"
                elif "proportionality_negative" in last_target:
                    task_name = "proportionality_negative("
                    last_argument = df_id.iloc[-1]["Arguments"]
                    splits = last_argument.split('"')
                    task_name += (
                        splits[splits.index("_fromroute") + 2]
                        .split("|")[0]
                        .split(":")[-1]
                    )
                    task_name += ":"
                    task_name += (
                        splits[splits.index("_fromroute") + 2]
                        .split("|")[1]
                        .split(":")[-1]
                    )
                    task_name += " - "
                    task_name += (
                        splits[splits.index("_toroute") + 2]
                        .split("|")[0]
                        .split(":")[-1]
                    )
                    task_name += ":"
                    task_name += (
                        splits[splits.index("_toroute") + 2]
                        .split("|")[1]
                        .split(":")[-1]
                    )
                    task_name += ")"
                else:
                    task_name = np.nan
                ################################################
                (correctness, error_type) = get_errors_level3(
                    df_modeldata, modelnum, task_name
                )

                row.append(task_name)
                row.append(correctness)
                row.append(error_type)

                # actions:
                for action in actions:
                    row.append((df_id.Action == action).sum())

                data.append(row)

    action_columns = []
    for action in actions:
        action_columns.append("#" + action)

    df_tasks = pd.DataFrame(
        data,
        columns=[
            "User ID",
            "Model ID",
            "Task#",
            "Indices",
            "Task",
            "Time_s",
            "#Steps",
            "Task as in ModelData",
            "Correct/Wrong",
            "Error type",
        ]
        + action_columns,
    )

    # drop the create model one (it's not a task)
    df_tasks = df_tasks[df_tasks.Task != "m:o:d:e:l"]

    if model == 0:
        if printed:
            print(df_tasks)
            return df_tasks
        else:
            return df_tasks
    else:
        if printed:
            with pd.option_context(
                "display.max_rows", None, "display.max_columns", None
            ):
                print(df_tasks[df_tasks["Model ID"] == model])
        return df_tasks[df_tasks["Model ID"] == model]


# # Input: action log df, df_learning, df_modeldata, tasks level 1, 2, 2.1, or 3
# # Output: summary of Tasks (#tasks, total time, #errors, ...) with learning measures from modeldata
def tasks_into_learning(df, df_learning, df_modeldata, level, printed=False):
    models = models_list(df_learning)

    def get_tasks(model):
        if model in df["Model ID"].values:
            if level == 1:
                df_tasks = tasks_level1(df, model)
            elif level == 2:
                df_tasks = tasks_level2(df, model)
            elif level == 2.1:
                df_tasks = tasks_level2_1(df, df_modeldata, model)
            elif level == 3:
                df_tasks = tasks_level3(df, df_modeldata, model)

            # lst = [(total) #Tasks, Time_s, #Steps, #create, #modify, #delete, #undo, #redo]
            lst = [len(df_tasks), df_tasks["Time_s"].sum(), df_tasks["#Steps"].sum()]
            return lst
        else:
            return [np.nan]

    tasks = []
    time = []
    steps = []
    breaks = []
    breaks_durations = []
    for model in models:
        tasks_info = get_tasks(model)
        if len(tasks_info) == 1:  # nan
            tasks.append(tasks_info[0])
            time.append(tasks_info[0])
            steps.append(tasks_info[0])
        else:
            tasks.append(tasks_info[0])
            time.append(tasks_info[1])
            steps.append(tasks_info[2])

        try:
            breaks_durations.append(
                df.loc[(df["Model ID"] == model) & (df["Break Duration"] != 0)][
                    "Break Duration"
                ].tolist()
            )
        except KeyError:
            breaks_durations.append(
                df.loc[(df["Model ID"] == model) & (df["Break_Duration"] != 0)][
                    "Break_Duration"
                ].tolist()
            )
        breaks.append(len(breaks_durations[-1]))

    df_learning["#Tasks"] = tasks
    df_learning["Time_s"] = time
    df_learning["#Steps"] = steps
    df_learning["#Breaks"] = breaks
    df_learning["Break_durations"] = breaks_durations

    ### add measure relating score to time (euclidean distance)###
    from scipy.spatial import distance

    time_median_score = df_learning[
        df_learning["%Score"] >= df_learning["%Score"].max()
    ]["Time_s"].median()
    # sometimes no one in class gets 100 so use max grade instead
    time_median_correct = df_learning[
        df_learning["%Correct"] >= df_learning["%Correct"].max()
    ]["Time_s"].median()
    optimal_speed_correct = time_median_correct / 100
    optimal_speed_score = time_median_score / 100
    print(
        "score time median: ",
        time_median_score,
        "\n" ">= grade (max): ",
        df_learning["%Score"].max(),
        "\n" "# of students considered for median: ",
        len(df_learning[df_learning["%Score"] >= df_learning["%Score"].max()]),
        "\n" "\n" "correct time median: ",
        time_median_correct,
        "\n" ">= grade (max): ",
        df_learning["%Correct"].max(),
        "\n" "# of students considered for median: ",
        len(df_learning[df_learning["%Correct"] >= df_learning["%Correct"].max()]),
        "\n" "\n" "optimal speed correct: ",
        optimal_speed_correct,
    )

    df_learning["Index"] = [x for x in range(0, len(df_learning))]
    df_learning = df_learning.set_index("Index")
    df_learning["Index"] = [x for x in range(0, len(df_learning))]

    def get_distance_score(index):
        if not math.isnan(df_learning.at[index, "%Score"]) and not math.isnan(
            df_learning.at[index, "Time_s"]
        ):
            return distance.euclidean(
                [100, time_median_score],
                [df_learning.at[index, "%Score"], df_learning.at[index, "Time_s"]],
            ) / 10 ** (
                len(str(int(time_median_score))) - 1
            )  # divide by 1000 to keep numbers small
        else:
            return np.nan

    def get_distance_correct(index):
        if not math.isnan(df_learning.at[index, "%Correct"]) and not math.isnan(
            df_learning.at[index, "Time_s"]
        ):
            return distance.euclidean(
                [100, time_median_correct],
                [df_learning.at[index, "%Correct"], df_learning.at[index, "Time_s"]],
            ) / 10 ** (
                len(str(int(time_median_correct))) - 1
            )  # divide by 1000 to keep numbers small
        else:
            return np.nan

    df_learning["Euclidean_Distance_Score"] = df_learning["Index"].apply(
        lambda x: get_distance_score(x)
    )

    df_learning["Euclidean_Distance_Correct"] = df_learning["Index"].apply(
        lambda x: get_distance_correct(x)
    )

    df_learning["Time_s"] = df_learning["Time_s"].replace(0, np.nan)

    df_learning["Relative_Speed"] = optimal_speed_correct - (
        df_learning["Time_s"] / df_learning["%Correct"]
    )

    df_learning["Relative_Speed_Score"] = optimal_speed_score - (
        df_learning["Time_s"] / df_learning["%Score"]
    )

    df_learning.drop(["Index"], axis=1, inplace=True)
    ##########################################

    # # check if Score/Time is a normal distribution --> it's not
    # # if it is, could use confidence interval or standard deviation instead of euclidean distance
    # df_learning["Score/Time"] = df_learning["%Score"] / df_learning["Time_s"]
    # # Kolmogorov-Smirnov test:
    # from scipy.stats import kstest, norm

    # df_score = df_learning[["Score/Time"]]
    # df_array = df_score.values

    # ks_statistic, p_value = kstest(df_array, "norm")
    # # print(ks_statistic, p_value)
    # if p_value > 0.05:
    #     print("::normal")
    # else:
    #     print(":: not normal")

    if printed:
        print(df_learning)
    return df_learning


# per model (tasks divided by time intervals)
# n is the number of divisions in time wanted (e.g. 4 or 10)
def into_tasks_time(df, n, model=0, printed=False, dct=False):
    df["Action"] = df["Action"].astype("string")
    df["Target type"] = df["Target type"].astype("string")
    df["Action+Target type"] = df["Action"] + "_" + df["Target type"]

    if dct or model != 0:
        user_models = {}
    else:
        df_output = pd.DataFrame(
            columns=["Model ID", "Task", "Time", "#Steps", "Actions+Target types"]
        )

    # for each model
    for modelnum, df_model in df.groupby("Model ID"):
        x = df_model.iloc[-1, df_model.columns.get_loc("Moment")]
        y = df_model.iloc[0, df_model.columns.get_loc("Moment")]
        total_time = x - y
        time_diff = total_time / n
        data = []
        # one row in data:
        # [ Task, Time, #Steps, [Actions+Target types]]

        time_start = df_model["Moment"].iloc[0]
        for i in range(1, n + 1):
            time_end = time_start + time_diff

            df_time = df_model[df_model["Moment"].between(time_start, time_end)]

            time_start = time_end

            if dct or model != 0:
                row = ["Task {}".format(i)]
            else:
                row = [modelnum, "Task {}".format(i)]

            if len(df_time) >= 1:
                row.append(
                    pd.to_timedelta(
                        df_time["Moment"].iloc[-1] - df_time["Moment"].iloc[0]
                    )
                )
                row.append(len(df_time))  # Steps

                actions = df_time["Action+Target type"].value_counts().index.tolist()
                count = df_time["Action+Target type"].value_counts().tolist()
                counts = [str(i) for i in count]
                actions_targets = [i + "x " + j for i, j in zip(counts, actions)]
                row.append(actions_targets)

            data.append(row)

        if dct or model != 0:
            df_tasks = pd.DataFrame(
                data, columns=["Task", "Time", "#Steps", "Actions+Target types"]
            )
            user_models[modelnum] = df_tasks
        else:
            df_tasks = pd.DataFrame(
                data,
                columns=["Model ID", "Task", "Time", "#Steps", "Actions+Target types"],
            )
            df_output = df_output.append(df_tasks, ignore_index=True)

    if model == 0:
        if dct:
            if printed:
                for key in user_models:
                    print(key, ":")
                    print(user_models[key])
            return user_models
        else:
            if printed:
                print(df_output)
            return df_output
    else:
        if printed:
            print(user_models[model])
        return user_models[model]


# --------------------------------------Modeling Engagement/Motivation----------------------------------------------------
# Input: df_actionlog
# Output: add column to df_learning with 1s or 0s for whether student engaged in off_task behavior
# (another column on why?)
def off_task_1(df, printed=False):
    # df["Action"] = df["Action"].astype("string")
    # df["Target type"] = df["Target type"].astype("string")
    # df["Action_Type"] = df["Action"] + "_" + df["Target type"]

    # df["Action_Type"] = df["Action_Type"].astype("category")
    # df["Action_Type cat"] = df["Action_Type"].cat.codes

    df["Index"] = [x for x in range(0, len(df))]

    from sklearn.ensemble import IsolationForest

    df_array = df[["Time_s"]].values
    # clf = IsolationForest(max_samples=100, random_state=1, contamination="auto")
    clf = IsolationForest(random_state=0)
    # preds is an array of predictions for each data point
    # If the result is -1, it means that this specific data point is an outlier
    preds = clf.fit_predict(df_array)

    def break_duration(index):
        pred = preds[index]
        if pred == -1:
            time = df.at[index, "Time_s"]
            if time not in times:
                times.append(time)
            return time
        else:
            return 0

    times = []

    df["Break_Duration_offtask"] = df["Index"].apply(lambda x: break_duration(x))

    if printed:
        times.sort()
        print("Times that are breaks and their index", times)
        print(df[df["Break_Duration"] != 0])
        if times != []:
            print(
                ":::Min times considered a break is: ",
                times[0],
                ", ",
                times[1],
                ", ",
                times[2],
                ", ",
                times[3],
                "seconds",
            )
        else:
            print(":::Min time considered a break is: there is no breaks")
        # print(min(256, len(df_array)))    #auto n_samples
    return df


def off_task_2(df, printed=False):
    import seaborn as sns
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    df["Index"] = [x for x in range(0, len(df))]

    # df_actionlog = df[df["Break_Duration"] != 0]["Break_Duration"].dropna().values.reshape(-1, 1)
    df_actionlog = df[["Model ID", "Break_Duration"]].dropna().values
    df_principal = pd.DataFrame(df_actionlog)
    df_principal.columns = ["Model ID", "Break_Duration"]
    ones = []
    for i in range(len(df_principal)):
        ones.append(1)
    df_principal["Ones"] = ones

    clustering_kmeans = KMeans(n_clusters=2, precompute_distances="auto", n_jobs=-1)
    preds = clustering_kmeans.fit_predict(
        df_principal["Break_Duration"].values.reshape(-1, 1)
    )
    df_principal["clusters"] = preds
    print(df_principal)

    kmeans_silhouette = silhouette_score(
        df_principal["Break_Duration"].values.reshape(-1, 1), clustering_kmeans.labels_
    ).round(2)
    print("kmeans silhouette coefficient: ", kmeans_silhouette)

    sns.scatterplot(
        x="Break_Duration",
        y="Ones",
        hue=df_principal["clusters"],
        palette="colorblind",
        data=df_principal[["Break_Duration", "Ones"]],
    )
    # plt.title('K-means Clustering with 2 dimensions')
    plt.show()

    times = []

    def break_duration(index):
        pred = preds[index]
        time = df.at[index, "Time_s"]
        if pred == 1:
            if time not in times:
                times.append(time)
            return time
        else:
            return 0

    df["cluster"] = preds
    df["Break_Duration_offtask"] = df["Index"].apply(lambda x: break_duration(x))

    if printed:
        times.sort()
        print("Times that are breaks and their index", times)
        print(df[df["Break_Duration_offtask"] != 0])
        if times != []:
            print(
                ":::Min time considered a break is: ",
                times[0],
                ", ",
                times[1],
                ", ",
                times[2],
                ", ",
                times[3],
                "seconds",
            )
        else:
            print(":::Min time considered a break is: there is no breaks")
    return df


# Output: add column to df_learning with 1s or 0s for whether student's mastery > performance avoidance goals or if we don't know
# Scale 1 to 5 (bc, be, ac, ae, db, ad)
# (another column on why?)
def goal_motivation_1(df_actionlog, df_learning, df_modeldata):
    df = tasks_into_learning(
        df_actionlog, df_learning, df_modeldata, level=1
    )  # doesn't matter which level

    df["Index"] = [x for x in range(0, len(df))]
    df = df.set_index("Index")
    df["Index"] = [x for x in range(0, len(df))]

    df_struggling = df.loc[(df["%Correct"] >= 35) & (df["%Correct"] <= 65)]

    quartiles = df_struggling.Relative_Speed.quantile([0.25, 0.5, 0.75]).tolist()

    def get_motivation_level(index):
        if 35 <= df.at[index, "%Correct"] <= 65:
            a = b = c = d = e = False

            if df.at[index, "%Missing"] >= (100 - df.at[index, "%Correct"]) / 2:
                a = True
            else:
                b = True

            if df.at[index, "Relative_Speed"] <= quartiles[0]:
                c = True
            elif df.at[index, "Relative_Speed"] >= quartiles[2]:
                d = True
            elif quartiles[0] < df.at[index, "Relative_Speed"] < quartiles[2]:
                e = True

            if a == True and d == True:
                return 6
            elif d == True and b == True:
                return 5
            elif a == True and e == True:
                return 4
            elif a == True and c == True:
                return 3
            elif b == True and e == True:
                return 2
            elif b == True and c == True:
                return 1
        else:
            return np.nan

    df["Goal motivation_correct"] = df["Index"].apply(lambda x: get_motivation_level(x))

    df.drop(["Index"], axis=1, inplace=True)

    return df


def goal_motivation_2(df_actionlog, df_learning, df_modeldata):
    df = tasks_into_learning(
        df_actionlog, df_learning, df_modeldata, level=1
    )  # doesn't matter which level

    df["Index"] = [x for x in range(0, len(df))]
    df = df.set_index("Index")
    df["Index"] = [x for x in range(0, len(df))]

    quartiles = df.Relative_Speed_Score.quantile([0.25, 0.5, 0.75]).tolist()

    def get_motivation_level(index):
        a = b = c = d = e = False

        if df.at[index, "%Missing"] >= df.at[index, "%Wrong"]:
            a = True
        else:
            b = True

        if df.at[index, "Relative_Speed_Score"] <= quartiles[0]:
            c = True
        elif df.at[index, "Relative_Speed_Score"] >= quartiles[2]:
            d = True
        elif quartiles[0] < df.at[index, "Relative_Speed_Score"] < quartiles[2]:
            e = True

        if a == True and d == True:
            return "aii"
        elif d == True and b == True:
            return "bii"
        elif a == True and e == True:
            return "aiii"
        elif a == True and c == True:
            return "ai"
        elif b == True and e == True:
            return "biii"
        elif b == True and c == True:
            return "bi"
        else:
            return np.nan

    df["Goal_motivation_correct"] = df["Index"].apply(lambda x: get_motivation_level(x))

    df["Goal_motivation_correct"] = df["Goal_motivation_correct"].astype("category")
    df["Goal_motivation_correct cat"] = df["Goal_motivation_correct"].cat.codes

    df.drop(["Index"], axis=1, inplace=True)

    return df


# # --------------------------------------Helper Functions-----------------------------------------------------------------


def correlations(df, var1, vars):
    x = df[var1]
    for var in vars:
        y = df[var]
        print("Correlation between {} and {}".format(var1, var))
        print("Pearson correlation = ", x.corr(y, method="pearson"))
        print("Spearman correlation = ", x.corr(y, method="spearman"))


def to_xlsx(df, filename):
    df.to_excel(filename)


# # --------------------------------------Removing wrong data-----------------------------------------------------------------
# Input: actionlog df
# Output: return list of models after remove all the extra daphne schippers models
# If remove_list=True, return list of models to be removed (to be used in model data)
def remove_daphne_schippers(df, remove_list=False):
    lst = models_list(df)
    remove = []
    # for each model
    for modelnum, df_model in df.groupby("Model ID"):
        for value in df_model["Target"]:
            if "Daphne Schippers" in value:
                remove.append(modelnum)
                break

    for model in remove:
        indices = df[df["Model ID"] == model].index
        df.drop(indices, inplace=True)

    if remove_list:
        return remove
    else:
        return df