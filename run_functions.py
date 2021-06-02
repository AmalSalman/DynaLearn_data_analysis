from seaborn import palettes
from actionlog import *
from modeldata import *
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import statistics

# Input: list of two lists of xlsx files (set), first: actionlog, second: modeldata
# Output: dfs list, with df_actionlog, df_learning, df_modeldata
def make_dfs(xlsx_lst):
    actionlog_file = "actionlog_file.csv"
    merge_datasets(xlsx_lst[0], actionlog_file)

    modeldata_file = "modeldata_file.csv"
    merge_datasets(xlsx_lst[1], modeldata_file, True)

    df_actionlog = change_dtypes(actionlog_file)

    df_learning = measure_learning(modeldata_file)

    df_modeldata = pd.read_csv(modeldata_file)
    df_modeldata.drop(df_modeldata.columns[0], axis=1, inplace=True)

    # remove Daphne Schippers from actionlog
    # and return list of removed models
    removed_daphne = remove_daphne_schippers(df_actionlog, remove_list=True)
    # remove Daphne Schippers from df_learning and df_modeldata
    for model in removed_daphne:
        df_learning = df_learning[df_learning["Model ID"] != model]
        df_modeldata = df_modeldata[df_modeldata["Model ID"] != model]

    # remove duplicate models leaving one with highest %Correct from df_learning
    # and return list of removed models
    removed_duplicates = remove_duplicates(df_learning, remove_list=True)
    df_learning = remove_duplicates(df_learning)
    # remove duplicates from df_actionlog and df_modeldata:
    for model in removed_duplicates:
        df_actionlog = df_actionlog[df_actionlog["Model ID"] != model]
        # remove spaces in columns name
        df_modeldata.columns = df_modeldata.columns.str.replace(" ", "_")
        index = df_modeldata[df_modeldata.Model_ID == model].index[0]
        df_modeldata.drop([index, index + 1, index + 2], inplace=True)
        # undo remove spaces in columns name
        df_modeldata.columns = df_modeldata.columns.str.replace("_", " ")

    # Add "Time (s)" column to actionlog
    df_actionlog = add_time_col(df_actionlog)

    # Remove "breaks" from Time_s
    df_actionlog = find_breaks(df_actionlog, forest=True)

    return [df_actionlog, df_modeldata, df_learning]


# return dfs for one of the six sets
def run_set(n):
    if n == 1:
        code = "S6587 L165317 K165317"
        dfs = make_dfs(
            [
                [
                    "Set {}/Data/ActionLog data ".format(n) + code + "a.xlsx",
                    "Set {}/Data/ActionLog data ".format(n) + code + "b.xlsx",
                ],
                [
                    "Set {}/Data/Model data ".format(n) + code + "a.xlsx",
                    "Set {}/Data/Model data ".format(n) + code + "b.xlsx",
                ],
            ]
        )
    elif n == 2:
        code = "S1144 L168799 K168799"
        dfs = make_dfs(
            [
                [
                    "Set {}/Data/ActionLog data ".format(n) + code + "a.xlsx",
                    "Set {}/Data/ActionLog data ".format(n) + code + "b.xlsx",
                ],
                [
                    "Set {}/Data/Model data ".format(n) + code + "a.xlsx",
                    "Set {}/Data/Model data ".format(n) + code + "b.xlsx",
                ],
            ]
        )
    elif n == 3:
        code = "S4885 L125247 K125247"
        dfs = make_dfs(
            [
                [
                    "Set {}/Data/ActionLog data ".format(n) + code + "a.xlsx",
                    "Set {}/Data/ActionLog data ".format(n) + code + "b.xlsx",
                    "Set {}/Data/ActionLog data ".format(n) + code + "c.xlsx",
                ],
                [
                    "Set {}/Data/Model data ".format(n) + code + "a.xlsx",
                    "Set {}/Data/Model data ".format(n) + code + "b.xlsx",
                    "Set {}/Data/Model data ".format(n) + code + "c.xlsx",
                ],
            ]
        )
    elif n == 4:
        code = "S6587 L178345 K178345"
        dfs = make_dfs(
            [
                [
                    "Set {}/Data/ActionLog data ".format(n) + code + "a.xlsx",
                ],
                [
                    "Set {}/Data/Model data ".format(n) + code + "a.xlsx",
                ],
            ]
        )
    elif n == 5:
        code = "S9881 L121196 K121196"
        dfs = make_dfs(
            [
                [
                    "Set {}/Data/ActionLog data ".format(n) + code + "a.xlsx",
                    "Set {}/Data/ActionLog data ".format(n) + code + "b.xlsx",
                ],
                [
                    "Set {}/Data/Model data ".format(n) + code + "a.xlsx",
                    "Set {}/Data/Model data ".format(n) + code + "b.xlsx",
                ],
            ]
        )
    elif n == 6:
        code = "S9881 L173392 K173392"
        dfs = make_dfs(
            [
                [
                    "Set {}/Data/ActionLog data ".format(n) + code + "a.xlsx",
                    "Set {}/Data/ActionLog data ".format(n) + code + "b.xlsx",
                ],
                [
                    "Set {}/Data/Model data ".format(n) + code + "a.xlsx",
                    "Set {}/Data/Model data ".format(n) + code + "b.xlsx",
                ],
            ]
        )

    return dfs


# make xlsx files for a cerain set n of: dfs, Tasks level 1/2/2.1/3, goal motivation
def output_files(n):
    dfs = run_set(n)

    df_actionlog = dfs[0]
    df_modeldata = dfs[1]
    df_learning = dfs[2]

    to_xlsx(df_actionlog, "Set {}/Output/df_actionlog.xlsx".format(n))
    to_xlsx(df_modeldata, "Set {}/Output/df_modeldata.xlsx".format(n))
    to_xlsx(df_learning, "Set {}/Output/df_learning.xlsx".format(n))
    to_xlsx(tasks_level1(df_actionlog), "Set {}/Output/Tasks_level1.xlsx".format(n))
    to_xlsx(tasks_level2(df_actionlog), "Set {}/Output/Tasks_level2.xlsx".format(n))
    to_xlsx(
        tasks_level2_1(df_actionlog, df_modeldata),
        "Set {}/Output/Tasks_level2_1.xlsx".format(n),
    )
    to_xlsx(
        tasks_level3(df_actionlog, df_modeldata),
        "Set {}/Output/Tasks_level3.xlsx".format(n),
    )
    to_xlsx(
        goal_motivation_1(df_actionlog, df_learning, df_modeldata),
        "Set {}/Output/df_goal_motivation_1.xlsx".format(n),
    )

    return dfs


# return df with overview of measures per set and one will all sets combined
def sets_summary():
    data = []
    df_concat = pd.DataFrame(columns=["%Score", "Relative_Speed_Score", "Set"])
    for i in range(1, 7):
        row = []
        # row = [#Students (after cleaning), #classes, corr: "Goal_motivation_correct cat" and "%Score”,
        # corr: “%Correct” and "%Score”, corr: “#Breaks” and "%Score”
        # #Pass, Average #errors per model, Average #breaks, error types ordered by frequency,
        # #ai, #aii, #aiii, #bi, #bii, #biii]

        dfs = run_set(i)
        df_motivation = goal_motivation_2(dfs[0], dfs[2], dfs[1])

        df_set = df_motivation
        df_set["Set"] = ["Set {}".format(i)] * len(df_set)
        df_concat = df_concat.append(df_set)

        row.append("Set {}".format(i))

        row.append(len(df_motivation))

        if i == 1:
            row.append(2)
        elif i == 2:
            row.append(2)
        elif i == 3:
            row.append(3)
        elif i == 4:
            row.append(1)
        elif i == 5:
            row.append(2)
        elif i == 6:
            row.append(2)

        row.append(len(df_motivation[df_motivation["%Score"] > 50]))

        row.append(dfs[2]["%Correct"].mean())
        row.append(dfs[2]["%Correct"].median())
        row.append(dfs[2]["%Correct"].max())
        row.append(dfs[2]["%Correct"].min())

        row.append(dfs[2]["%Missing"].mean())
        row.append(dfs[2]["%Missing"].median())
        row.append(dfs[2]["%Missing"].max())
        row.append(dfs[2]["%Missing"].min())

        row.append(dfs[2]["%Wrong"].mean())
        row.append(dfs[2]["%Wrong"].median())
        row.append(dfs[2]["%Wrong"].max())
        row.append(dfs[2]["%Wrong"].min())

        row.append(dfs[2]["%Score"].mean())
        row.append(dfs[2]["%Score"].median())
        row.append(dfs[2]["%Score"].max())
        row.append(dfs[2]["%Score"].min())

        row.append(dfs[2]["Partial grades"].mean())
        row.append(dfs[2]["Partial grades"].median())
        row.append(dfs[2]["Partial grades"].max())
        row.append(dfs[2]["Partial grades"].min())

        row.append(dfs[2]["#Geen match"].mean())
        row.append(dfs[2]["#Geen match"].median())
        row.append(dfs[2]["#Geen match"].max())
        row.append(dfs[2]["#Geen match"].min())

        row.append(dfs[2]["Time_s"].mean())
        row.append(dfs[2]["Time_s"].median())
        row.append(dfs[2]["Time_s"].max())
        row.append(dfs[2]["Time_s"].min())

        row.append(dfs[2]["#Errors"].mean())
        row.append(dfs[2]["#Errors"].median())
        row.append(dfs[2]["#Errors"].max())
        row.append(dfs[2]["#Errors"].min())

        row.append(df_motivation["#Breaks"].mean())
        row.append(df_motivation["#Breaks"].median())
        row.append(df_motivation["#Breaks"].max())
        row.append(df_motivation["#Breaks"].min())

        # error_types = [tuple[0] for tuple in errors_overall(dfs[1])]
        error_types = [tuple[0][6:] for tuple in errors_overall(dfs[1])]
        row.append(", ".join(error_types))

        row.append("Figure {}".format(i))

        row.append(len(df_motivation[df_motivation["Goal_motivation_correct"] == "ai"]))
        row.append(
            len(df_motivation[df_motivation["Goal_motivation_correct"] == "aii"])
        )
        row.append(
            len(df_motivation[df_motivation["Goal_motivation_correct"] == "aiii"])
        )
        row.append(len(df_motivation[df_motivation["Goal_motivation_correct"] == "bi"]))
        row.append(
            len(df_motivation[df_motivation["Goal_motivation_correct"] == "bii"])
        )
        row.append(
            len(df_motivation[df_motivation["Goal_motivation_correct"] == "biii"])
        )

        goal_nums = row[-6:]
        goal_names = ["ai", "aii", "aii", "bi", "bii", "biii"]

        l = []
        for i in range(0, 6):
            l.append((goal_nums[i], goal_names[i]))

        l.sort()

        row.append(", ".join([x[1] for x in l]))

        data.append(row)

    df_sets = pd.DataFrame(
        data,
        columns=[
            " ",
            "#Students",
            "#Classes",
            "#Pass",
            "Mean %Correct",
            "Median %Correct",
            "Max %Correct",
            "Min %Correct",
            "Mean %Missing",
            "Median %Missing",
            "Max %Missing",
            "Min %Missing",
            "Mean %Wrong",
            "Median %Wrong",
            "Max %Wrong",
            "Min %Wrong",
            "Mean %Score",
            "Median %Score",
            "Max %Score",
            "Min %Score",
            "Mean Partial grades",
            "Median Partial grades",
            "Max Partial grades",
            "Min Partial grades",
            "Mean #Irrelevant elements",
            "Median #Irrelevant elements",
            "Max #Irrelevant elements",
            "Min #Irrelevant elements",
            "Mean Total time (s)",
            "Median Total time (s)",
            "Max Total time (s)",
            "Min Total time (s)",
            "Mean #Errors",
            "Median #Errors",
            "Max #Errors",
            "Min #Errors",
            "Mean #Breaks",
            "Median #Breaks",
            "Max #Breaks",
            "Min #Breaks",
            "Error types ordered by frequency",
            "Errors plot",
            "#ai",
            "#aii",
            "#aiii",
            "#bi",
            "#bii",
            "#biii",
            "Motivation category ordered by frequency",
        ],
    )

    df_sets = df_sets.T

    cols = df_sets.columns.to_list()
    df_sets[cols] = df_sets[cols].round(2)

    return (df_sets, df_concat)


# # run sets_summary and safe df as xlsx file:
# to_xlsx(sets_summary()[0], "Sets_summary.xlsx")
# to_xlsx(sets_summary()[1], "Sets_concat.xlsx")


#### Plot set summary motivation cats #####
# xlsx_to_csv("Sets_summary.xlsx", "Sets_summary.csv")
# df_sets_summary = pd.read_csv("Sets_summary.csv")
# df_sets_summary.set_index("Unnamed: 0", inplace=True)
# df_sets_summary = df_sets_summary.T
# df_sets_summary = (
#     df_sets_summary["#ai"]
#     .append(df_sets_summary["#aii"])
#     .append(df_sets_summary["#aiii"])
#     .append(df_sets_summary["#bi"])
#     .append(df_sets_summary["#bii"])
#     .append(df_sets_summary["#biii"])
#     .reset_index(drop=True)
# )
# df_sets_summary = pd.DataFrame(df_sets_summary)
# df_sets_summary.columns = ["count"]
# df_sets_summary["category"] = (
#     ["ai"] * 6 + ["aii"] * 6 + ["aiii"] * 6 + ["bi"] * 6 + ["bii"] * 6 + ["biii"] * 6
# )
# df_sets_summary["set"] = ["Set 1", "Set 2", "Set 3", "Set 4", "Set 5", "Set 6"] * 6
# df_sets_summary["count"] = df_sets_summary["count"].astype("int")
# print(df_sets_summary)

# sns.set_theme(style="whitegrid")
# sns.barplot(x = "set", y = "count", hue = "category", palette='Blues', data = df_sets_summary)
# plt.show()
###########################################

#### plot %Score and relative speed per set #####
# xlsx_to_csv("Sets_concat.xlsx", "Sets_concat.csv")
# df_concat = pd.read_csv("Sets_concat.csv")
# sns.catplot(x = "Set", y = "%Score", color="indianred", kind="box", dodge=False, data = df_concat)
# sns.catplot(x = "Set", y = "Relative_Speed_Score", color="rosybrown", kind="box", dodge=False, data = df_concat)
# plt.show()
#########################################

#### plot average %Score for each category #####
# xlsx_to_csv("Sets_concat.xlsx", "Sets_concat.csv")
# df_concat = pd.read_csv("Sets_concat.csv")
# df_concat.dropna(inplace=True)
# # without sets:
# # ax = sns.catplot(x = "Goal_motivation_correct", y = "%Score", color = "cornflowerblue", order = ["ai","aii","aiii","bi","bii","biii"], kind="box", data = df_concat)
# # with sets:
# # ax = sns.catplot(x = "Set", y = "%Score", hue = "Goal_motivation_correct", palette='colorblind', kind="box", data = df_concat, legend=False)
# ax.set(xlabel='Set')
# plt.legend(loc='lower right')
# print(df_concat[["Goal_motivation_correct", "Set"]].value_counts())
# plt.tight_layout()
# plt.show()
#########################################

#### Get #Tasks for each level/set and plot####
# # # make df and save it:
# # df_task = pd.DataFrame(columns=["#Tasks", "Time Duration (s)", "Level", "Set"])
# # # df_time = pd.DataFrame(columns=["Task#"])
# # for set in range(1,7):
# #     xlsx_to_csv("Set {}/Output/Tasks_level1.xlsx".format(set), "Tasks_level1.csv")
# #     df_task_1 = pd.read_csv("Tasks_level1.csv")
# #     tasks = []
# #     times = []
# #     for modelnum, df_model1 in df_task_1.groupby("Model ID"):
# #         tasks.append(len(df_model1))
# #         times.append(df_model1["Time_s"].sum())
# #     levels = ['Level 1'] * len(tasks)
# #     sets = ['Set {}'.format(set)] * len(tasks)
# #     df_tasks_1 = pd.DataFrame({'#Tasks': tasks, "Time Duration (s)": times, 'Level': levels, 'Set': sets})
# #     df_task = df_task.append(df_tasks_1)

# #     xlsx_to_csv("Set {}/Output/Tasks_level2.xlsx".format(set), "Tasks_level2.csv")
# #     df_task_2 = pd.read_csv("Tasks_level2.csv")
# #     tasks = []
# #     times = []
# #     for modelnum, df_model2 in df_task_2.groupby("Model ID"):
# #         tasks.append(len(df_model2))
# #         times.append(df_model2["Time_s"].sum())
# #     levels = ['Level 2'] * len(tasks)
# #     sets = ['Set {}'.format(set)] * len(tasks)
# #     df_task_2 = pd.DataFrame({'#Tasks': tasks, "Time Duration (s)": times, 'Level': levels, 'Set': sets})
# #     df_task = df_task.append(df_task_2)

# #     xlsx_to_csv("Set {}/Output/Tasks_level2_1.xlsx".format(set), "Tasks_level2_1.csv")
# #     df_task_21 = pd.read_csv("Tasks_level2_1.csv")
# #     tasks = []
# #     times = []
# #     for modelnum, df_model in df_task_21.groupby("Model ID"):
# #         tasks.append(len(df_model))
# #         times.append(df_model["Time_s"].sum())
# #     levels = ['Level 2.1'] * len(tasks)
# #     sets = ['Set {}'.format(set)] * len(tasks)
# #     df_task_21 = pd.DataFrame({'#Tasks': tasks, "Time Duration (s)": times, 'Level': levels, 'Set': sets})
# #     df_task = df_task.append(df_task_21)

# #     xlsx_to_csv("Set {}/Output/Tasks_level3.xlsx".format(set), "Tasks_level3.csv")
# #     df_task_3 = pd.read_csv("Tasks_level3.csv")
# #     tasks = []
# #     times = []
# #     for modelnum, df_model in df_task_3.groupby("Model ID"):
# #         tasks.append(len(df_model))
# #         times.append(df_model["Time_s"].sum())
# #     levels = ['Level 3'] * len(tasks)
# #     sets = ['Set {}'.format(set)] * len(tasks)
# #     df_task_3 = pd.DataFrame({'#Tasks': tasks, "Time Duration (s)": times, 'Level': levels, 'Set': sets})
# #     df_task = df_task.append(df_task_3)

# # df_task["#Tasks"] = df_task["#Tasks"].astype("int")
# # df_task["Time Duration (s)"] = df_task["Time Duration (s)"].astype("float")
# # print(df_task)
# # to_xlsx(df_task, "df_tasks.xlsx")

# # # run plot:
# sns.barplot(data=df_task, x="Set", y="#Tasks", hue="Level", palette='colorblind')


# fig, ax1 = plt.subplots()

# color = 'tab:blue'
# ax1.set_xlabel('Task Level')
# ax1.set_ylabel('#Tasks', color = color)
# # ax1.plot(t, data1, color = color)
# sns.lineplot(data=df_task, x="Set", y="Time Duration (s)", hue="Level", marker='o', sort = False, palette='colorblind', ax=ax1)
# ax1.tick_params(axis ='y', labelcolor = color)

# ax2 = ax1.twinx()

# color = 'tab:green'
# ax2.set_ylabel('Time Duration (s)', color = color)
# sns.barplot(data=df_task, x="Set", y="#Tasks", hue="Level", palette='colorblind')
# # ax2.plot(t, data2, color = color)
# ax2.tick_params(axis ='y', labelcolor = color)
# # plt.tight_layout()

# fig.suptitle('matplotlib.axes.Axes.twinx() \
# function Example\n\n', fontweight ="bold")

# plt.show()
##########################################################


# return correlation matrix
def sets_correlations():
    df_concat = pd.DataFrame(
        columns=[
            "%Correct",
            "%Missing",
            "%Wrong",
            "%Score",
            "Partial grades",
            "#Irrelevant elements",
            "Total time (s)",
            "#Steps",
            "#Breaks",
            "Relative Speed",
            "#Tasks Level 1",
            "#Tasks Level 2",
            "#Tasks Level 2.1",
            "#Tasks Level 3",
        ]
    )
    for i in range(1, 7):
        dfs = run_set(i)
        df_set = tasks_into_learning(dfs[0], dfs[2], dfs[1], level=1)

        df_set["Set"] = ["Set {}".format(i)] * len(df_set)
        df_set = df_set.rename(
            columns={
                "#Tasks": "#Tasks Level 1",
                "#Geen match": "#Irrelevant elements",
                "Time_s": "Total time (s)",
                "Relative_Speed_Score": "Relative Speed",
            }
        )
        df_set.drop(
            [
                "Model ID",
                "User ID",
                "#Correct",
                "#Missing",
                "#Errors",
                "#Score",
                "Break_durations",
                "Euclidean_Distance_Score",
                "Euclidean_Distance_Correct",
                "Relative_Speed",
            ],
            axis=1,
            inplace=True,
        )

        df_level2 = tasks_into_learning(dfs[0], dfs[2], dfs[1], level=2)
        df_set.insert(len(df_set.columns), "#Tasks Level 2", df_level2["#Tasks"])

        df_level21 = tasks_into_learning(dfs[0], dfs[2], dfs[1], level=2.1)
        df_set.insert(len(df_set.columns), "#Tasks Level 2.1", df_level21["#Tasks"])

        df_level3 = tasks_into_learning(dfs[0], dfs[2], dfs[1], level=3)
        df_set.insert(len(df_set.columns), "#Tasks Level 3", df_level3["#Tasks"])

        df_concat = df_concat.append(df_set)
        print("::::::::!!!!  appended set {}".format(i))

    return df_concat


# # save correlation matrix as xlsx
# to_xlsx(sets_correlations(), 'sets_correlations().xlsx')

#### plot correlations heatmaps####
# # reference: https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec
# xlsx_to_csv("sets_correlations().xlsx", "sets_correlations().csv")
# df_sets_corr = pd.read_csv("sets_correlations().csv")
# # ALL sets, run below:
# df_sets_corr.drop(
#     [
#         "Model ID",
#         "User ID",
#         "#Correct",
#         "#Missing",
#         "#Errors",
#         "#Score",
#         "Break_durations",
#         "Euclidean_Distance_Score",
#         "Euclidean_Distance_Correct",
#         "Relative_Speed",
#         "Set",
#         'Unnamed: 0',
#     ],
#     axis=1,
#     inplace=True,
# )

# corr = df_sets_corr.corr()
# cols = corr.columns.to_list()
# corr[cols] = corr[cols].round(2)

# ax = sns.heatmap(
#     corr,
#     vmin=-1,
#     vmax=1,
#     annot=True,
#     center=0,
#     cmap=sns.diverging_palette(20, 220, n=200),
#     square=True,
# )
# ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
# plt.tight_layout()
# plt.show()

# # Run 6 figures' plot:
# dct = {}

# for set, df_set in df_sets_corr.groupby("Set"):
#     df_set.drop(
#         [
#             "Model ID",
#             "User ID",
#             "#Correct",
#             "#Missing",
#             "#Errors",
#             "#Score",
#             "Break_durations",
#             "Euclidean_Distance_Score",
#             "Euclidean_Distance_Correct",
#             "Relative_Speed",
#             "Set",
#             'Unnamed: 0',
#         ],
#         axis=1,
#         inplace=True,
#     )
#     dct[set] = df_set

# dct_corr = {}
# for key in dct:
#     corr = dct[key].corr()
#     cols = corr.columns.to_list()
#     corr[cols] = corr[cols].round(2)
#     dct_corr[key] = corr

# # print(dct_corr)

# fig, axs = plt.subplots(nrows=3, ncols=2)
# fig.set_figheight(10)
# fig.set_figwidth(10)

# sns.set(font_scale=0.5)

# sns.heatmap(
#     dct_corr['Set 1'],
#     vmin=-1,
#     vmax=1,
#     annot=True,
#     center=0,
#     cbar=False,
#     xticklabels='',
#     cmap=sns.diverging_palette(20, 220, n=200),
#     square=True,
#     ax=axs[0,0],
# )
# # axs[0,0].set_xticklabels(axs[0,0].get_xticklabels(), rotation=45, horizontalalignment="right")
# axs[0,0].set_title("Set 1", fontdict={'fontsize': 10})

# sns.heatmap(
#     dct_corr['Set 2'],
#     vmin=-1,
#     vmax=1,
#     annot=True,
#     xticklabels='',
#     yticklabels=False,
#     center=0,
#     cmap=sns.diverging_palette(20, 220, n=200),
#     square=True,
#     ax=axs[0,1],
# )
# # axs[0,1].set_xticklabels(axs[0,1].get_xticklabels(), rotation=45, horizontalalignment="right")
# axs[0,1].set_title("Set 2", fontdict={'fontsize': 10})

# sns.heatmap(
#     dct_corr['Set 3'],
#     vmin=-1,
#     vmax=1,
#     annot=True,
#     cbar=False,
#     xticklabels='',
#     center=0,
#     cmap=sns.diverging_palette(20, 220, n=200),
#     square=True,
#     ax=axs[1,0],
# )
# # axs[1,0].set_xticklabels(axs[1,0].get_xticklabels(), rotation=45, horizontalalignment="right")
# axs[1,0].set_title("Set 3", fontdict={'fontsize': 10})

# sns.heatmap(
#     dct_corr['Set 4'],
#     vmin=-1,
#     vmax=1,
#     annot=True,
#     cbar=False,
#     xticklabels='',
#     yticklabels=False,
#     center=0,
#     cmap=sns.diverging_palette(20, 220, n=200),
#     square=True,
#     ax=axs[1,1],
# )
# # axs[1,1].set_xticklabels(axs[1,1].get_xticklabels(), rotation=45, horizontalalignment="right")
# axs[1,1].set_title("Set 4", fontdict={'fontsize': 10})

# sns.heatmap(
#     dct_corr['Set 5'],
#     vmin=-1,
#     vmax=1,
#     annot=True,
#     cbar=False,
#     center=0,
#     cmap=sns.diverging_palette(20, 220, n=200),
#     square=True,
#     ax=axs[2,0],
# )
# axs[2,0].set_xticklabels(axs[2,0].get_xticklabels(), rotation=45, horizontalalignment="right", fontsize=8)
# axs[2,0].set_title("Set 5", fontdict={'fontsize': 10})

# sns.heatmap(
#     dct_corr['Set 6'],
#     vmin=-1,
#     vmax=1,
#     annot=True,
#     cbar=False,
#     yticklabels=False,
#     center=0,
#     cmap=sns.diverging_palette(20, 220, n=200),
#     square=True,
#     ax=axs[2,1],
# )

# axs[2,1].set_xticklabels(axs[2,1].get_xticklabels(), rotation=45, horizontalalignment="right", fontsize=8)
# axs[2,1].set_title("Set 6", fontdict={'fontsize': 10})

# fig.savefig("output.png")
# plt.tight_layout()
# plt.show()
###############################

#### 6 fig errors plot ####
# # dct = {}
# # for set in range(1, 7):
# #     dfs = run_set(set)
# #     df_set = errors_per_stage2(dfs[1])
# #     df_set = df_set.rename(columns={"Stage": "Target type"})
# #     dct["Set {}".format(set)] = df_set

# # # Save
# # np.save("error_stages_plot_dict.npy", dct)

# # Load
# dct = np.load("error_stages_plot_dict.npy", allow_pickle="TRUE").item()

# fig, axs = plt.subplots(nrows=3, ncols=2)

# dct["Set 1"].pivot(index="Error", columns="Target type", values="#Error").plot(
#     kind="bar", rot=0, ax=axs[0, 0], legend=False, xlabel="", ylabel="Count", fontsize=7
# )
# axs[0, 0].set_title("Set 1", fontdict={"fontsize": 10})

# dct["Set 2"].pivot(index="Error", columns="Target type", values="#Error").plot(
#     kind="bar", rot=0, ax=axs[0, 1], legend=False, xlabel="", fontsize=7
# )
# axs[0, 1].set_title("Set 2", fontdict={"fontsize": 10})

# dct["Set 3"].pivot(index="Error", columns="Target type", values="#Error").plot(
#     kind="bar", rot=0, ax=axs[1, 0], legend=False, xlabel="", ylabel="Count", fontsize=7
# )
# axs[1, 0].set_title("Set 3", fontdict={"fontsize": 10})

# dct["Set 4"].pivot(index="Error", columns="Target type", values="#Error").plot(
#     kind="bar", rot=0, ax=axs[1, 1], legend=False, xlabel="", fontsize=7
# )
# axs[1, 1].set_title("Set 4", fontdict={"fontsize": 10})

# dct["Set 5"].pivot(index="Error", columns="Target type", values="#Error").plot(
#     kind="bar", rot=0, ax=axs[2, 0], legend=False, ylabel="Count", fontsize=7
# )
# axs[2, 0].set_title("Set 5", fontdict={"fontsize": 10})

# dct["Set 6"].pivot(index="Error", columns="Target type", values="#Error").plot(
#     kind="bar", rot=0, ax=axs[2, 1], legend=False, fontsize=7
# )
# axs[2, 1].set_title("Set 6", fontdict={"fontsize": 10})

# handles, labels = axs[2, 1].get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper left')

# plt.tight_layout()
# plt.show()
###################################################


######################Linear Regression Speed & Missing -> Score###############################
def reg_1(csvs=False, plot=False):
    if csvs:
        ### make csv files ####
        df_all = []
        for i in range(1, 5):
            dfs = run_set(i)
            df_all.append(goal_motivation_2(dfs[0], dfs[2], dfs[1]))

        df_train = pd.concat(df_all)
        df_train.to_csv("df_train_motivation.csv")

        df_all = []
        for i in range(5, 7):
            dfs = run_set(i)
            df_all.append(goal_motivation_2(dfs[0], dfs[2], dfs[1]))

        df_test = pd.concat(df_all)
        df_test.to_csv("df_test_motivation.csv")

    #### make dfs from csv files and run model ####

    df_train = pd.read_csv("df_train_motivation.csv")
    df_test = pd.read_csv("df_test_motivation.csv")

    x_train = df_train.dropna()[["%Missing", "Relative_Speed_Score"]].values
    y_train = df_train.dropna()["%Score"].values

    x_test = df_test.dropna()[["%Missing", "Relative_Speed_Score"]].values
    y_test = df_test.dropna()["%Score"].values

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import classification_report, confusion_matrix

    model = LinearRegression().fit(x_train, y_train)
    r_sq = model.score(x_train, y_train)
    print("coefficient of determination:", r_sq)
    print("intercept:", model.intercept_)
    print("slope:", model.coef_)

    y_pred = model.predict(x_test)

    if plot == False:
        pass
    else:
        if plot == 1:
            ### plot version 1####

            # Reference: https://aegis4048.github.io/mutiple_linear_regression_and_visualization_in_python
            from mpl_toolkits.mplot3d import Axes3D

            x = x_train[:, 0]
            y = x_train[:, 1]
            z = y_train

            x_pred = np.linspace(x.min(), x.max(), 30)  # range of x values
            y_pred = np.linspace(y.min(), y.max(), 30)  # range of y values
            xx_pred, yy_pred = np.meshgrid(x_pred, y_pred)
            model_viz = np.array([xx_pred.flatten(), yy_pred.flatten()]).T

            predicted = model.predict(model_viz)

            plt.style.use("default")

            fig = plt.figure(figsize=(12, 4))

            ax1 = fig.add_subplot(131, projection="3d")
            ax2 = fig.add_subplot(132, projection="3d")
            ax3 = fig.add_subplot(133, projection="3d")

            axes = [ax1, ax2, ax3]

            for ax in axes:
                ax.plot(
                    x,
                    y,
                    z,
                    color="k",
                    zorder=15,
                    linestyle="none",
                    marker="o",
                    alpha=0.5,
                )
                ax.scatter(
                    xx_pred.flatten(),
                    yy_pred.flatten(),
                    predicted,
                    facecolor=(0, 0, 0, 0),
                    s=20,
                    edgecolor="#70b3f0",
                )
                ax.set_xlabel("Missing (%)", fontsize=10)
                ax.set_ylabel("Relative speed (% per sec)", fontsize=10)
                ax.set_zlabel("Score (%)", fontsize=10)
                ax.locator_params(nbins=4, axis="x")
                ax.locator_params(nbins=5, axis="x")

            ax1.view_init(elev=28, azim=120)
            ax2.view_init(elev=4, azim=114)
            ax3.view_init(elev=60, azim=165)

            fig.suptitle("$R^2 = %.2f$" % r_sq, fontsize=20)

            fig.tight_layout()
            plt.show()
        elif plot == 2:
            #### plot version 2 ####

            # Reference: https://gist.github.com/aricooperdavis/c658fc1c5d9bdc5b50ec94602328073b
            from mpl_toolkits.mplot3d import Axes3D

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(x_train[:, 0], x_train[:, 1], y_train, marker=".", color="red")
            ax.set_xlabel("%Missing")
            ax.set_ylabel("Relative speed")
            ax.set_zlabel("%Score")

            coefs = model.coef_
            intercept = model.intercept_
            xs = np.tile(np.arange(61), (61, 1))
            ys = np.tile(np.arange(61), (61, 1)).T
            zs = xs * coefs[0] + ys * coefs[1] + intercept

            ax.plot_surface(xs, ys, zs, alpha=0.5)
            plt.show()


# reg_1(plot=1)

###############################################################################################


######################Linear Regression Breaks & Tasks3 -> Score (and try other combinations)###################
def reg_2(csvs=False, plot=False):
    if csvs:
        ### make csv files ####
        df_all = []
        for i in range(1, 5):
            dfs = run_set(i)
            df_all.append(tasks_into_learning(dfs[0], dfs[2], dfs[1], 3))

        df_train = pd.concat(df_all)
        df_train.to_csv("df_train_tasks.csv")

        df_all = []
        for i in range(5, 7):
            dfs = run_set(i)
            df_all.append(tasks_into_learning(dfs[0], dfs[2], dfs[1], 3))

        df_test = pd.concat(df_all)
        df_test.to_csv("df_test_tasks.csv")

    #### make dfs from csv files and run model ####

    df_train = pd.read_csv("df_train_tasks.csv")
    df_test = pd.read_csv("df_test_tasks.csv")

    df_train_2 = pd.read_csv("df_train_motivation.csv")[
        ["Model ID", "Goal_motivation_correct", "Goal_motivation_correct cat"]
    ]
    df_test_2 = pd.read_csv("df_test_motivation.csv")[
        ["Model ID", "Goal_motivation_correct", "Goal_motivation_correct cat"]
    ]

    df_train = pd.merge(df_train, df_train_2, on="Model ID")
    df_test = pd.merge(df_test, df_test_2, on="Model ID")

    x_train = df_train.dropna()[["#Breaks", "#Tasks"]].values
    y_train = df_train.dropna()["%Score"].values

    x_test = df_test.dropna()[["#Breaks", "#Tasks"]].values
    y_test = df_test.dropna()["%Score"].values

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import classification_report, confusion_matrix

    model = LinearRegression().fit(x_train, y_train)
    r_sq = model.score(x_train, y_train)
    print("coefficient of determination:", r_sq)
    print("intercept:", model.intercept_)
    print("slope:", model.coef_)

    y_pred = model.predict(x_test)

    if plot == False:
        pass
    else:
        if plot == 1:
            ### plot version 1####

            # Reference: https://aegis4048.github.io/mutiple_linear_regression_and_visualization_in_python
            from mpl_toolkits.mplot3d import Axes3D

            x = x_train[:, 0]
            y = x_train[:, 1]
            z = y_train

            x_pred = np.linspace(x.min(), x.max(), 30)  # range of x values
            y_pred = np.linspace(y.min(), y.max(), 30)  # range of y values
            xx_pred, yy_pred = np.meshgrid(x_pred, y_pred)
            model_viz = np.array([xx_pred.flatten(), yy_pred.flatten()]).T

            predicted = model.predict(model_viz)

            plt.style.use("default")

            fig = plt.figure(figsize=(12, 4))

            ax1 = fig.add_subplot(131, projection="3d")
            ax2 = fig.add_subplot(132, projection="3d")
            ax3 = fig.add_subplot(133, projection="3d")

            axes = [ax1, ax2, ax3]

            for ax in axes:
                ax.plot(
                    x,
                    y,
                    z,
                    color="k",
                    zorder=15,
                    linestyle="none",
                    marker="o",
                    alpha=0.5,
                )
                ax.scatter(
                    xx_pred.flatten(),
                    yy_pred.flatten(),
                    predicted,
                    facecolor=(0, 0, 0, 0),
                    s=20,
                    edgecolor="#70b3f0",
                )
                ax.set_xlabel("#Breaks", fontsize=10)
                ax.set_ylabel("#Tasks", fontsize=10)
                ax.set_zlabel("Score (%)", fontsize=10)
                ax.locator_params(nbins=4, axis="x")
                ax.locator_params(nbins=5, axis="x")

            ax1.view_init(elev=28, azim=120)
            ax2.view_init(elev=4, azim=114)
            ax3.view_init(elev=60, azim=165)

            fig.suptitle("$R^2 = %.2f$" % r_sq, fontsize=20)

            fig.tight_layout()
            plt.show()
        elif plot == 2:
            #### plot version 2 ####

            # Reference: https://gist.github.com/aricooperdavis/c658fc1c5d9bdc5b50ec94602328073b
            from mpl_toolkits.mplot3d import Axes3D

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(x_train[:, 0], x_train[:, 1], y_train, marker=".", color="red")
            ax.set_xlabel("%Missing")
            ax.set_ylabel("Relative speed")
            ax.set_zlabel("%Score")

            coefs = model.coef_
            intercept = model.intercept_
            xs = np.tile(np.arange(61), (61, 1))
            ys = np.tile(np.arange(61), (61, 1)).T
            zs = xs * coefs[0] + ys * coefs[1] + intercept

            ax.plot_surface(xs, ys, zs, alpha=0.5)
            plt.show()


# reg_2(plot=1)

###############################################################################################


def kmeans_cluster(n_clusters=0, elbow=False, silho=False):
    import seaborn as sns
    from sklearn.cluster import KMeans
    from matplotlib import pyplot
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import normalize
    from sklearn.decomposition import PCA

    df_train = pd.read_csv("df_train_tasks.csv")
    df_test = pd.read_csv("df_test_tasks.csv")

    df_train_2 = pd.read_csv("df_train_motivation.csv")[
        ["Model ID", "Goal_motivation_correct", "Goal_motivation_correct cat"]
    ]
    df_test_2 = pd.read_csv("df_test_motivation.csv")[
        ["Model ID", "Goal_motivation_correct", "Goal_motivation_correct cat"]
    ]

    df_train = pd.merge(df_train, df_train_2, on="Model ID")
    df_test = pd.merge(df_test, df_test_2, on="Model ID")

    df = pd.concat([df_train, df_test])

    # make dataset suitable for clustering. referece: https://stackoverflow.com/a/28020783
    df = df.drop("Model ID", axis=1)
    df = df.drop("Break_durations", axis=1)
    df = df.drop("Goal_motivation_correct", axis=1)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(method="ffill", inplace=True)

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    df_normalized = normalize(df_scaled)  # returns array
    df_normalized = pd.DataFrame(df_normalized)

    # pca = PCA(n_components = 2)
    # df_principal = pca.fit_transform(df)
    # df_principal = pd.DataFrame(df_principal)
    # # "%Missing", "Relative_Speed_Score", "%Wrong", "#Tasks", "#Breaks"
    df_principal = df[["#Tasks", "#Breaks", "Relative_Speed_Score"]]
    df_principal.columns = ["P1", "P2", "P3"]

    def k_cluster(df_principal, n_clusters):
        clustering_kmeans = KMeans(
            n_clusters=n_clusters, precompute_distances="auto", n_jobs=-1
        )
        df_principal["clusters"] = clustering_kmeans.fit_predict(df_principal)
        print(df_principal)

        kmeans_silhouette = silhouette_score(
            df_principal, clustering_kmeans.labels_
        ).round(2)
        print("kmeans silhouette coefficient: ", kmeans_silhouette)

        # reduced_data = PCA(n_components=2).fit_transform(df_principal)
        # results = pd.DataFrame(reduced_data,columns=['pca1','pca2'])

        # sns.scatterplot(x="pca1", y="pca2", hue=df_principal['clusters'], palette="colorblind", data=results[['pca1','pca2']])
        sns.scatterplot(
            x="P1",
            y="P2",
            hue=df_principal["clusters"],
            palette="colorblind",
            data=df_principal[["P1", "P2"]],
        )
        plt.title("K-means Clustering with 2 dimensions")
        plt.show()

        # # create scatter plot for samples from each cluster
        # ones = []
        # for i in range(len(data)):
        #     ones.append(1)
        # for cluster in clusters:
        #     # get row indexes for samples with this cluster
        #     row_ix = where(yhat == cluster)
        #     # create scatter of these samples
        #     # pyplot.scatter(data, ones)  # because data is 1D
        #     pyplot.scatter(data[:, 0], data[:, 1])
        # # show the plot
        # print("::Cluster centers::")
        # print(model.cluster_centers_)
        # pyplot.show()
        # return (model, kmeans_silhouette)
        return df_principal

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
        # Notice you start at 2 clusters for silhouette coefficient
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
        sse(df_principal)
    elif silho:
        silhouette(df_principal)
    # elif h:
    #     hierar(df_principal)
    elif n_clusters == 0:
        print("Please enter the number of clusters")
    else:
        return k_cluster(df_principal, n_clusters)


# kmeans_cluster(elbow=True)
# kmeans_cluster(silho=True)
# kmeans_cluster(n_clusters=2)
# to_xlsx(kmeans_cluster(n_clusters=6), "kmeans_.xlsx")


def cluster_match():
    # xlsx_to_csv('kmeans_all features.xlsx', 'kmeans_all features.csv')
    df = pd.read_csv("kmeans_all features.csv")

    all_columns = list(df)  # Creates list of all column headers
    df[all_columns] = df[all_columns].astype(str)

    print("\ntotal no. of models: ", len(df))
    df["Goal-1"] = (
        df["Goal_motivation_correct cat"] + "-" + df["all features-3-3-3 (0.62)"]
    )
    df["Goal-2"] = (
        df["Goal_motivation_correct cat"] + "-" + df["%missing-speed-4-8-4 (0.55)"]
    )
    df["Goal-3"] = (
        df["Goal_motivation_correct cat"]
        + "-"
        + df["missing--wrong-speed-3-4-4 (0.47)"]
    )
    df["Goal-4"] = (
        df["Goal_motivation_correct cat"]
        + "-"
        + df["missing--wrong--tasks-speed-3-5-5 (0.43)"]
    )
    df["Goal-5"] = (
        df["Goal_motivation_correct cat"]
        + "-"
        + df["missing--wrong--tasks--breaks-speed-3-5-5 (0.43)"]
    )
    df["Goal-6"] = (
        df["Goal_motivation_correct cat"] + "-" + df["tasks-breaks-3-6-6 (0.51)"]
    )

    for i in range(1, 7):
        print(" \nGoal-{}: ".format(i))
        dct = df["Goal-{}".format(i)].value_counts().to_dict()
        sorted_dict = {}
        sorted_keys = sorted(dct, key=dct.get, reverse=True)

        for w in sorted_keys:
            sorted_dict[w] = dct[w]

        for k, v in list(sorted_dict.items()):
            if k[0] == "-":
                del sorted_dict[k]

        print(sorted_dict)
        print(len(sorted_dict))


# cluster_match()


### To cluster time ###

# Run elbow then silhouette to decide on number of clusters
# Then run cluster with the number of clusters decided on:

# cluster_time(df_actionlog, elbow=True)    # look for elbow point
# cluster_time(df_actionlog, silho=True)      # look for max point
# cluster_time(df_actionlog, n_clusters=3)

#######################

### Errors visualization ###

# errors_overall(df_modeldata)
# errors_overall(df_modeldata, True)
# errors_per_stage(df_modeldata)
# errors_per_stage(df_modeldata, True)

############################