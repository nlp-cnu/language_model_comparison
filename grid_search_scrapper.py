import pandas as pd
import os


def best_row_in_file(filename, master):
    """
    finds the best row in the file - by micro f1 - and appends to master dataframe
    :param filename: file where hyper parameter search results came
    :param master: master dataframe to append best results
    :return: void
    """
    fn_split = os.path.splitext(filename)[0].split(os.sep)
    hyper_parameters = fn_split[-1].split('_')
    learning_rate, dropout, arch = hyper_parameters[0], hyper_parameters[1], fn_split[1]

    df = pd.read_csv(filename)
    df['info'] = "{}_{}_{}".format(arch, learning_rate, dropout)
    df['total_time'] = sum_time_column(df['time'])
    best_index = df['micro_F1'].idxmax()
    row_list = df.values.tolist()[best_index]
    master.loc[len(master)] = row_list


def sum_time_column(column):
    """
    sums column of time
    :param column: pandas column of seconds
    :return: total amount of seconds taken to train as string
    """
    sep_values = column.tolist()
    total = 0
    for i in sep_values:
        total += int(i[:-1])
    return "{}s".format(total)


def create_template_pandas():
    """
    :return: template dataframe
    """
    return pd.DataFrame(columns=['epoch', 'time', 'loss', 'num_neg', 'macro_precision', 'macro_recall',
                                 'macro_F1', 'micro_precision', 'micro_recall', 'micro_F1', 'info', 'total_time'])


def iterate_directories(directory, bp_choice):
    """
    iterates through grid_search directory and creates a dataframe of the best performance of each architecture either
    with or without back propagation on
    :param directory: directory where results are in (grid_search)
    :param bp_choice: either BP or noBP depending if back propagation was used
    :return: nothing, writes multiple csv files of results
    """
    directory_b = os.fsencode(directory)
    master_dataframe = create_template_pandas()
    # iterate through each directory of architecture "grid_search/"
    for directory2 in os.listdir(directory_b):
        d_name = os.fsdecode(directory2)
        if d_name == ".DS_Store":
            continue
        architecture_dataframe = create_template_pandas()
        # iterate through each architecture's results "grid_search/[arch]/BP/"
        for file in os.listdir(os.fsencode(os.path.join(directory_b, directory2, os.fsencode(bp_choice)))):
            filename = os.path.join(directory, d_name, bp_choice, os.fsdecode(file))
            best_row_in_file(filename, architecture_dataframe)

        architecture_dataframe = architecture_dataframe.sort_values(by=['micro_F1'], ascending=False)
        architecture_dataframe.to_csv('output/architecture_specific/best_{}_{}.csv'.format(d_name, bp_choice),
                                      index=False)

        row_list = architecture_dataframe.values.tolist()[0]
        master_dataframe.loc[len(master_dataframe)] = row_list

    master_dataframe = master_dataframe.sort_values(by=['micro_F1'], ascending=False)
    master_dataframe.to_csv('output/{}_overall.csv'.format(bp_choice), index=False)


def main():
    master_directory = "grid_search"
    iterate_directories(master_directory, 'noBP')
    iterate_directories(master_directory, 'BP')


if __name__ == "__main__":
    main()
    # open grid search folder
    # for each architecture folder... enter BP folder... for each hyper parameter combination - append to other
    # dataframe... then find absolute best in architecture dataframe and append to master dataframe
