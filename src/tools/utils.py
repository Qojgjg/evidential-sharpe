from pathlib import Path

import numpy as np
import pandas as pd


def write_result(result_folder, output_filename, performance_list, inference_list):
    """

    :param result_folder: Place to store all runs result.
    :param output_filename: Name related to the run configuration.
    :param performance_list:
    :param inference_list:
    :return:
    """
    store_folder = f'{result_folder}/{output_filename}'
    if Path(store_folder).exists():
        store_folder = f'{store_folder}_temp'

    Path(store_folder).mkdir(parents=True, exist_ok=False)

    row_list = []
    metrics = []

    # Loop over folds.
    for i in range(len(performance_list)):
        performance = performance_list[i]
        fold_val_result = []

        for metric_name in performance.keys():
            name_parts = metric_name.split('_')
            if name_parts[0] == 'val':
                report_metric = '_'.join(name_parts[1:])
                metrics.append(report_metric)

                fold_val_result.append(performance['val_' + report_metric][-1])

        row_list.append(fold_val_result)

        inference = inference_list[i]
        for inf_name in inference.keys():
            inf_result = inference[inf_name]
            np.save(store_folder + f'/{inf_name}_fold_{i}.npy', inf_result)

    df = pd.DataFrame(row_list, columns=list(dict.fromkeys(metrics)))
    average_row = df.mean(axis=0)
    df = df.append(average_row.T, ignore_index=True)

    df.to_csv(store_folder + f'/{output_filename}.csv', sep=',', index=False)


def average_run_results(result_folder, nb_runs):
    df_list = []

    for i in range(nb_runs):
        result_folder_name = f"{result_folder}/{i}"
        file_list = list(Path(result_folder_name).iterdir())
        for file in file_list:
            if file.suffix == '.csv':
                df = pd.read_csv(file)
                df_list.append(df)

    df_average = pd.concat(df_list).groupby(level=0).mean()

    df_average.to_csv(f'{result_folder}/average.csv', index=False)


def median_run_results(result_folder, nb_runs):
    mean_sharpe_list = []
    for i in range(nb_runs):
        result_folder_name = f"{result_folder}/{i}"
        file_list = list(Path(result_folder_name).iterdir())
        for file in file_list:
            if file.suffix == '.csv':
                df = pd.read_csv(file)
                mean_sharpe_list.append(df.iloc[-1]["sharpe_ratio"])

    min_index = np.argsort(mean_sharpe_list)[len(mean_sharpe_list) // 2]
    df_median = pd.read_csv(f'{result_folder}/{min_index}/{min_index}.csv')
    df_median.to_csv(f'{result_folder}/median_is_{min_index}.csv', index=False)

