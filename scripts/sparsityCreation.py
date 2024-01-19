import os
import re
import csv
import pandas as pd
import numpy as np
from csv import writer
import random
from nnsight import LanguageModel
import time
from numpy.core.multiarray import empty
import random
import string

random.seed(42)

def generate_sparse_table(input_dir, table_filepath, sparsity, output_dir=None, columns_subset=None, sep=";"):   
    input_csv = input_dir + table_filepath
    with open(input_csv, 'r') as input_csv_file:
        input_table = []
        csv_reader = csv.reader(input_csv_file, delimiter=sep)
        line_count = 0
        for row in csv_reader:
            input_table.append(row)
        if columns_subset == None:
            num_columns = len(input_table[0]) - 1
            columns_subset = [i for i in range(num_columns)]
        num_rows = len(input_table) - 1
        num_null_cells = int(sparsity * num_rows * len(columns_subset))
        # Generate a list of all cell positions in the table
        all_positions = []
        for i in range(num_rows):
            for j in range(len(columns_subset)):
                curr_row = input_table[i]
                if 0 <= j < len(curr_row):
                    all_positions.append((i,j))
        # Randomly select positions to set as null (None) values
        null_positions = random.sample(all_positions, num_null_cells)
        for i, row in enumerate(input_table):
            for j, value in enumerate(row):
                if (i, j) in null_positions:
                    input_table[i][j] = None
        if output_dir is not None:
            output_file = output_dir + table_filepath
            df = pd.DataFrame(input_table)
            df.to_csv(output_file, sep)


def main():
    bench_data_dir = "../data/ben_y"
    sparse_val = 5
    bench_sparse_dir = f"../data/ben_y_sparse_{sparse_val}"
    groundtruth_file = bench_data_dir + "/groundtruth.csv"
    query_dir = bench_data_dir + "/query/"
    dl_dir = bench_data_dir + "/datalake/"
    output_query_dir_anon = bench_sparse_dir + "/query/"
    output_dl_dir_anon = bench_sparse_dir + "/datalake/"
    gt = pd.read_csv(groundtruth_file)
    query_table = []
    os.makedirs(bench_sparse_dir, exist_ok=True)
    os.makedirs(output_query_dir_anon, exist_ok=True)
    os.makedirs(output_dl_dir_anon, exist_ok=True)
    data_lake_table = []
    for index, row in gt.iterrows():
        table_1 = row['query_table']
        table_2 = row['data_lake_table']
        if table_1 not in query_table:
            generate_sparse_table(query_dir, table_1, sparse_val/100.0, output_dir=output_query_dir_anon, columns_subset=None, sep=";")
            query_table.append(table_1)
        if table_2 not in data_lake_table:
            generate_sparse_table(dl_dir, table_2, sparse_val/100.0, output_dir=output_dl_dir_anon, columns_subset=None, sep=";")
            data_lake_table.append(table_2)
        

if __name__ == "__main__":
    main()
    