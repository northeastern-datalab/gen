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
root_dir = ""
data_dir = "../data"
model = LanguageModel('mistralai/Mixtral-8x7B-Instruct-v0.1', device_map='auto')

def get_subject_cols(query_tables, data_lake_tables, table_1, table_2, sep=';'):
    query_file_path = query_tables + table_1
    data_lake_file_path = data_lake_tables + table_2
    query_table_header_list = []
    dl_table_header_list = []
    with open(query_file_path, mode ='r') as file:
        csvFile = csv.reader(file, delimiter=sep)
        line_counter = 0
        for row in csvFile:
            line_counter += 1
            if line_counter == 1:
                query_table_header_list = row
            if line_counter > 1:
                break
    with open(data_lake_file_path, mode ='r') as file:
        csvFile = csv.reader(file, delimiter=sep)
        line_counter = 0
        for row in csvFile:
            line_counter += 1
            if line_counter == 1:
                dl_table_header_list = row
            if line_counter > 1:
                break
    table_1_headers = ";".join(query_table_header_list)
    table_2_headers = ";".join(dl_table_header_list)
    prompt = f"[INST] Given the following column headers for a pair of tables, give the subject column name from table 1 that aligns with the matching column in table 2. Here's table 1 column header: {table_1_headers}. Here's table 2 column header: {table_2_headers}. Respond to this instruction with just the requested subject column header. [/INST]"
    start_time = time.time()
    with model.generate(max_new_tokens=50, temperature=0.3, remote=False) as generator:
        with generator.invoke(prompt):
            pass
    end_time = time.time()
    model_response = model.tokenizer.batch_decode(generator.output)[0]
    prompt_output = model_response.split("[/INST]")[-1]
    final_answer = ""
    for col in query_table_header_list:
        curr_col = col.replace('"', '')
        if curr_col in prompt_output:
            final_answer = col
            break
    # if it couldn't find anything, grab the first column
    if final_answer == "":
        final_answer = query_table_header_list[0]
    print("Q FILENAME:::", table_1, flush=True)
    print("PROMPT_OUTPUT::", prompt_output, flush=True)
    print("final_answer:::", final_answer, flush=True)
    return final_answer
    

def main():
    bench_data_dir = "../data/ben_y/"
    groundtruth_file = bench_data_dir + "SANTOS_gt.csv"
    query_tables = bench_data_dir + "/query/"
    data_lake_tables = bench_data_dir + "/datalake/"
    gt = pd.read_csv(groundtruth_file)
    query_table = []
    data_lake_table = []
    intent_col_names = []
    for index, row in gt.iterrows():
        table_1 = row['query_table']
        table_2 = row['data_lake_table']
        unionable = str(row['unionable'])
        if unionable == "0":
            intent_col_names.append('N/A')
        else:
            intent_curr_col = get_subject_cols(query_tables,data_lake_tables,table_1,table_2)
            intent_col_names.append(intent_curr_col)
    gt['intent_col_name'] = intent_col_names
    gt.to_csv(groundtruth_file)
        


if __name__ == "__main__":
    main()
    