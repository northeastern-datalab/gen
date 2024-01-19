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

def add_first_10_rows(input_dir, table_file_name, sep=";", separation="semi-colon", save_output_dir=None):
    file_path = input_dir + table_file_name
    table_header_list = []
    with open(file_path, mode ='r') as file:
        csvFile = csv.reader(file, delimiter=sep)
        line_counter = 0
        for row in csvFile:
            line_counter += 1
            if line_counter == 1:
                table_header_list = row
            if line_counter > 1:
                break
    table_topic = " ".join(table_file_name.split('_')[:-1])
    textuality = random.randint(1, 5) * len(table_header_list)
    table_headers = ";".join(table_header_list)
    prompt = f"[INST] Given the following column header for a table about {table_topic}, generate 10 table rows where each row has at least {textuality} words. Here's the column header: {table_headers}. Answer this task in the format of {separation}-separated rows, where each row is in a new line. [/INST]"
    start_time = time.time()
    with model.generate(max_new_tokens=30000, do_sample=True, temperature=1, remote=False) as generator:
        with generator.invoke(prompt):
            pass
    end_time = time.time()
    model_response = model.tokenizer.batch_decode(generator.output)[0]
    prompt_output = model_response.split("[/INST]")[-1]
    print("PROMPT_OUTPUT::", prompt_output, flush=True)
    potential_rows = prompt_output.split('\n')
    all_rows = [table_header_list]
    for row in potential_rows:
        if sep in row:
            curr_row = row.split(sep)
            if len(curr_row) > len(table_header_list):
                squished_row = curr_row[:len(table_header_list)]
                squished_row.append(" ".join(curr_row[len(table_header_list):]))
                all_rows.append(squished_row)
            elif len(curr_row) < len(table_header_list):
                squished_row = curr_row[:len(table_header_list)]
                while len(squished_row) < len(table_header_list):
                    squished_row.append(None)
                all_rows.append(squished_row)
            else:
                all_rows.append(curr_row)
    time_taken = end_time - start_time
    print(f'Time taken to generate 10 rows: {time_taken} seconds')
    if save_output_dir is not None:
        file_path = save_output_dir + table_file_name
        with open(file_path, mode='w') as writing_file:
            writer = csv.writer(writing_file, delimiter=';', quotechar='"')
            for curr_row in all_rows:
                writer.writerow(curr_row)
    return all_rows


def add_more_rows(input_dir, table_file_name, sep=";", separation="semi-colon", save_output_dir=None):
    file_path = input_dir + table_file_name
    table_header_list = []
    first_set_of_rows = []
    with open(file_path, mode ='r') as file:
        csvFile = csv.reader(file, delimiter=sep)
        line_counter = 0
        for row in csvFile:
            line_counter += 1
            if line_counter == 1:
                table_header_list = row
            if line_counter > 1:
                first_set_of_rows.append(";".join(row))
    row_texts = "\n".join(first_set_of_rows[-3:])
    table_topic = " ".join(table_file_name.split('_')[:-1])
    textuality = random.randint(1, 5) * len(table_header_list)
    table_headers = ";".join(table_header_list)
    prompt = f"[INST] Given the column headers and last couple rows for a table about {table_topic}, generate 10 more table rows where each row has at least {textuality} words. Here's the column header: {table_headers}. Here's the last couple rows:{row_texts}. Answer this task in the format of {separation}-separated rows, where each row is in a new line. [/INST]"
    start_time = time.time()
    with model.generate(max_new_tokens=30000, do_sample=True, temperature=1, remote=False) as generator:
        with generator.invoke(prompt):
            pass
    end_time = time.time()
    model_response = model.tokenizer.batch_decode(generator.output)[0]
    prompt_output = model_response.split("[/INST]")[-1]
    print("FILENAME::", table_file_name, flush=True)
    print("PROMPT_OUTPUT::", prompt_output, flush=True)
    potential_rows = prompt_output.split('\n')
    all_rows = []
    for row in potential_rows:
        if row in first_set_of_rows:
            continue
        if sep in row:
            curr_row = row.split(sep)
            if len(curr_row) > len(table_header_list):
                squished_row = curr_row[:len(table_header_list)]
                squished_row.append(" ".join(curr_row[len(table_header_list):]))
                all_rows.append(squished_row)
            elif len(curr_row) < len(table_header_list):
                squished_row = curr_row[:len(table_header_list)]
                while len(squished_row) < len(table_header_list):
                    squished_row.append(None)
                all_rows.append(squished_row)
            else:
                all_rows.append(curr_row)
    time_taken = end_time - start_time
    print(f'Time taken to generate 10 rows: {time_taken} seconds')
    if save_output_dir is not None:
        print("came here")
        file_path = save_output_dir + table_file_name
        with open(file_path, mode='a+') as writing_file:
            writer = csv.writer(writing_file, delimiter=';', quotechar='"')
            for curr_row in all_rows:
                writer.writerow(curr_row)
    return all_rows
            

def process_malformedtables_using_logs(input_dir, table_file_name, log_file, sep=";"):
    file_path = input_dir + table_file_name
    table_header_list = []
    with open(file_path, mode ='r') as file:
        csvFile = csv.reader(file, delimiter=sep)
        line_counter = 0
        for row in csvFile:
            line_counter += 1
            if line_counter == 1:
                table_header_list = row
            if line_counter > 1:
                break
    if len(table_header_list) <= 2:
        with open(log_file, mode ='r') as file:
            log_file = file.read()
        split_by_file_names = log_file.split("FOR FILE:::-")
        content_to_care = ""
        filtered_list = []
        for content in split_by_file_names:
            if table_file_name in content:
                content_to_care = content
                print(f"CONTENT FOUND FOR FILE {table_file_name}")
                print(content_to_care)
                pattern = r'Table 2:((?:.*?;)+)'
                matches = re.search(pattern, content_to_care, re.DOTALL)
                if matches:
                    semi_colon_separated_content = matches.group(1).strip().split(';')
                    if len(semi_colon_separated_content) > 3:
                        for item in semi_colon_separated_content:
                            curr_item = item.strip()
                            if curr_item:
                                if "\n\n" not in item:
                                    filtered_list.append(curr_item)
                                else:
                                    break
                        if len(filtered_list) == 0:
                            alt_content = semi_colon_separated_content[0]
                            new_match = alt_content.split("\n\n")
                            new_cols = []
                            if len(new_match) >= 2:
                                new_cols = new_match[0].split("\n")[:-1]
                            filtered_list = new_cols
                        # TRYING ALTERNATE SOLUTIONS
                        if len(filtered_list) < 2:
                            match_output = re.search(r"(?<=Table 2: )(.*)\n\n", content, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
                            if match_output:
                                print("MATCH OUTPUT", match_output)
                                filtered_list = match_output.group(0).split(';')
                        if len(filtered_list) < 2:
                            match_output = re.search(r"(?<=Table 2: )(.*)\n", content, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
                            if match_output:
                                filtered_list = match_output.group(0).split('\n\n')[0].split(',')         
        if content_to_care == "":
            print(f"CONTENT NOT FOUND FOR FILE {table_file_name}")
        if len(filtered_list) > 2:
            with open(file_path, mode='w') as writing_file:
                writer = csv.writer(writing_file, delimiter=';', quotechar='"')
                writer.writerow(filtered_list)

def fill_missingcontent_first10rows(input_dir, table_file_name, log_file, counter, save_output_dir_anon, sep=";"):
    file_path = input_dir + table_file_name
    line_counter = 0
    with open(file_path, mode ='r') as file:
        csvFile = csv.reader(file, delimiter=sep)  
        for row in csvFile:
            line_counter += 1
    if line_counter <= 10:
        print("FILENAME::", table_file_name, flush=True)
        filtered_list = []
        trying_out = 0
        while trying_out < 3:
            all_rows = add_first_10_rows(input_dir, table_file_name, sep=sep, separation="semi-colon")
            if len(all_rows) > 10:
                new_file_path = save_output_dir_anon + table_file_name
                with open(file_path, mode='w') as writing_file:
                    writer = csv.writer(writing_file, delimiter=sep, quotechar='"')
                    for curr_row in all_rows:
                        writer.writerow(curr_row)
                break
            else:
                print("CURR+TRY+OUT:::", trying_out, flush=True)
                trying_out += 1

def final_post_processing(input_dir, table_file_name, save_output_dir_anon, sep=";"):
    file_path = input_dir + table_file_name
    table_header_list = []
    first_set_of_rows = []
    with open(file_path, mode ='r') as file:
        csvFile = csv.reader(file, delimiter=sep)
        line_counter = 0
        for row in csvFile:
            line_counter += 1
            if line_counter == 1:
                table_header_list = row
            if line_counter > 1:
                row[0] = re.sub('\d+\.', '', row[0])
                first_set_of_rows.append(";".join(row))
    if len(table_header_list) < 1:
        print("FILENAME:::", table_file_name, flush=True)
        df = pd.read_csv(file_path, sep='\s+')
        print("LEN BEFORE: ", len(df))
        df[df.columns[0]] = df[df.columns[0]].str.replace(r'\d+\.', '', regex=True)
        df.drop_duplicates(inplace=True)
        print("LEN AFTER: ", len(df))
        output_file_path = save_output_dir_anon + table_file_name
        df.to_csv(output_file_path, sep=sep)
        return
    all_rows = []
    for row in first_set_of_rows:
        if sep in row:
            curr_row = row.split(sep)
            if len(curr_row) > len(table_header_list):
                squished_row = curr_row[:len(table_header_list)-1]
                squished_row.append(" ".join(curr_row[len(table_header_list)-1:]))
                all_rows.append(squished_row)
            elif len(curr_row) < len(table_header_list):
                squished_row = curr_row[:len(table_header_list)]
                while len(squished_row) < len(table_header_list):
                    squished_row.append(None)
                all_rows.append(squished_row)
            else:
                all_rows.append(curr_row)
    print("FILENAME:::", table_file_name, flush=True)
    df = pd.DataFrame(all_rows, columns=table_header_list)
    print("LEN BEFORE: ", len(df))
    df.drop_duplicates(inplace=True)
    print("LEN AFTER: ", len(df))
    output_file_path = save_output_dir_anon + table_file_name
    df.to_csv(output_file_path, sep=sep)
    

    
    
    


def main():
    output_dir_anon = "../data/ben_y/"
    output_query_dir_anon = output_dir_anon + "/query/"
    output_dl_dir_anon = output_dir_anon + "/datalake/"
    gt = pd.read_csv(groundtruth_file)
    query_table = []
    os.makedirs(output_dir_anon, exist_ok=True)
    os.makedirs(output_query_dir_anon, exist_ok=True)
    os.makedirs(output_dl_dir_anon, exist_ok=True)
    data_lake_table = []
    for index, row in gt.iterrows():
        table_1 = row['query_table']
        table_2 = row['data_lake_table']
        if table_1 not in query_table:
            # generate first 10 rows
            add_first_10_rows(output_query_dir_anon, table_1, save_output_dir=output_query_dir_anon)
            # see if we missed anything we needed to grab for the first 10 rows using log files -- optional
            fill_missingcontent_first10rows(output_query_dir_anon, table_1, "log.txt", len(query_table), save_output_dir_anon=output_query_dir_anon)
            # add more rows -- optional (required if we want more than 10 rows)
            random_loops = random.randint(10, 20)
            for i in range(random_loops):
                add_more_rows(query_dir_anon, table_1, save_output_dir=query_dir_anon)
            # final processing after adding more rows -- optional
            final_post_processing(query_dir_anon, table_1, output_query_dir_anon, sep=";")
            query_table.append(table_1)
        if table_2 not in data_lake_table:
                        # generate first 10 rows
            add_first_10_rows(output_dl_dir_anon, table_2, save_output_dir=output_dl_dir_anon)
            # see if we missed anything we needed to grab for the first 10 rows using log files -- optional
            fill_missingcontent_first10rows(output_dl_dir_anon, table_2, "log.txt", len(data_lake_table), save_output_dir_anon=output_dl_dir_anon)
            # add more rows -- optional (required if we want more than 10 rows)
            random_loops = random.randint(10, 20)
            for i in range(random_loops):
                add_more_rows(output_dl_dir_anon, table_2, save_output_dir=output_dl_dir_anon)
            # final processing after adding more rows -- optional
            final_post_processing(output_dl_dir_anon, table_2, output_dl_dir_anon, sep=";")
            data_lake_table.append(table_2)
        


if __name__ == "__main__":
    main()
    