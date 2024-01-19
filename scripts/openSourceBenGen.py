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
data_dir = "../data/"
model = LanguageModel('mistralai/Mixtral-8x7B-Instruct-v0.1', device_map='auto')

def gen_topics(num_topics=50, max_new_tokens=15000, temperature=1, remote=True, save_output_file_path=None):
    curr_run = 0
    generate_topics = f"[INST] Generate {num_topics} one-to-three word distinct subjects. Answer with each subject in a new line. [/INST]"
    with model.generate(max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature, remote=remote, remote_include_output=True) as generator:
        with generator.invoke(generate_topics):
            pass

    model_response = model.tokenizer.batch_decode(generator.output)
    print("model_response", model_response)
    topics = model_response[0].split('[/INST]')[-1].strip().replace("</s>", "").replace("<s>", "")
    curr_run += 1
    post_process_topics = re.split('\d+.', topics.replace("\n", " "))
    final_topics = []
    for topic in post_process_topics:
        topic = topic.strip()
        if topic:
            final_topics.append(topic)
    if len(set(final_topics)) < num_topics:
        required_num = num_topics - len(set(final_topics))
        past_chat = model_response[0]
        generate_more_topics = past_chat + f"[INST] Generate {required_num} more subjects apart from the ones mentioned. Answer with each subject in a new line. [/INST]"
        with model.generate(max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature, remote=remote) as generator2:
            with generator2.invoke(generate_more_topics):
                pass
        new_model_response = model.tokenizer.batch_decode(generator2.output)
        print("model_response", new_model_response)
        newer_topics = new_model_response[0].split('[\INST]')[-1].strip().replace("</s>", "")
        curr_run += 1
        post_process_topics = re.split('\d+.', newer_topics.replace("\n", " "))
        for topic in post_process_topics:
            topic = topic.strip()
            if topic:
                final_topics.append(topic)
            else:
                break
    if save_output_file_path is not None: 
        f = open(save_output_file_path,'w')
        f.write("Topics\n")
        for t in final_topics:
            f.write(t+"\n")
        f.close()
    return final_topics[:num_topics]


def ran_gen(size, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for x in range(size))


def gen_initial_column_headers(task, task_label, table_1, table_2, table_separation="semi-colon", save_output_dir=None, save_ground_truth=None):
    is_success_task = "cannot" if not task_label else "can"
    separator_char = ";"
    prompt = f"[INST]  Create 2 {table_separation}-separated table column headers. "
    # Table 1 Characteristics
    curr_topic_1 = table_1['topic']
    curr_columns_1 = table_1['columns']
    if "table" in table_1:
        prompt = f"[INST] Given the following table column headers for table 1, create table 2's column headers, "
    else:
        prompt += f"Table 1 has {curr_columns_1} columns on the topic of {curr_topic_1}. "
    # Table 2 Characteristics
    curr_columns_2 = table_2['columns']
    if "table" in table_1:
        prompt += f"where this table has {curr_columns_2} columns on the same topic, i.e., {curr_topic_1}."
    else:
        prompt += f"Table 2 has {curr_columns_2} columns on the same topic."
    # Topics for both tables (no need since topics will be same)
    #topics = [curr_topic_1, curr_topic_2]
    # Set task label in the prompt
    match_column_num = 0
    if task_label:
        match_column_num = random.randint(2, 8)
        prompt += f"They can be {task}ed because they have {match_column_num} semantically similar columns that can be aligned in both tables.  In other words, {match_column_num} of the columns in table 2 resemble columns in table 1. The remaining columns don't necessarily resemble any of the columns in table 1."
    else:
        prompt += f"They cannot be {task}ed because there are no columns in table 2 that are semantically similar to any columns in table 1 and vice-versa. In other words, none of the columns in table 2 resemble any of the columns in table 1."
    # Instruct format of answering the task
    if "table" in table_1:
        table_1_columns = table_1["table"]
        prompt += f"Table 1's column headers are {table_1_columns}."
    prompt += "Answer the above task in the following format:\n"
    if "table" in table_1:
        prompt += f"Table 2: <{table_separation}-separated table 2 column headers>\n\n"
    else:
        prompt += f"Table 1: <{table_separation}-separated table 1 column headers>\n\nTable 2:<{table_separation}-separated table 2 column headers>\n"
    prompt += "[/INST]"
    start_time = time.time()
    with model.generate(max_new_tokens=5000, do_sample=True, temperature=0.7, top_p=0.7, repetition_penalty=2.0, remote=False) as generator1:
        with generator1.invoke(prompt):
            pass
    new_prompt = model.tokenizer.batch_decode(generator1.output)[0]
    if "table" in table_1:
        if task_label:
            new_prompt += f"[INST] Verify that Table 2, has {curr_columns_2} columns and the tables can be unioned. Re-generate your answer with the corrected response. [\INST]"
        else:
            new_prompt += f"[INST] Verify that Table 2, has {curr_columns_2} columns and the tables cannot be unioned, i.e., none of the columns are semantically similar to any columns in table 1 and vice-versa. Re-generate your answer with the corrected response. [\INST]"
    else:
        new_prompt += f"[INST] Verify that Table 1 has {curr_columns_1} columns and Table 2 has {curr_columns_2} columns and the generated tables can be unioned. Re-generate your answer with the corrected response in the requested format. [\INST]"
    with model.generate(max_new_tokens=5000, do_sample=True, temperature=0.7, remote=False) as generator2:
        with generator2.invoke(new_prompt):
            pass
    end_time = time.time()
    time_taken = end_time - start_time
    print(f'Time taken: {time_taken} seconds', flush=True)
    prompt_output = model.tokenizer.batch_decode(generator2.output)[0]
    prompt_output = prompt_output.split("[/INST]")[-1]    
    table_inds_to_process = []
    if "table" in table_1:
        match_output = re.search(r"(?<=Table 2: )(.*)\n", prompt_output, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
        if match_output:
            curr_table = match_output.group(0)
            lines = curr_table.encode('utf-8').decode('utf8').strip().splitlines()
            correct_lines = [line for line in lines if separator_char in line]
            cleaned_table = ""
            if correct_lines:
                cleaned_table = correct_lines[0]
                checker_cleaned = cleaned_table.split(separator_char)
                if len(checker_cleaned) <= 1:
                    cleaned_table = " ".join(correct_lines)
            print(cleaned_table, flush=True)
            table_2['table'] = cleaned_table
            table_inds_to_process.append(1)
        else:
            return prompt, prompt_output, None, None, None
    else:
        match_output = re.search(r"(?<=Table 1: )(.*)(?=Table 2)", prompt_output, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
        if match_output:
            curr_table = match_output.group(0)
            lines = curr_table.encode('utf-8').decode('utf8').strip().splitlines()
            correct_lines = [line for line in lines if separator_char in line]
            cleaned_table = ""
            if correct_lines:
                cleaned_table = correct_lines[0]
                checker_cleaned = cleaned_table.split(separator_char)
                if len(checker_cleaned) <= 1:
                    cleaned_table = " ".join(correct_lines)
            table_1['table'] = cleaned_table
            print(cleaned_table, flush=True)
            table_inds_to_process.append(0)
        else:
            return prompt, prompt_output, None, None, None
        match_output = re.search(r"(?<=Table 2: ).*\n?", prompt_output, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
        if match_output:
            curr_table = match_output.group(0)
            lines = curr_table.encode('utf-8').decode('utf8').strip().splitlines()
            correct_lines = [line for line in lines if separator_char in line]
            cleaned_table = ""
            if correct_lines:
                cleaned_table = correct_lines[0]
                checker_cleaned = cleaned_table.split(separator_char)
                if len(checker_cleaned) <= 1:
                    cleaned_table = " ".join(correct_lines)
            print(cleaned_table, flush=True)
            table_2['table'] = cleaned_table
            table_inds_to_process.append(1)
        else:
            return prompt, prompt_output, None, None, None
  
    # Get groundtruth
    actual_table_index = 0
    curr_topic_1_file = "-".join(curr_topic_1.split(" "))
    for t in table_inds_to_process:
        curr_table = None
        if t == 0:
            curr_table = table_1
        elif t == 1:
            curr_table = table_2
        table_rows = curr_table['table'].split("\n")
        if save_output_dir is not None:
            if 'table_csv' in curr_table:
                continue
            ran_gen_id = ran_gen(8)
            output_file_gen = f"{curr_topic_1_file}_{ran_gen_id}.csv"
            print("FOR FILE:::-", output_file_gen)
            print("PROMPT-OUTPUT:-", prompt_output)
            output_file_path = os.path.join(save_output_dir, output_file_gen)
            if t == 0:
                table_1['table_csv'] = output_file_gen
            elif t == 1:
                table_2['table_csv'] = output_file_gen
            with open(output_file_path, 'a+') as csv_f:
                writer_csv_f = writer(csv_f, delimiter=separator_char)
                for t_rows in range(0, len(table_rows)):
                    writer_csv_f.writerow(table_rows[t_rows].split(separator_char))
    if save_ground_truth is not None:
        with open(save_ground_truth, 'a+') as gtf:
            writer_gtf = writer(gtf)
            task_gt_label = 0
            if task_label:
                task_gt_label = 1
            writer_gtf.writerow([table_1['table_csv'], table_2['table_csv'], task_gt_label, match_column_num])
    return prompt, prompt_output, table_1, table_2, match_column_num

def reset_table(topic):
    table = {}
    table['topic'] = topic
    table['columns'] = random.randint(10, 15)
    table['sparsity'] = random.uniform(0, 1)
    table['textuality'] = random.randint(1, 5)
    return table

def gen_union_benchmark_step1(log_file, benchmark_dir, groundtruth_csv, topics, data_lake_size=20, continuation=False):
    with open(data_dir + 'log.txt', 'a+') as f:
        if not os.path.exists(benchmark_dir):
            os.makedirs(benchmark_dir)
        output_dir = benchmark_dir + "data"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        groundtruth_csv = benchmark_dir + "groundtruth.csv"
        with open(groundtruth_csv, 'a+') as gtf:
            writer_gtf = writer(gtf)
            if not continuation:
                writer_gtf.writerow(["query_table", "data_lake_table", "unionable", "num_union_cols"])
            gtf.flush()
        curr_topic_counter = 0
        curr_range_counter = 0
        while curr_topic_counter < len(topics):
            print("At the first while loop: ", curr_topic_counter, flush=True)
            topic_index = curr_topic_counter
            task = "union"
            task_label = True
            topic = topics[topic_index]
            print(topic)
            table_1 = {}
            table_1 = reset_table(topic)
            table_2 = reset_table(topic)
            prompt, tables, table_1, table_2_copy, table_groundtruth = gen_initial_column_headers(task, task_label, table_1, table_2,
                                                                                 table_separation="semi-colon",
                                                                                 save_output_dir=output_dir,
                                                                                 save_ground_truth=groundtruth_csv)
            f.write(f"\nprompt: {prompt}\n")
            f.write(f"\ntables:\n")
            f.write(tables)
            f.flush()
            if table_1 is None:
                continue
            curr_range_counter = 1
            while curr_range_counter <= (data_lake_size - 1):
                i = curr_range_counter
                if i % 2 == 0:
                    task_label = True
                else:
                    task_label = False
                prompt, tables, table_1_copy, table_2_copy, table_groundtruth = gen_initial_column_headers(task, task_label, table_1, reset_table(topic),table_separation="semi-colon",save_output_dir=output_dir,save_ground_truth=groundtruth_csv)
                f.write(f"\nprompt: {prompt}\n")
                f.write(f"\ntables:\n")
                f.write(tables)
                f.flush()
                if table_1_copy is not None:
                    curr_range_counter += 1
                    print("curr_range_counter: ", curr_range_counter, flush=True)
                if (table_1 is not None) and (curr_range_counter == data_lake_size):
                    curr_topic_counter += 1
                    curr_range_counter = 0
                    print("curr_topic_counter: ", curr_topic_counter, flush=True)
                    break
    return

def main():
    num_topics=50
    output_file_path = data_dir + f"topics_{num_topics}.txt"
    # if topic file is present, we can just read that txt file
    topics = []
    if os.path.isfile(output_file_path):
        f = open(output_file_path,'r')
        lines = f.readlines()
        topics = [t.strip() for t in lines[1:num_topics+1]]
    else:
        topics = gen_topics(save_output_file_path=output_file_path, num_topics=num_topics)
    print("Topics", topics)
    print("Topics len", len(topics))
    log_file = data_dir + 'log.txt'
    benchmark_dir = data_dir + "ben_y/"
    os.makedirs(benchmark_dir, exist_ok=True)
    groundtruth_csv = benchmark_dir + "groundtruth.csv"
    gen_union_benchmark_step1(log_file, benchmark_dir, groundtruth_csv, topics, data_lake_size=20, continuation=True)


if __name__ == "__main__":
    main()
    