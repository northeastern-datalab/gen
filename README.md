# SYSTEM-A
Codes and supplementary materials for Generative Benchmark Creation for Table Union Search

## SYSTEM-A framework
To create your own table union search benchmark using GPT3, you can run the ```altGenRun.ipynb``` notebook within the ```notebook``` folder. You would need to update the first block with your OPEN_API_KEY.

## BEN-Y benchmark
The BEN-Y benchmark is located within the ```data``` folder (named as ```ben_y```).

## Starmie-LLM method
To run the Starmie-LLM method, you can run the script ```llm_prompting.py``` found within the ```scripts``` folder. Within this python file, you can replace ```MODEL_NAME``` with the LLM model that you would like to test out.

## result files
The ```evaluation``` folder contains code we used to evaluate our results from both existing and new table union search methods. The ```new_stats``` folder within this folder contains our result pickle files from our experiments.
The ```manual_benchmark_validation_results``` folder contains our manually validated results for BEN-Y and 100 sampled non-unionable pairs from TUS-Small.
