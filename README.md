# ALT-GEN
Codes and supplementary materials for Generative Benchmark Creation for Table Union Search

## ALT-GEN framework
To create your own table union search benchmark using ```Mixtral-8x7B-Instruct-v0.1```, you can first run the ```getSubjectCols.py``` and then run ```addRows.py``` scripts within the ```scripts``` folder.

## UGEN_V2 benchmark
The UGEN_V2 benchmark is located within the ```data``` folder (named as ```ugen_v2```).

## Starmie-LLM method
To run the Starmie-LLM method, you can run the script ```llm_prompting.py``` found within the ```scripts``` folder. Within this python file, you can replace ```MODEL_NAME``` with the LLM model that you would like to test out.

## result files
The ```evaluation``` folder contains code we used to evaluate our results from both existing and new table union search methods. The ```new_stats``` folder within this folder contains our result pickle files from our experiments.
The ```manual_benchmark_validation_results``` folder contains our manually validated results for UGEN_V1, UGEN_V2, and 100 sampled non-unionable pairs from TUS-Small.
