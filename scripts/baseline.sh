# This is the script for model direct baselines

cd /BIRD/code

# Vllm load model
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.1-70B-Instruct --tensor-parallel-size 4 --download-dir /cache --enable-prefix-caching --max_model_len=101888 --port 8000  &

# Baselines for experiment 1: human annotated probability comparison
# Direct
python -m baseline.model_probability_direct --save_file_dic=your assigned directory --dataset_file_dic=/BIRD/data 

# Compare
python -m baseline.model_probability_compare --save_file_dic=your assigned directory --dataset_file_dic=/BIRD/data 

# Baselines for experiment 2: decision making
# You can choose from common2sense / plasma / today
python -m baseline.model_direct --dataset_name=plasma --save_file_dic=your assigned directory --dataset_file_dic=/BIRD/data 