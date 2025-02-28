# This is the script for BIRD

cd /BIRD/code

# Vllm load model
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.1-70B-Instruct --tensor-parallel-size 4 --download-dir /cache --enable-prefix-caching --max_model_len=101888 --port 8000  &

# Step 1 Generate abductive factors
# You can choose from common2sense_human_annotation / common2sense / plasma / today
python -m run.scenario_analysis --dataset_name=common2sense_human_annotation --save_file_dic=your assigned directory --dataset_file_dic=/BIRD/data 

sleep 5h

# Step 2 (must wait for step 1 to first finish) Learn the CPT
python -m run.scenario_train --dataset_name=common2sense_human_annotation --save_file_dic=the same assigned directory as in step 1 --dataset_file_dic=/BIRD/data 

# Step 3 inference (can run simultaneously with step 2)
python -m run.scenario_inference --dataset_name=common2sense_human_annotation --save_file_dic=the same assigned directory as in step 1 --dataset_file_dic=/BIRD/data 