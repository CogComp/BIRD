# This is the script for evaluation

cd /BIRD/code

# Evaluation for probability estimation
# Step 1: calculation of the probability for each scenario with a condition
python -u evaluation/evaluation_probability_calculation_human_preprocess.py --dataset_name=common2sense_human_annotation --save_file_dic=your assigned directory --dataset_file_dic=/BIRD/data/ 
sleep 3m
# Step 2 (must wait for step 1 to first finish): evaluation for human annotated probabilistic comparison
python -u evaluation/evaluation_probability_calculation_human.py --dataset_name=common2sense_human_annotation --compare_w_model_direct=True --compare_w_model_compare=True --compare_w_bird=True --save_file_dic=your assigned directory --dataset_file_dic=/BIRD/data/ 


# Evaluation for decision making 
# You can choose from common2sense / plasma / today
python -u evaluation/evaluation_probability_calculation.py --dataset_name=plasma --save_file_dic=your assigned directory --dataset_file_dic=/BIRD/data/ 
