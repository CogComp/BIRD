# This is the script for evaluation

cd /BIRD/code

# Step 1: Evaluation for decision making and calculation of the probability for each scenario with a condition
# You can choose from common2sense_human_annotation / common2sense / plasma / today
python -u evaluation/evaluation_probability_calculation.py --dataset_name=common2sense_human_annotation --save_file_dic=your assigned directory --dataset_file_dic=/BIRD/data/ 

sleep 10m

# Step 2 (must wait for step 1 to first finish): evaluation for human annotated probabilistic comparison
python -u evaluation/evaluation_probability_calculation_human.py --dataset_name=common2sense_human_annotation --compare_w_model_direct=True --compare_w_model_compare=True --compare_w_bird=True --save_file_dic=your assigned directory --dataset_file_dic=/BIRD/data/ 