import time
import json
from collections import defaultdict
import argparse
import pandas as pd
from utils.utils import ask_gpt

def direct_probability(scenario, condition, outcome_1, outcome_2, temp=0.15, token=512, model_name="meta-llama/Llama-3.1-70B-Instruct"):
    message = [
        {
            "role": "system",
            "content": "You are an AI assistant that helps people make decisions. The user will give you a statement. Your task is to answer as faithfully as you can. While answering think step-by-step and justify your answer.\n\nYou are asked to assign a probability for each outcome based on the scenario and the additional condition. P(outcome 1) + P(outcome 2) =1 .\n"
        },
        {
            "role": "user",
            "content": "Scenario: The availability of parking spaces for bikes and cars at a local park is being compared.\nAdditional Condition: The local park is in an area where car use is prevalent and cycling is less common.\nOutcome 1: There is more parking available for bikes than cars at a local park.\nOutcome 2: There is more parking available for cars than bikes at a local park."
        },
        {
            "role": "assistant",
            "content": "Given the additional condition that the local park is in an area where car use is prevalent and cycling is less common, it would indeed necessitate more parking spaces for cars.\n\nOutcome 1: There is more parking available for bikes than cars at a local park. With the high prevalence of car use and less common cycling, this outcome is less likely. Therefore, P(outcome 1) = 0.375.\n\nOutcome 2: There is more parking available for cars than bikes at a local park. As the park is in an area where car use is prevalent, it is reasonable to expect that more parking will be available for cars. Therefore, P(outcome 2) = 0.625."
        },
    ]

    prompt = 'Scenario: ' + scenario + '\nAdditional Condition: ' + condition + '\nOutcome 1: ' + outcome_1 +'\nOutcome 2: ' + outcome_2


    message.append({"role": "user", "content": prompt})


    attempts = 0
    while attempts < 5:
        response = ask_gpt(message, use_temp=temp, max_token=token, model_name=model_name)

        if '\n\n### User:' in response:
            response = response.split('\n\n### User:')[0]
        try:
            p_1 = None
            p_2 = None

            if 'P(outcome 1) = ' in response:
                p_1 = response.split('P(outcome 1) = ')[1].split('\n')[0][0:-1]
            if 'P(outcome 2) = ' in response:
                p_2 = response.split('P(outcome 2) = ')[1].split('\n')[0][0:-1]

            p_1 = float(p_1)
            p_2 = float(p_2)
            break

        except ValueError:
            print("Error: Could not convert p_1 or p_2 to float.")

        attempts += 1


    return p_1, p_2, response

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-70B-Instruct", help="select the model name")
    parser.add_argument("--dataset_file_dic", type=str, default="/BIRD/data/", help="data file dictionary")
    parser.add_argument("--save_file_dic", type=str, default="/BIRD/results/",help="save file dictionary")
    parser.add_argument("--start", type=int, default=0, help="start instance")
    parser.add_argument("--end", type=int, default=10000, help="end instance")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)

    df = pd.read_csv(args.dataset_file_dic + "common2sense_human_annotation.csv")
    out_objs = []
    for i in range(len(df)):
        print(i)
        scenario = df['scenario'][i]
        statement_1 = df['statement_1'][i]
        statement_2 = df['statement_2'][i]
        condition_1 = df['sentence_1'][i]
        condition_2 = df['sentence_2'][i]

        c1_p_1, c1_p_2, c1_response = direct_probability(scenario, condition_1, statement_1, statement_2, model_name=args.model_name)
        c2_p_1, c2_p_2, c2_response = direct_probability(scenario, condition_2, statement_1, statement_2, model_name=args.model_name)

        print(condition_1, c1_p_1, c1_p_2)
        print(condition_2, c2_p_1, c2_p_2)

        obj = {
            "scenario": scenario,
            "statement": statement_1,
            "opposite_statement": statement_2,
            "addition_sentence_probability": {
                condition_1: {
                    "probability": [c1_p_1, c1_p_2],
                    "generation":c1_response
                },
                condition_2: {
                    "probability": [c2_p_1, c2_p_2],
                    "generation": c2_response
                }
            }}
        out_objs.append(obj)
        print('\n')

        json.dump(out_objs, open(args.save_file_dic + "common2sense_human_annotation_" + args.model_name.replace("/", "-") + "_direct.json", "w"),
                  indent=4)
