import time
import json
import pandas as pd
from collections import defaultdict
import argparse
from utils.utils import ask_gpt

def compare_probability(scenario, outcome, condition_1, condition_2, temp=0.15, token=512, model_name="meta-llama/Llama-3.1-70B-Instruct"):

    message = [{
                "role": "system",
                "content": """Can you decide which condition makes the outcome more likely to be true based on the scenario? If you think both conditions make the outcome equally likely to be true, you must only respond with "Both".
        Provide your explanation and condition.
        For example:
        Explanation: <your chain-of-thought reasoning process>.
        Answer: <you can only answer Condition 1, Statement 2, or Both>."""
            },
            {
                "role": "user",
                "content": "Scenario: Alex is training for a marathon and is deciding whether to focus more on endurance runs or speed workouts.\nOutcome: Since Alex wants to maximize his marathon performance, he decides to prioritize endurance runs over speed workouts.\nCondition 1: Alex understands that long-distance runners typically incorporate endurance runs into their training routines.\nCondition 2: Alex reads a study showing that endurance training significantly improves marathon performance by increasing aerobic capacity and muscular endurance more effectively than speed workouts."
            },
            {
                "role": "assistant",
                "content": "Explanation: Both conditions support the outcome that Alex prioritizes endurance runs over speed workouts. Condition 1 highlights a general understanding that long-distance runners often use endurance runs, but it does not explain why they are beneficial or more effective than speed workouts. In contrast, Condition 2 presents a study with specific evidence that endurance training is more effective for marathon performance by improving aerobic capacity and muscular endurance. Since Condition 2 provides a concrete, research-backed reason, it more strongly justifies Alex’s decision.\nAnswer: Condition 2"
            },
            {
                "role": "user",
                "content": "Scenario: Liam is deciding whether to carry an umbrella before leaving for work.\nOutcome: Since Liam wants to avoid getting wet, he decides to bring an umbrella.\nCondition 1: Liam checks the weather forecast and sees that rain is expected today.\nCondition 2: Liam sees that the forecast predicts precipitation throughout the day."
            },
            {
                "role": "assistant",
                "content": "Explanation: Both conditions provide the same reasoning for Liam's decision—they both indicate that rain is expected, just phrased slightly differently. Condition 1 explicitly states that rain is expected, while Condition 2 mentions precipitation, which means the same thing. Since both conditions convey the same information and lead to the same conclusion, neither provides a stronger justification than the other.\nAnswer: Both"
            }
    ]

    prompt = 'Scenario: ' + scenario + '\nOutcome: ' + outcome + '\nCondition 1: ' + condition_1 +'\nCondition 2: ' + condition_2


    message.append({"role": "user", "content": prompt})
    response = ask_gpt(message, use_temp=temp, max_token=token, model_name=model_name)

    if '\n\n### User:' in response:
        response = response.split('\n\n### User:')[0]
    answer = response.split('\nAnswer: ')[-1]


    if ('Both' in answer or 'both' in answer or 'equally likely' in answer) and 'While both' not in answer:
        predict = 3
    elif 'Condition 1' in answer:
        predict = 1
    elif 'Condition 2' in answer:
        predict = 2
    else:
        predict = 3

    print(response)
    print(answer)
    print(predict)



    return predict, response



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

    df = pd.read_csv( args.dataset_file_dic + "common2sense_human_annotation.csv")
    out_objs = []
    count_correct = 0
    for i in range(len(df)):
        if i >= args.end: break
        if i < args.start: continue
        print(i)
        scenario = df['scenario'][i]
        statement_1 = df['statement_1'][i]
        statement_2 = df['statement_2'][i]
        condition_1 = df['sentence_1'][i]
        condition_2 = df['sentence_2'][i]
        statement = df['gold_statement'][i]

        answer, c1_response = compare_probability(scenario, statement, condition_1, condition_2, model_name=args.model_name)

        if answer == df['human_prediction'][i]:
            count_correct += 1

        obj = {
            "scenario": scenario,
            "statement": statement_1,
            "opposite_statement": statement_2,
            "gold_statement": statement,
            "sentence_1": condition_1,
            "sentence_2": condition_2,
            "answer": answer,
            "generation": c1_response}
        out_objs.append(obj)
        json.dump(out_objs,
                  open(args.save_file_dic + "common2sense_human_annotation_" + args.model_name.replace("/", "-") + "_compare.json", "w"),
                  indent=4)
        print('\n')

    print("Total percentage of correct answer:", count_correct/len(df))


