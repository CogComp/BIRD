import random
import time
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from utils.utils import ask_gpt

class Pipeline:

    def __init__(self, args):

        self.data_type = args.dataset_name
        self.model_name = args.model_name

        # LLM entailment for context-factor mapping parameters
        self.num_direct_prompt = 3

    def direct_inference(self,scenario, statement, oppo_statement, sent, temp=0.15):

        message = [
            {
                "role": "system",
                "content": """Given a scenario and an additional condition, identify which of two statements is supported by the additional condition.  You must follow the below structure to just generate json with an explanation.
                        For example:
                                    {
                                    "explanation":<you thinking process here>,
                                    "response":<output only Statement 1 or Statement 2>}"""
            },
            {
                "role": "user",
                "content": """Scenario: Complete a 10-week yoga teacher training program
                Condition: The yoga teacher training program has a strict no-adjustment policy for injuries.
                Statement 1: You would complete the yoga teacher training program without making any adjustments for your injury.
                Statement 2: You would adjust the yoga teacher training program to accommodate your injury and complete it."""
            },
            {
                "role": "assistant",
                "content": """{
                                    "explanation":"The condition suggests that the program does not allow for modifications or adjustments to be made for injuries, which implies that you would have to complete the program as is, without making any adjustments for your injury",
                                    "response": "Statement 1" }"""
            }
        ]



        prompt = "Scenario: " + scenario + "\nCondition:" + sent + "\nStatement 1: " + statement + "\nStatement 2: " + oppo_statement
        message.append({"role": "user", "content": prompt})

        attempts = 0
        final_response = dict()
        while attempts < 5:
            if attempts == 1:
                message.append({"role": "assistant", "content": response})
                message.append({"role": "user",
                                "content": "Your response cannot be loaded as json. Regenerate, remember to give an explanation first and then a final response and use double quotation marks."})
            response = ask_gpt(message, use_temp=temp, max_token=768, model_name=self.model_name)
            print(response)
            try:
                if '{' in response and '}' in response:
                    final_response = json.loads('{' + response.split('{')[1].split('}')[0] + '}')
                    break
            except json.JSONDecodeError:
                print(f"Failed to decode JSON on attempt {attempts + 1}")
            attempts += 1

        return final_response, response



    def run_condition_direct(self, scenario, statement, oppo_statement, conditions):

        obj = {
            "scenario": scenario,
            "statement": statement,
            "opposite_statement": oppo_statement,
        }

        obj["direct_additional_sentence"] = dict()
        obj["direct_additional_sentence_explanation"] = dict()


        for sent in conditions:
            print(sent)
            response, exp = self.direct_inference(scenario, statement, oppo_statement, sent)
            obj["direct_additional_sentence"][sent] = response['response']
            obj["direct_additional_sentence_explanation"][sent] = response


        return obj



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-70B-Instruct",help="select the model name")
    parser.add_argument("--dataset_name", type=str, default="plasma", help="dataset name")
    parser.add_argument("--dataset_file_dic", type=str, default="/BIRD/data/",
                        help="data file dictionary")
    parser.add_argument("--save_file_dic", type=str, default="/BIRD/results/",help="save file dictionary")
    parser.add_argument("--start", type=int, default=0, help="start instance")
    parser.add_argument("--end", type=int, default=1000, help="end instance")

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    pipe = Pipeline(args)

    with open(args.dataset_file_dic + args.dataset_name + ".json") as f:
        df_all = json.load(f)


    out_objs = []
    for i, df in enumerate(df_all):

        if i >= args.end: break
        if i < args.start: continue
        print(i)
        scenario = df['scenario']
        statement = df['statement']
        oppo_statement = df['opposite_statement']

        if args.dataset_name == 'common2sense':
            conditions = df["added_information"] + df["oppo_added_information"]
        else:
            conditions = df["additional_sentences"]


        obj = pipe.run_condition_direct(scenario, statement, oppo_statement, conditions)
        out_objs.append(obj)

        json.dump(out_objs, open(args.save_file_dic + args.dataset_name + "_" + args.model_name.replace("/", "-") + "_" +  str(args.start) + "_direct.json",
            "w"), indent=4)

        print('\n')
