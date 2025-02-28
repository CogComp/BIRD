import json
from itertools import combinations
from collections import defaultdict
import pandas as pd
import math
import copy
from sklearn.metrics import f1_score
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-70B-Instruct", help="select the model name")
    parser.add_argument("--dataset_name", type=str, default="common2sense", help="dataset name")
    parser.add_argument("--dataset_file_dic", type=str, default="/BIRD/data/", help="data file dictionary")
    parser.add_argument("--save_file_dic", type=str, default="/BIRD/results/",help="save file dictionary")
    parser.add_argument("--compare_w_bird", type=bool, default=False, help="compare performance with our proposed method BIRD")
    parser.add_argument("--compare_w_model_direct", type=bool, default=False, help="compare performance with model directly output a proability for each condition")
    parser.add_argument("--compare_w_model_compare", type=bool, default=False,help="compare performance with model directly compare the two conditions")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print(args)


    # load human evaluation data
    df_human = pd.read_csv(args.dataset_file_dic + "common2sense_human_annotation.csv")
    with open(args.dataset_file_dic + "common2sense_human_annotation.json") as f:
        df_human_json = json.load(f)
    df_human_json_dict = dict()
    for i, df in enumerate(df_human_json):
        df_human_json_dict[df['scenario'] + df['statement']] = df


    # load direct model inference if required
    if args.compare_w_model_compare:
        with open(args.save_file_dic + "common2sense_human_annotation_"+ args.model_name.replace("/","-") +"_compare.json") as f:
            df_model_compare = json.load(f)

    if args.compare_w_model_direct:
        with open(args.save_file_dic + "common2sense_human_annotation_"+ args.model_name.replace("/","-") +"_direct.json") as f:
            df_model_direct = json.load(f)

    # load inference from BIRD
    if args.compare_w_bird:
        with open(args.save_file_dic + args.dataset_name + "_" + args.model_name.replace("/","-") + "_all_prob.json") as f:
            df_bird = json.load(f)

        df_bird_dict = dict()
        for i, df in enumerate(df_bird):
            if df['scenario'] + df['statement'] not in df_human_json_dict:
                continue
            else:
                df_human_json = df_human_json_dict[df['scenario'] + df['statement']]
            new_addition_sentence_probability = dict()
            for sent in df_human_json["additional_sentences"]:
                temp = {
                    "mapped_values": [],
                    "probability": [0.5, 0.5]
                }
                if sent in df["addition_sentence_probability"].keys():
                    temp["mapped_values"] = df["addition_sentence_probability"][sent]["mapped_values"]
                    temp["probability"] = df["addition_sentence_probability"][sent]["probability"]
                new_addition_sentence_probability[sent] = temp
            df["addition_sentence_probability"] = new_addition_sentence_probability
            df["additional_sentences"] = []
            df_bird_dict[df['scenario'] + df['statement']] = df

    # Performance analysis
    human_prediction = []

    if args.compare_w_model_compare:
        model_compare_prediction = []
    if args.compare_w_model_direct:
        model_direct_prediction = []
    if args.compare_w_bird:
        count_unknown = 0
        bird_prediction = []
        instance_w_missing_factors = []



    for i in range(len(df_human)):
        scenario = df_human['scenario'][i]
        statement_1 = df_human['statement_1'][i]
        statement_2 = df_human['statement_2'][i]
        condition_1 = df_human['sentence_1'][i]
        condition_2 = df_human['sentence_2'][i]
        statement = df_human['gold_statement'][i]
        if args.compare_w_bird:
            if scenario + statement_1 not in df_bird_dict:
                count_unknown += 1
                instance_w_missing_factors.append(df_human_json_dict[scenario + statement_1])
                continue

            df_bird = df_bird_dict[scenario + statement_1]
            origin_probability_1 = df_bird['addition_sentence_probability'][condition_1]["probability"]
            origin_probability_2 = df_bird['addition_sentence_probability'][condition_2]["probability"]
            if statement == statement_1:
                probability_1 = origin_probability_1[-2]
                probability_2 = origin_probability_2[-2]
            elif statement == statement_2:
                probability_1 = origin_probability_1[-1]
                probability_2 = origin_probability_2[-1]

            # Identify instances where BIRD outputs "UNKNOWN" due to a failure in mapping conditions to factors.
            flag = 0
            if probability_1 == 0.5 and len(df_bird['addition_sentence_probability'][condition_1]["mapped_values"]) == 0:
                flag += 1
                df_bird["additional_sentences"].append(condition_1)
            if probability_2 == 0.5 and len(df_bird['addition_sentence_probability'][condition_2]["mapped_values"]) == 0:
                flag += 1
                df_bird["additional_sentences"].append(condition_2)
            if flag != 0:
                count_unknown += 1
                continue

            if probability_1 > probability_2:
                bird_prediction.append(1)
            elif probability_2 > probability_1:
                bird_prediction.append(2)
            else:
                bird_prediction.append(3)


        if args.compare_w_model_compare:
            model_compare_prediction.append(df_model_compare[i]['answer'])

        if args.compare_w_model_direct:
            model_probability_1 = df_model_direct[i]['addition_sentence_probability'][condition_1]["probability"]
            model_probability_2 = df_model_direct[i]['addition_sentence_probability'][condition_2]["probability"]
            if statement == statement_1:
                probability_1 = float(model_probability_1[-2])
                probability_2 = float(model_probability_2[-2])
            elif statement == statement_2:
                probability_1 = float(model_probability_1[-1])
                probability_2 = float(model_probability_2[-1])
            if probability_1 > probability_2:
                model_direct_prediction.append(1)
            elif probability_2 > probability_1:
                model_direct_prediction.append(2)
            else:
                model_direct_prediction.append(3)

        human_prediction.append(df_human['human_prediction'][i])



    print("Total number of evaluated instances: {}".format(len(human_prediction)))

    if args.compare_w_model_compare:
        print("Model Compare Individual F1 score for each category: ",f1_score(model_compare_prediction, human_prediction, average=None))
        print("Model Compare Average F1 score: ", f1_score(model_compare_prediction, human_prediction, average='micro'))

    if args.compare_w_model_compare:
        print("Model Direct Individual F1 score for each category: ",f1_score(model_direct_prediction, human_prediction, average=None))
        print("Model Direct Average F1 score: ", f1_score(model_direct_prediction, human_prediction, average='micro'))


    if args.compare_w_bird:
        print("Number of instances where BIRD outputs unknown: {}".format(count_unknown))
        print("BIRD Individual F1 score for each category: ", f1_score(bird_prediction, human_prediction, average=None))
        print("BIRD Average F1 score: ", f1_score(bird_prediction, human_prediction, average='micro'))

        # Save instances where BIRD outputs "unknown" to either perform another round of BIRD inference or request human annotations for the condition-factor mapping.
        final_objs_list = []
        for key, value in df_bird_dict.items():
            if value["additional_sentences"] != []:
                value["additional_sentences"] = list(set(value["additional_sentences"]))
                final_objs_list.append(value)
        json.dump(final_objs_list, open(
            args.save_file_dic + args.dataset_name + "_human_eval_missing_mapping_values.json","w"),indent=4)
        json.dump(instance_w_missing_factors, open(
            args.save_file_dic + args.dataset_name + "_human_eval_missing_mapping_factors.json","w"), indent=4)









