import json
import pandas as pd
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import argparse


def get_all_prob(structured_factors):
    structure_list = []
    structure_list_flat = []
    count_num = 0
    for key, value in structured_factors.items():
        structure_list_temp = []
        for v in value:
            structure_list_temp.append(v)
            structure_list_flat.append(v)
            count_num += 1
        structure_list.append(structure_list_temp)

    def dynloop(data, cur_y=0, final_lst=[], temp_lst=[]):
        max_y = len(data) - 1
        for x in range(len(data[cur_y])):
            temp_lst.append(data[cur_y][x])
            if cur_y == max_y:
                final_lst.append([*temp_lst])
            else:
                dynloop(data, cur_y + 1, final_lst, temp_lst)

            temp_lst.pop()

        return final_lst

    complete_test = dynloop(structure_list)

    prediction_x = []
    prediction_y = []

    for sample in complete_test:
        sent_index = [structure_list_flat.index(sent) for sent in sample]

        def evaluate(i):
            if i in sent_index:
                return 1
            else:
                return 0

        x = [evaluate(i) for i in range(len(structure_list_flat))]

        prediction_x.append(x)
        prediction_y.append(1)

    return structure_list_flat, complete_test, prediction_x, prediction_y


class Probnetwork(nn.Module):
    def __init__(self, input_prob):
        super(Probnetwork, self).__init__()
        self.para = nn.Parameter(torch.tensor(input_prob))

    def forward(self, x):
        current_para = torch.prod(self.para.masked_fill((1 - x).bool(), 1), 1)
        current_para_opposite = torch.prod(1 - self.para.masked_fill((1 - x).bool(), 0), 1)
        return current_para / (current_para + current_para_opposite)


def calculate_prob_for_sent(implied_values, value_prob_list):
    p_1_list = []
    for value_prob in value_prob_list:
        if set(implied_values).issubset(set(value_prob[0:-1])):
            p_1_list.append(value_prob[-1])

    if len(p_1_list) != 0:
        p_1 = sum(p_1_list) / len(p_1_list)
        p_2 = 1 - p_1
    else:
        p_1 = 0.5
        p_2 = 0.5

    if p_1 > 0.5:
        prediction = 'Statement 1'
    elif p_1 < 0.5:
        prediction = 'Statement 2'
    else:
        prediction = 'Unknown'

    return prediction, [p_1, p_2]


def get_outcome(mapping_addition_sentence_dic, value_prob_list):
    addition_sentence_final_prediction = dict()
    addition_sentence_final_probability = dict()

    for sent, value in mapping_addition_sentence_dic.items():
        if len(value) == 0:
            addition_sentence_final_prediction[sent], addition_sentence_final_probability[sent] = 'Unknown', [0.5,0.5]
        else:
            addition_sentence_final_prediction[sent], addition_sentence_final_probability[sent] = calculate_prob_for_sent(value, value_prob_list)
    return addition_sentence_final_prediction, addition_sentence_final_probability



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-70B-Instruct", help="select the model name")
    parser.add_argument("--dataset_name", type=str, default="common2sense", help="dataset name")
    parser.add_argument("--dataset_file_dic", type=str, default="/BIRD/data/", help="data file dictionary")
    parser.add_argument("--save_file_dic", type=str, default="/BIRD/results/",help="save file dictionary")
    parser.add_argument("--prob_type", type=str, default="all", help="saving file prob type")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)


    with open(args.dataset_file_dic + args.dataset_name + ".json") as f:
        df_origin = json.load(f)

    with open(args.save_file_dic + args.dataset_name + "_" + args.model_name.replace("/", "-") + "_0_direct.json")  as f:
        df_direct = json.load(f)

    with open(args.save_file_dic + args.dataset_name + "_" + args.model_name.replace("/", "-") + "_0_w_factors.json") as f:
        df_structure = json.load(f)

    with open(args.save_file_dic + args.dataset_name + "_" + args.model_name.replace("/", "-") + "_0_w_factors_w_condition_mapping.json") as f:
        df_factor = json.load(f)

    with open(args.save_file_dic + args.dataset_name + "_" + args.model_name.replace("/", "-") + "_0_w_factors_train.json") as f:
        df_train = json.load(f)

    df_direct_dict = dict()
    for i, df in enumerate(df_direct):
        df_direct_dict[df['scenario'] + df['statement']] = df

    df_structure_dict = dict()
    for i, df in enumerate(df_structure):
        df_structure_dict[df['scenario'] + df['statement']] = df

    df_factor_dict = dict()
    for i, df in enumerate(df_factor):
        df_factor_dict[df['scenario'] + df['statement']] = df

    df_train_dict = dict()
    for i, df in enumerate(df_train):
        df_train_dict[df['scenario'] + df['statement']] = df

    bird_known_count_correct = 0
    bird_known_model_direct_count_correct = 0
    model_direct_count_correct = 0
    bird_unknown_count_all = 0
    count_all = 0
    bird_unknown_type_1 = 0
    correct_all = 0
    out_objs = []

    for i, df in enumerate(df_origin):

        print(i)
        print(df['scenario'])
        print(df['statement'])
        print(df['opposite_statement'])

        # Initialization
        df_direct = df_direct_dict[df['scenario'] + df['statement']]
        df_structure = df_structure_dict[df['scenario'] + df['statement']]
        df_factor = df_factor_dict[df['scenario'] + df['statement']]
        df_train = df_train_dict[df['scenario'] + df['statement']]

        structured_factors = df_structure['structured_factors']
        mapping_addition_sentence_dic = df_factor['condition_factor_mapping']
        initial_prob = df_train["para_prob"]

        model = Probnetwork(input_prob=initial_prob)
        factor = []
        for key, value in structured_factors.items():
            for v in value:
                factor.append(v)
        for j in range(len(factor)):
            print(factor[j], initial_prob[j])

        # Creating datasets
        structure_list_flat, complete_test, test_x, test_y = get_all_prob(structured_factors)
        X_test_tensor = torch.FloatTensor(test_x)
        y_test_tensor = torch.FloatTensor(test_y)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)


        # Model inference for value prbability
        prob_prediction = []
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # No gradients needed for evaluation
            predictions = []
            actuals = []
            for inputs, targets in test_loader:
                outputs = model(inputs)
                predictions.extend(outputs.view(-1).tolist())
                actuals.extend(targets.view(-1).tolist())
                label_sent = [structure_list_flat[i] for i in
                              (inputs.squeeze(0) == 1).nonzero().reshape(1, -1).tolist()[0]]
                prob_prediction.append(label_sent + outputs.view(-1).tolist())

        addition_sentence_final_prediction, addition_sentence_final_probability = get_outcome(mapping_addition_sentence_dic, prob_prediction)


        if args.dataset_name == "common2sense":
            obj = {
                "scenario": df["scenario"],
                "statement": df["statement"],
                "opposite_statement": df["opposite_statement"],
                "added_information": df["added_information"],
                "oppo_added_information": df["oppo_added_information"],
                "additional_sentence_label": df["additional_sentence_label"],
                "direct_additional_sentence": df_direct['direct_additional_sentence'],
                "structured_factors": structured_factors,
                "factor_outcome_mapping": df_structure['factor_outcome_mapping'],
                "para_prob": initial_prob,
                "condition_factor_mapping": mapping_addition_sentence_dic
            }
        else:
            obj = {
                "scenario": df["scenario"],
                "statement": df["statement"],
                "opposite_statement": df["opposite_statement"],
                "additional_sentences": df['additional_sentences'],
                "additional_sentence_label": df['additional_sentence_label'],
                "direct_additional_sentence": df_direct['direct_additional_sentence'],
                "structured_factors": structured_factors,
                "factor_outcome_mapping": df_structure['factor_outcome_mapping'],
                "para_prob": initial_prob,
                "condition_factor_mapping": mapping_addition_sentence_dic
            }



        obj["addition_sentence_probability"] = defaultdict(dict)
        for sent, value in addition_sentence_final_prediction.items():
            if value == 'Unknown':
                bird_unknown_count_all += 1
                if len(mapping_addition_sentence_dic[sent]) == 0:
                    bird_unknown_type_1 += 1
                if df_direct['direct_additional_sentence'][sent] == df['additional_sentence_label'][sent]:
                    correct_all += 1
            else:
                if value == df['additional_sentence_label'][sent]:
                    bird_known_count_correct += 1
                    print(sent, " is correct")
                    correct_all += 1
                if df_direct['direct_additional_sentence'][sent] == df['additional_sentence_label'][sent]:
                    bird_known_model_direct_count_correct += 1

            if df_direct['direct_additional_sentence'][sent] == df['additional_sentence_label'][sent]:
                model_direct_count_correct += 1


            if args.prob_type == 'all':
                obj["addition_sentence_probability"][sent] = {
                    "gold_label": df['additional_sentence_label'][sent],
                    "prediction": addition_sentence_final_prediction[sent],
                    "mapped_values": mapping_addition_sentence_dic[sent],
                    "probability": addition_sentence_final_probability[sent]
                }

            elif args.prob_type == 'correct':
                if df_direct['direct_additional_sentence'][sent] != addition_sentence_final_prediction[sent]:
                    if addition_sentence_final_prediction[sent] == df['additional_sentence_label'][sent]:
                        obj["addition_sentence_probability"][sent] = {
                            "gold_label": df['additional_sentence_label'][sent],
                            "prediction": addition_sentence_final_prediction[sent],
                            "mapped_values": mapping_addition_sentence_dic[sent],
                            "probability": addition_sentence_final_probability[sent]
                        }

            elif args.prob_type == 'wrong':
                if df_direct['direct_additional_sentence'][sent] == df['additional_sentence_label'][sent]:
                    if addition_sentence_final_prediction[sent] != df['additional_sentence_label'][sent]:
                        if addition_sentence_final_prediction[sent] != 'Unknown':
                            obj["addition_sentence_probability"][sent] = {
                                "gold_label": df['additional_sentence_label'][sent],
                                "prediction": addition_sentence_final_prediction[sent],
                                "mapped_values": mapping_addition_sentence_dic[sent],
                                "probability": addition_sentence_final_probability[sent]
                            }

            elif args.prob_type == 'unknown':
                if addition_sentence_final_prediction[sent] == 'Unknown':
                    obj["addition_sentence_probability"][sent] = {
                        "gold_label": df_origin['additional_sentence_label'][sent],
                        "prediction": addition_sentence_final_prediction[sent],
                        "mapped_values": mapping_addition_sentence_dic[sent],
                        "probability": addition_sentence_final_probability[sent]
                    }



        count_all += len(addition_sentence_final_prediction)

        if args.prob_type == 'all':
            out_objs.append(obj)
        else:
            if len(obj["addition_sentence_probability"]) != 0:
                out_objs.append(obj)

        json.dump(out_objs, open(args.save_file_dic + args.dataset_name + "_" + args.model_name.replace("/","-") + "_" + args.prob_type + "_prob.json", "w"),indent=4)

        print("\n")

    print("Total number of Evaluated Instances: ", count_all)
    print("BIRD Unknown Percentage: ", bird_unknown_count_all / count_all)
    print("BIRD Known BIRD Accuracy: ", bird_known_count_correct / (count_all - bird_unknown_count_all))
    print("BIRD Known Model Direct Accuracy: ", bird_known_model_direct_count_correct / (count_all - bird_unknown_count_all))
    print("Model Direct Accuracy: ", model_direct_count_correct / count_all)
    print("General Accuracy with Mixed Methods: ", correct_all / count_all) # BIRD Known use BIRD, BIRD Unknown use DIRECT
