import random
import time
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from utils.utils import ask_gpt

def llm_outcome_analysis(answer_outcome1, answer_outcome2):
    """
    This function is designed to ensure consistent judgment when LLMs estimate coarse decisions under complete information. Specifically, it addresses potential discrepancies in the outcomes.
    For instance, if a LLM assesses the probability  p(O1 | f ) as 'very likely' and p(O2 | f ) as 'unlikely' in rare cases, manual adjustments will be made to set p(O1 | f ) to 'likely'.
    This adjustment aims to rectify any inconsistencies and prevent overconfidence in the model's predictions.
    """

    outcome_pair = {
        "Very likely": "Very unlikely",
        "Likely": "Unlikely",
        "Somewhat likely": "Somewhat unlikely",
        "Neutral": "Neutral",
        "Somewhat unlikely": "Somewhat likely",
        "Unlikely": "Likely",
        "Very unlikely": "Very likely"
    }

    def judge(answer):
        if "Very unlikely" in answer:
            answer = "Very unlikely"
        elif "Unlikely" in answer:
            answer = "Unlikely"
        elif "Somewhat unlikely" in answer:
            answer = "Somewhat unlikely"
        elif "Neutral" in answer:
            answer = "Neutral"
        elif "Somewhat likely" in answer:
            answer = "Somewhat likely"
        elif "Likely" in answer:
            answer = "Likely"
        elif "Very likely" in answer:
            answer = "Very likely"
        else:
            answer = "Neutral"
        return answer

    answer_outcome1 = judge(answer_outcome1)
    answer_outcome2 = judge(answer_outcome2)

    if outcome_pair[answer_outcome1] != answer_outcome2:
        if answer_outcome1 == answer_outcome2:
            answer_outcome1 = "Neutral"
            answer_outcome2 = "Neutral"
        elif "Somewhat likely" in answer_outcome1 or "Somewhat unlikely" in answer_outcome2:
            answer_outcome1 = "Somewhat likely"
            answer_outcome2 = "Somewhat unlikely"
        elif "Somewhat unlikely" in answer_outcome1 or "Somewhat likely" in answer_outcome2:
            answer_outcome1 = "Somewhat unlikely"
            answer_outcome2 = "Somewhat likely"
        elif "Likely" in answer_outcome1 or "Unlikely" in answer_outcome2:
            answer_outcome1 = "Likely"
            answer_outcome2 = "Unlikely"
        elif "Unlikely" in answer_outcome1 or "Likely" in answer_outcome2:
            answer_outcome1 = "Unlikely"
            answer_outcome2 = "Likely"

    return answer_outcome1, answer_outcome2


def get_llm_approximate_prob(structured_factors, scenario, outcome1, outcome2, total_num=128, use_temp=0.7,
                             max_token=256, model_name="meta-llama/Llama-3.1-70B-Instruct"):
    message = [
        {
            "role": "system",
            "content": "As an AI assistant, your role is to respond accurately to user queries. While answering think step-by-step and justify your answer.\nAnalyze the given scenario and condition to determine the likelihood of the outcomes. Use only the information provided, without relying on prior knowledge. Assess the probability using the specified terms: 'very likely', 'likely', 'somewhat likely', 'neutral', 'somewhat unlikely', 'unlikely', 'very unlikely'. Ensure that your assessments are complementary: if one outcome is deemed 'likely', the other must be 'unlikely', and so on.  You should first give your reasons and then format your final answer."
        },
        {
            "role": "user",
            "content": "Scenario:  Dave was a scientist. Dave wanted to make a great scientific discovery. Dave worked with algae to make electricity. Dave discovered he could make electricity with algae! Dave was awarded for his great discovery.\nCondition: The experiment can be self-funded. Dave tends to plan as needed.\nOutcome 1: Dave applied for a grant for his project starts before Dave worked with algae to make electricity\nOutcome 2: Dave applied for a grant for his project starts after Dave worked with algae to make electricity"
        },
        {
            "role": "assistant",
            "content": "Since Dave can self-fund and plans as he goes, he may not need a grant to start his work. However, after a major discovery, like generating electricity from algae, applying for a grant could be more strategic, as his success would attract funders.\nTherefore,\nOutcome 1: Somewhat unlikely\nOutcome 2: Somewhat likely"
        },
        {
            "role": "user",
            "content": "Scenario: You want to move around with your cell phone when it is being charged.\nCondition: The charger is portable. The user stays very close to the charger. The user carries the charger.\nOutcome 1: You can move around more freely with your cell phone with a one feet cord rather than a six feet cord.\nOutcome 2: You can move around more freely with your cell phone with a six feet cord rather than a one feet cord."
        },
        {
            "role": "assistant",
            "content": "A one-foot cord improves mobility for users with portable chargers, while a six-foot cord may hinder movement due to tangling or excess length.\nTherefore, \nOutcome 1: Likely\nOutcome 2: Unlikely"
        }
    ]

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

    all_list = dynloop(structure_list)


    # sample training data of total_num size
    if len(all_list) <= total_num:
        sampled_train = (all_list * (int(total_num / len(all_list)) + 1))[0:total_num]
    else:
        final_total_num = total_num
        sampled_train = random.sample(all_list, final_total_num)


    prediction_x = []
    prediction_y = []

    log = []

    for sample in sampled_train:

        condition = ''
        for sent in sample:
            if sent[-1] != '.':
                condition += sent + '. '
            else:
                condition += sent + ' '
        condition = condition[0:-1]

        # ask for LLM response
        prompt = "Scenario: " + scenario + "\nCondition: " + condition + "\nOutcome 1: " + outcome1 + "\nOutcome 2: " + outcome2
        message.append({"role": "user", "content": prompt})

        answer = ''
        count = 0
        label = ''

        while (not ("Outcome 1:" in answer and "\nOutcome 2:" in answer)):
            response = ask_gpt(message, use_temp=use_temp, max_token=max_token, model_name=model_name)
            if "\nTherefore, \n" in response:
                answer = response.split("\nTherefore, \n")[-1]
            else:
                answer = response
            count += 1
            if count == 5:
                break
        message.pop()

        if (not ("Outcome 1:" in answer and "\nOutcome 2:" in answer)):
            continue

        elif len(answer.split("Outcome 1: ")) >= 2 and len(answer.split("\nOutcome 2: ")) >= 2:
            answer_outcome1 = answer.split("Outcome 1: ")[1].split("\nOutcome 2:")[0]
            answer_outcome2 = answer.split("\nOutcome 2: ")[1]
            answer_outcome1, answer_outcome2 = llm_outcome_analysis(answer_outcome1, answer_outcome2)
            if "Very unlikely" in answer_outcome1:
                y = 0
            elif "Unlikely" in answer_outcome1:
                y = 0.2
            elif "Somewhat unlikely" in answer_outcome1:
                y = 0.4
            elif "Neutral" in answer_outcome1:
                y = 0.5
            elif "Somewhat likely" in answer_outcome1:
                y = 0.6
            elif "Likely" in answer_outcome1:
                y = 0.8
            elif "Very likely" in answer_outcome1:
                y = 1
            prediction_y.append(y)

            sent_index = [structure_list_flat.index(sent) for sent in sample]

            def evaluate(i):
                if i in sent_index:
                    return 1
                else:
                    return 0

            x = [evaluate(i) for i in range(len(structure_list_flat))]
            prediction_x.append(x)

            temp = {
                "condition": sample,
                "response": response,
                "index_condition_flat": x,
                "outcome 1": answer_outcome1,
                "outcome 2": answer_outcome2,
                "label_condition": y
            }
            log.append(temp)

    return log, prediction_x, prediction_y


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


def calculate_mse_loss(outputs, target):
    mse_loss = nn.MSELoss()
    output = mse_loss(outputs, target)
    return output


def calculate_margin_loss(para, origin, target):
    # para = torch.Tensor([0.75, 0.05, 0.75, 0.25, 0.75, 0.5, 0.25])
    # origin =  torch.Tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    # target = torch.LongTensor([1,-1,1,-1,1,1,-1])

    margin_loss = nn.MarginRankingLoss()
    output = margin_loss(para, origin, target)
    return output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-70B-Instruct", help="select the model name")
    parser.add_argument("--dataset_name", type=str, default="plasma", help="dataset name")
    parser.add_argument("--save_file_dic", type=str, default="/shared/fengyu/BIRD/results/", help="save file dictionary")
    parser.add_argument("--sample_num", type=int, default=128, help="sample size")
    parser.add_argument("--use_temp", type=int, default=0.7, help="LLM's temperature")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--epoch", type=int, default=20, help="training epoch")
    parser.add_argument("--lr", type=int, default=1e-2, help="learning rate")
    parser.add_argument("--start", type=int, default=0, help="start instance")
    parser.add_argument("--end", type=int, default=10000, help="end instance")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)



    with open(args.save_file_dic + args.dataset_name + "_" + args.model_name.replace("/", "-") + "_0_w_factors.json") as f:
        df_list = json.load(f)

    out_objs = []
    for i, df in enumerate(df_list):
        if i < args.start:
            continue
        if i >= args.end:
            break
        print(i)
        scenario = df['scenario']
        outcome1 = df['statement']
        outcome2 = df['opposite_statement']
        structured_factors = df['structured_factors']
        factor_outcome_mapping = df['factor_outcome_mapping']

        if len(structured_factors) == 0:
            objs = {
                "scenario": scenario,
                "statement": outcome1,
                "opposite_statement": outcome2,
                "structured_factors": df['structured_factors'],
                "sampled_train": [],
                "train_x": [],
                "train_y": [],
                "para_prob": [],

            }

            out_objs.append(objs)
            continue

        def get_initial_prob(structured_factors, factor_outcome_mapping):
            initial_prob = []
            for key, value in structured_factors.items():
                for v in value:
                    mapped_value = factor_outcome_mapping[v]
                    if mapped_value == 'Statement 1':
                        initial_prob.append(0.75)
                    elif mapped_value == 'Statement 2':
                        initial_prob.append(0.25)
                    elif mapped_value == 'Undecided' or mapped_value == 'Neutral':
                        initial_prob.append(0.5)
            return initial_prob
        initial_prob = get_initial_prob(structured_factors, factor_outcome_mapping)



        origin = torch.Tensor([0.5] * len(initial_prob))
        def evaluate(i):
            if i - 0.5 > 0:
                return 1
            elif i - 0.5 < 0:
                return -1
            elif i - 0.5 == 0:
                return 0
        target = torch.LongTensor([evaluate(i) for i in initial_prob])

        model = Probnetwork(input_prob=initial_prob)

        # Loss and optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)


        # data preparation
        print("Preparing training data...")
        sampled_train, train_x, train_y = get_llm_approximate_prob(structured_factors, scenario, outcome1, outcome2,
                                                                   total_num=args.sample_num, use_temp=args.use_temp, model_name=args.model_name)

        structure_list_flat, complete_test, test_x, test_y = get_all_prob(structured_factors)

        X_train_tensor = torch.FloatTensor(train_x)
        y_train_tensor = torch.FloatTensor(train_y)
        X_test_tensor = torch.FloatTensor(test_x)
        y_test_tensor = torch.FloatTensor(test_y)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        # Creating DataLoaders
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

        if X_train_tensor.shape[1] != X_test_tensor.shape[1]:
            objs = {
                "scenario": scenario,
                "statement": outcome1,
                "opposite_statement": outcome2,
                "structured_factors": df['structured_factors'],
                "sampled_train": [],
                "train_x": [],
                "train_y": [],
                "para_prob": [],
            }

            out_objs.append(objs)
            continue

        # Training loop
        print("Start training...")
        losses = []
        epochs = args.epoch
        for epoch in range(epochs):
            for inputs, targets in train_loader:
                # Forward pass
                outputs = model(inputs)

                test_outputs = model(X_test_tensor.to(inputs.device))
                for j in range(len(initial_prob)):
                    elements_are_ones = X_test_tensor.to(inputs.device)[:, j] == 1
                    line_indices = torch.nonzero(elements_are_ones, as_tuple=True)[0]
                    prob = torch.mean(test_outputs[line_indices])
                    if j == 0:
                        loss_advance_mr = calculate_margin_loss(prob, origin[j], target[j])
                    else:
                        loss_advance_mr += calculate_margin_loss(prob, origin[j], target[j])
                loss_advance_mr /= len(initial_prob)

                loss_mse = calculate_mse_loss(outputs, targets)
                loss = loss_mse + 10 * loss_advance_mr

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

        # Evaluate the model with test data to see how well it performs
        prob_predict = []
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
                prob_predict.append(label_sent + outputs.view(-1).tolist())

        objs = {
            "scenario": scenario,
            "statement": outcome1,
            "opposite_statement": outcome2,
            "structured_factors": df['structured_factors'],
            "sampled_train": sampled_train,
            "train_x": train_x,
            "train_y": train_y,
            "para_prob": model.para.tolist(),
        }

        out_objs.append(objs)

        print("Finished value probability calculation.")
        for j in range(len(structure_list_flat)):
            print(structure_list_flat[j], model.para.tolist()[j])


        json.dump(out_objs, open(
            args.save_file_dic + args.dataset_name + "_" + args.model_name.replace("/", "-") + "_" + str(args.start) + "_w_factors_train.json",
            "w"), indent=4)

        print('\n')





