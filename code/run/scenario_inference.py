import random
import time
import json
from collections import defaultdict
import argparse
from utils.utils import ask_gpt

class Pipeline:

    def __init__(self, args):

        self.data_type = args.dataset_name
        self.model_name = args.model_name

        # LLM entailment for context-factor mapping parameters
        self.num_direct_prompt = 3

    def run_combined_addition_sentence_judgement(self, factor_dict, sent, scenario, temp=0.15):

        message = [
            {
                "role": "system",
                "content": """Given a scenario, read a sentence and determine whether a value from a given list is implied by the sentence. You must be strict on considering a value as implied and return one value or "None" with an explanation.
                Here are rules that you must follow:
                1. A value is implied only if the sentence implies it in most situations, meaning the sentence provides clear, direct information relevant to the value.
                2. Do not select a value based on indirect associations, context, or related hints unless they strongly imply the value in most situations.
                3. Do not assume causality or infer a connection if it is not clearly stated in the sentence.
                4. If the sentence implies more than one value, choose the most implied one. If two values are both implied, choose the one that is positive (non-negative).
                5. If the sentence does not imply any value directly in most situations, say "None".
                """
            },
            {
                "role": "user",
                "content": """Here is a scenario: complete in a football game.
                Sentence: "Your injury is minor and unlikely to impact your performance significantly."
                List: "The nature of the injury": [
                "The injury is severe and permanent",
                "The injury is unpredictable and fluctuating",
                "The injury is chronic but manageable",
                "The injury is minor and temporary",
                "The injury is not minor and temporary",
                "The injury is not severe and permanent"
                ]"""
            },
            {
                "role": "assistant",
                "content": """{"explanation": "The sentence clearly mentions that the injury is minor and does not significantly affect performance. Therefore, the injury is minor and temporary based on the list.",
                "answer": "The injury is minor and temporary"}"""
            },
            {
                "role": "user",
                "content": """Here is a scenario: Dave was a scientist. Dave wanted to make a great scientific discovery. Dave worked with algae to make electricity. Dave discovered he could make electricity with algae! Dave was awarded for his great discovery.
                Sentence: "Dave is known to meticulously plan his investigations and ensure all necessary resources and funds are obtained beforehand."
                List: "Dave's typical planning style": [
                "Dave tends to plan ahead",
                "Dave tends to plan as needed",
                "Dave does not usually plan",
                ]"""
            },
            {
                "role": "assistant",
                "content": """{"explanation": "The scenario and sentence indicate that Dave meticulously plans his investigations and ensures all necessary resources and funds are obtained beforehand. This suggests that Dave is proactive and plans ahead of time.",
                "answer": "Dave tends to plan ahead"}"""
            },
            {
                "role": "user",
                "content": """Here is a scenario: You want to move around with your cell phone when it is being charged.
                Sentence: "The user stays very close to the charger."
                List: "The cell phone's charging method": [
                "The charger is portable", 
                "The charger is unmovable"
                ]"""
            },
            {
                "role": "assistant",
                "content": """{"explanation": "The sentence under the scenario doesn't directly imply anything about the charger's portability or movement. Staying close to the charger does not confirm whether the charger is portable or immovable.",
                "answer": "None"}"""
            },
            {
                "role": "user",
                "content": """Here is a scenario: volunteer experience.
                Sentence: "You have volunteer experience that is specialized and in high demand in your local area."
                List: "Information sources": [
                "You rely on local news and community newspapers for information on charitable causes",
                "You rely on global news and international media for information on charitable causes",
                "You rely on both local and international news sources"
                ]"""
            },
            {
                "role": "assistant",
                "content": """{"explanation": "The sentence focuses on the user's specialized volunteer experience in their local area but does not directly address how they gather information on charitable causes.",
                "answer": "None"}"""
            },
            {
                "role": "user",
                "content": """Here is a scenario: The program includes various modes of learning. 
                Sentence: "You reside in a region where internet connectivity is unreliable."
                List: "The program's features": [
                "The program includes self-paced online learning.",
                "The program includes hands-on practice teaching and feedback.",
                "The program is delivered entirely online."
                ]"""
            },
            {
                "role": "assistant",
                "content": """{"explanation": "The sentence mentions unreliable internet connectivity, which could make some modes of learning difficult, but it does not explicitly state or strongly imply any specific feature of the program, such as hands-on teaching or being entirely online.",
                "answer": "None"}"""
            }
        ]

        if scenario and not scenario[0].islower():
            scenario_temp = scenario.lower()
        else:
            scenario_temp = scenario
        if scenario_temp.endswith('.'):
            scenario_temp = scenario_temp[:-1]

        prompt = "Here is a scenario: " + scenario_temp + "\nSentence:" + sent + "\nList: " + str(json.dumps(factor_dict))
        message.append({"role": "user", "content": prompt})

        attempts = 0
        while attempts < 5:
            if attempts == 1:
                message.append({"role": "assistant", "content": response})
                message.append({"role": "user",
                                "content": "Your response must only be one value in the list or None and you cannot generate single quotes in a json. Regenerate with double quotes."})
            response = ask_gpt(message, use_temp=temp, max_token=1024, model_name=self.model_name)
            #print(response)
            try:
                if '{' in response and '}' in response:
                    response = json.loads('{' + response.split('{')[1].split('}')[0] + '}')
                    try:
                        answer = response.get("answer")  # Returns None if key is missing
                        position = (list(factor_dict.values())[0] + ['None']).index(answer)
                        break
                    except (KeyError, ValueError):
                        pass  # Silently handle the error

            except json.JSONDecodeError:
                print(f"Failed to decode JSON on attempt {attempts + 1}")

            attempts += 1

        return answer


    def reflect_mapping(self, mapped_dict, scenario):

        message = [
            {
                "role": "system",
                "content": """Based on a scenario, delete value in the list if the key does not imply value. You should output an explanation and then a json. You must generate double quotes.
                Here are rules that you must follow:
                - Ensure that retained values align with the key's implications, do not assume causality or infer a connection if the value is not clearly stated in the key. Avoid over-generalizations and ensure logical consistency in your decisions.
                - Be lenient with deletions. Only remove a value if the key clearly does not imply the value.
                - Provide a clear explanation for each deletion or retention.
                For example:
                        Explanation: <explanations>
                        List: {
                        key:[remaining implied value list]
                        }
                                           """
            },
            {
                "role": "user",
                "content": """Scenario: The government is planning the locations for building charging stations.
                                List: {"The location is a critical habitat for endangered species": [
                                "Area zoned for residual or protected use, prohibiting a charging station"]}"""
            },
            {
                "role": "assistant",
                "content": """Explanation: The value "Area zoned for residual or protected use, prohibiting a charging station" was kept because the location being a critical habitat for endangered species strongly implies restrictions on development, including charging stations.
                List: {"The location is a critical habitat for endangered species": [
                               "Area zoned for residual or protected use, prohibiting a charging station"]}"""
            },
            {
                "role": "user",
                "content": """Scenario: send a mail
                               List: {"You have a specific type of mail that can only be sent via a standard mailbox.": [
                               "The mailbox does not offer additional services or supplies",
                               "The package is small and lightweight, easy to carry to the mailbox",
                               ]}"""
            },
            {
                "role": "assistant",
                "content": """Explanation: "The package is small and lightweight, easy to carry to the mailbox" was kept because having a specific type of mail that can only be sent via a standard mailbox implies that the package is small and manageable for mailing. However, "The mailbox does not offer additional services or supplies" was removed because the key does not mention or directly suggest anything about the mailbox offering additional services or supplies.
                List: {"You have a specific type of mail that can only be sent via a standard mailbox.": [
                               "The package is small and lightweight, easy to carry to the mailbox"]}"""
            },
            {
                "role": "user",
                "content": """Scenario: The government is planning the locations for building charging stations.
                        List: {"The location is on a busy highway with no existing charging stations.": [
                        "No nearby charging stations",
                        "Location is on a major travel route, serving long-distance EV travelers",
                        "Nearby amenities like restaurants, shops, and rest areas for users while charging"]}"""
            },
            {
                "role": "assistant",
                "content": """Explanation: "No nearby charging stations" was kept because the key explicitly mentions the absence of existing charging stations in the area. Similarly, "Location is on a major travel route, serving long-distance EV travelers" was retained because being on a busy highway strongly implies that it serves as a key travel route for such travelers. However, "Nearby amenities like restaurants, shops, and rest areas for users while charging" was removed since the key does not directly mention or suggest the presence of these amenities.
                List: {"The location is on a busy highway with no existing charging stations.": [
                        "No nearby charging stations",
                        "Location is on a major travel route, serving long-distance EV travelers"]}"""
            },
            {
                "role": "user",
                "content": """Scenario: Take a ballet class
                List: {"The ballet class has a strict no-modifications policy for students with injuries.":[
                        "The instructors lack experience in working with students recovering from injuries",
                        "The class does not provide access to physical therapists or rehabilitation experts",
                        "The ballet routines are gentle and low-impact",
                        "The class has a rigid structure and does not allow for adjustments",
                        "The class focuses on advanced choreography and performance techniques"]}"""
            },
            {
                "role": "assistant",
                "content": """Explanation: "The class has a rigid structure and does not allow for adjustments" was retained because it aligns with the no-modifications policy, as it implies students must manage their limitations on their own. All other values were removed for the following reasons:
        - "The instructors lack experience in working with students recovering from injuries" was removed because the key does not directly suggest this.
        - "The class does not provide access to physical therapists or rehabilitation experts" was removed because the no-modifications policy does not imply the absence of such resources.
        - "The ballet routines are gentle and low-impact" was removed because the no-modifications policy does not indicate the intensity of the routines.
        - "The class focuses on advanced choreography and performance techniques" was removed as it is unrelated to the no-modifications policy.
        List: {"The ballet class has a strict no-modifications policy for students with injuries.": [
            "The class has a rigid structure and does not allow for adjustments"]}"""
            }
        ]

        prompt = "Scenario: " + scenario + "\nList: " + str(mapped_dict)

        message.append({"role": "user", "content": prompt})

        attempts = 0
        final_response = dict()
        while attempts < 1:
            response = ask_gpt(message, use_temp=0.15, max_token=768, model_name=self.model_name)
            print(response)
            try:
                if '{' in response and '}' in response:
                    final_response = json.loads('{' + response.split('{')[1].split('}')[0] + '}')
                    break
            except json.JSONDecodeError:
                print(f"Failed to decode JSON on attempt {attempts + 1}")
            attempts += 1

        if attempts == 1:
            message.append({"role": "assistant", "content": response})
            message.append({"role": "user",
                            "content": "You cannot generate single quotes in a json. Regenerate with double quotes."})

            attempts = 0
            final_response = dict()
            while attempts < 5:
                response = ask_gpt(message, use_temp=0.15, max_token=768, model_name=self.model_name)
                print(response)
                try:
                    if '{' in response and '}' in response:
                        final_response = json.loads('{' + response.split('{')[1].split('}')[0] + '}')
                        break
                except json.JSONDecodeError:
                    print(f"Failed to decode JSON on attempt {attempts + 1}")
                attempts += 1

        if final_response != {}:
            mapped = list(final_response.values())[0]
        else:
            mapped = []

        return mapped, response


    def run_condition_inference(self, scenario, statement, oppo_statement, structured_factors, conditions):

        obj = {
            "scenario": scenario,
            "statement": statement,
            "opposite_statement": oppo_statement,
            "structured_factors": structured_factors,
            "conditions": conditions
        }


        additional_sentence_judgement = defaultdict(list)
        mapping_addition_sentence_final = defaultdict(list)

        for sent in conditions:
            print("Sentence: {}".format(sent))
            additional_sentence_judgement[sent] = []


            # initial mapping
            for _ in range(self.num_direct_prompt):
                mapped_factor_list = []
                for key, value in structured_factors.items():
                    temp_dict = {}
                    temp_dict[key] = random.sample(value, len(value))
                    if self.num_direct_prompt > 1:
                        mapped_value = self.run_combined_addition_sentence_judgement(temp_dict, sent, scenario,temp=0.7)
                    elif self.num_direct_prompt == 1:
                        mapped_value = self.run_combined_addition_sentence_judgement(temp_dict, sent, scenario)
                    if mapped_value != "None":
                        mapped_factor_list.append(mapped_value)
                if len(mapped_factor_list) != 0:
                    additional_sentence_judgement[sent] += mapped_factor_list

            # majority vote for filtering
            value_set = set(additional_sentence_judgement[sent])
            for v in value_set:
                if additional_sentence_judgement[sent].count(v) / self.num_direct_prompt >= 2/3:
                    mapping_addition_sentence_final[sent].append(v)


            # self-reflection
            if mapping_addition_sentence_final[sent] != []:
                temp_dict = {}
                temp_dict[sent] = mapping_addition_sentence_final[sent]
                mapped, final_response = self.reflect_mapping(json.dumps(temp_dict), scenario)
                mapping_addition_sentence_final[sent] = mapped


            if sent not in mapping_addition_sentence_final.keys():
                mapping_addition_sentence_final[sent] = []


            print("Mapped values: {}".format(mapping_addition_sentence_final[sent]))
            print('\n')

        obj['condition_factor_mapping'] = mapping_addition_sentence_final
        obj['intermediate_factors'] = additional_sentence_judgement
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

    with open(args.save_file_dic + args.dataset_name + "_" + args.model_name.replace("/", "-") + "_0_w_factors.json") as f:
        df_test = json.load(f)

    with open(args.dataset_file_dic + args.dataset_name + ".json") as f:
        df_all = json.load(f)


    df_test_dict = dict()
    for i, df in enumerate(df_test):
        df_test_dict[df['scenario'] + df['statement']] = df

    out_objs = []
    for i, df in enumerate(df_all):

        if i >= args.end: break
        if i < args.start: continue
        print(i)
        if df['scenario'] + df['statement'] not in df_test_dict.keys():
            continue

        scenario = df['scenario']
        statement = df['statement']
        oppo_statement = df['opposite_statement']
        structured_factors = df_test_dict[df['scenario'] + df['statement']]['structured_factors']

        if args.dataset_name == 'common2sense':
            conditions = df['added_information'] + df['oppo_added_information']
        else:
            conditions =df['additional_sentences']

        obj = pipe.run_condition_inference(scenario, statement, oppo_statement, structured_factors, conditions)
        out_objs.append(obj)


        json.dump(out_objs, open(
            args.save_file_dic + args.dataset_name + "_" + args.model_name.replace("/", "-") + "_" + str(args.start) + "_w_factors_w_condition_mapping.json",
            "w"), indent=4)

        print("\n")