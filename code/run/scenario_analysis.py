import time
import json
from collections import defaultdict
import argparse
from utils.utils import ask_gpt



class Pipeline:

    def __init__(self, args):

        self.data_type = args.dataset_name
        self.model_name = args.model_name

        # Abductive factor generation parameters
        self.num_added_sentences = 2
        self.num_combined_factor_structure = 1

        # Additional fator generation processing parametes
        self.generate_detailed_factors = args.generate_detailed_factors
        self.extend_factors = args.extend_factors

        # Factor-outcome mapping parameters
        self.factor_outcome_judgement = True
        self.num_factor_outcome_mapping = 3


    def generate_additional_sentence_hypothesis(self, scenario, statement, oppo_statement, add_sample_list,
                                                oppo_add_sample_list, add_sample_exp_list, oppo_add_sample_exp_list,
                                                num_sentence=1):
        message = [
            {
                "role": "system",
                "content": """You are given a scenario and an accompanying hypothesis. Generate 5 sentences covering different conditions that would add objective information relevant to the hypothesis such that the hypothesis is more likely to hold true. The information should not definitively imply the hypothesis. You must follow the below structure to just generate sentences with no explanations.
                For example:
                            # <generated sentence 1>
                            # <generated sentence 2>.
                            # <generated sentence 3>.
                            # <generated sentence 4>.
                            # <generated sentence 5>."""
            },
            {
                "role": "user",
                "content": "Scenario: You want to move around with your cell phone when it is being charged.\nHypothesis:  You can move around more freely with your cell phone if it is being charged with a one-foot cord rather than a six-foot cord."
            },
            {
                "role": "assistant",
                "content": """# The cell phone is being charged with a portable power bank located in your pocket, allowing you to move around without being tethered to a fixed outlet.
                # The user is working in a compact space where longer cords could easily snag on furniture or equipment, thus a one-foot cord could minimize this risk.
                # The phone is needed for tasks that require frequent handling and close proximity to the user, making a shorter cord more practical to avoid excessive dangling.
                # The charging setup includes a small desktop charger that keeps the phone elevated and stable, limiting the practicality of a longer cord.
                # The user is in a busy environment like a kitchen or workshop, where shorter cords can reduce the hazard of tripping or catching on moving objects."""},
        ]
        message.append({"role": "user", "content": "Scenario: " + scenario + "\nHypothesis: " + statement})


        for _ in range(num_sentence):
            answer_final = ''
            attempts = 0
            while attempts < 5:
                answer = ask_gpt(message, max_token=256, model_name=self.model_name)
                try:
                    answer = answer.replace("\n\n", "\n")
                    for ans in answer.split("\n"):
                        if '# ' in ans:
                            answer_final = ans.split('# ')[1]
                    break
                except (IndexError, json.JSONDecodeError) as e:
                    print(f"Failed to generate an extracted answer on attempt {attempts + 1}: {e}")

                attempts += 1

            for ans in answer.split("\n"):
                if '# ' in ans:
                    answer_final = ans.split('# ')[1]
                    add_sample_list.append(answer_final)
            add_sample_exp_list.append(answer)

        message.pop()
        message.append({"role": "user", "content": "Scenario: " + scenario + "\nHypothesis: " + oppo_statement})


        for _ in range(num_sentence):
            answer_final = ''
            attempts = 0
            while attempts < 5:
                answer = ask_gpt(message, max_token=256, model_name=self.model_name)
                try:
                    answer = answer.replace("\n\n", "\n")
                    for ans in answer.split("\n"):
                        answer_final = ans.split('# ')[1]
                    break
                except (IndexError, json.JSONDecodeError) as e:
                    print(f"Failed to generate an extracted answer on attempt {attempts + 1}: {e}")

                attempts += 1
            for ans in answer.split("\n"):
                answer_final = ans.split('# ')[1]
                oppo_add_sample_list.append(answer_final)
            oppo_add_sample_exp_list.append(answer)


        return add_sample_list, add_sample_exp_list, oppo_add_sample_list, oppo_add_sample_exp_list

    def run_structured_factors_generation(self, added_information, oppo_added_information, sceneraio, statement,
                                          oppo_statement, temp=0.15, token=768):

        message = [
            {
                "role": "system",
                "content": """From the given sentences for each statement, identify and list distinct and concrete factors, ensuring each is broad yet specific and focuses on a unique aspect.
                            Your response should strictly adhere to the JSON format provided, without additional explanations.
                            For example:
                                    "distinct factor" <1. ENSURE each factor focuses on a unique aspect>: "factor values" <Each factor MUST cover at least one condition to support the Statement and one condition to support the Opposite statement.
                                    2. Ensure that each factor's value MUST directly reference specific elements mentioned in the statements, avoiding vague terms like 'the object'.
                                    3. Ensure the factor values are not too concrete, e.g., instead of saying a new grocery store with healthy food options has openned, say healthy food options are available when eating out.
                                    4. Do not only mention the common situations, e.g, a bottle could also be heavier than a bowling ball.>"""
            },
            {
                "role": "user",
                "content": "Scenario: You are charging your cell phone and wish to move around with your cell phone. \nStatement: You can move around more freely with your cell phone if it is being charged with a six feet cord rather than a one feet cord.\nSentence:\n#1 A longer cord provides more flexibility and allows for a greater range of movement while using the cell phone. This is because the additional length of the six feet cord gives the user a larger radius of movement, enabling them to comfortably use their phone while it is charging without feeling restricted or confined to a specific location.\nOppo_statement: You can move around more freely with your cell phone if it is being charged with a one feet cord rather than a six feet cord.\nSentence:\n#2 If the cell phone is plugged into a portable power bank or a USB port on a computer, a one feet cord provides greater mobility because it is shorter and less likely to get tangled or caught on objects while moving.\n#3 If the cell phone is constantly being used while charging and the user prefers to keep the phone close to the charger at all times, a one foot cord allows for easier mobility and reduces the risk of tripping over a longer cord."
            },
            {
                "role": "assistant",
                "content": """{
                                    "The cell phone's charging method": ["The charger is portable", "The charger is unmovable"],
                                    "The user's movement range": ["The user stays very close to the charger", "The user has large radius of movement"],
                                    "The location of the phone charger": ["The user leaves the charger somewhere", "The user carries the charger"]}"""
            },
            {
                "role": "user",
                "content": "#Scenario: The government is planning where to build charging stations.\nStatement: The government should build a charging station here.\nSentence:\n#1 With a high adoption rate of electric vehicles in the area and no nearby charging stations, this location would fill a critical gap in the network while meeting the growing local demand.\n#2 The site is on a major highway and the presence of nearby restaurants and shops would make it convenient for EV users to recharge while taking a break.\n#3 This location is equipped with the necessary infrastructure to support a charging station and is assigned for commercial use, making it an ideal spot for development.\nOppo_statement: The government should not build a charging station here.\nSentence:\n#4 The low ownership of electric vehicles in this area, combined with several nearby charging stations, makes it an unnecessary and redundant investment.\n#5 This site lacks the infrastructure needed to support a charging station.\n#6 With insufficient resources at the location and a lack of public support due to the area's low EV adoption rate, building a charging station here would likely be inefficient.\n#7 The presence of multiple charging stations within a short distance reduces the need for a new facility here, especially given the low demand from local residents."
            },
            {
                "role": "assistant",
                "content": """{"Local demand and support": ["High adoption rate of electric vehicles (EVs) in the area","Low adoption rate of electric vehicles (EVs) among residents in the area"],
                               "Availability of resources and infrastructure": ["Sufficient resources and infrastructure to support a charging station","Insufficient resources and infrastructure to support a charging station"],
                               "Proximity to Existing Charging Stations": ["No nearby charging stations","Several charging stations nearby"],
                               "Land use and zoning regulations": ["Areas designated for commercial or industrial use where charging stations are permitted","Areas designated for residential or protected use where charging stations are prohibited"],
                               "Route to Popular Long-Haul Destinations": ["Location is on a major travel route, serving long-distance EV travelers.","Off main travel routes, limiting usage to local residents."],
                               "Availability of Amenities": ["Presence of nearby amenities like restaurants, shops, or rest areas for users while charging.", "Lack of nearby amenities like restaurants, shops, or rest areas for users while charging."]}"""
            }
        ]

        if self.generate_detailed_factors:
            message += [
                {
                    "role": "user",
                    "content": "#Scenario: The difficulty of lifting a metal toolbox compared to a cardboard box is being evaluated.\nStatement: A metal toolbox is more difficult to lift than a cardboard box is.\nSentence:\n#1 The metal toolbox is made of steel and filled with tools, giving it a dense and heavy structure that increases the effort required to lift it.\n#2 The cardboard box is empty and made of lightweight corrugated material, making it easy to pick up with minimal effort.\n#3 The person attempting to lift the objects has no bodily impairments and is using both hands for lifting.\nOppo_statement: A metal toolbox is less difficult to lift than a cardboard box is.\nSentence:\n#4 The person trying to lift the objects is a professional weightlifter.\n#5 The cardboard box is exceptionally large or awkwardly shaped, making it difficult to grasp or balance.\n#6 The person has a health condition such as arthritis or grip strength issues that make it difficult to hold onto lightweight objects.\n#7 The metal toolbox has a sturdy, ergonomic handle designed for easy lifting.\n#8 The person is accustomed to lifting heavy objects and has developed techniques that make lifting the toolbox easier.\n#9 The cardboard box is wet or damaged, making it fragile and challenging to lift without it falling apart."
                },
                {
                    "role": "assistant",
                    "content": """{"The metal toolbox's material": ["The metal toolbox is made of dense metal", "The metal toolbox is made of lightweight metal"],
                            "The metal toolbox's inside filling": ["The metal toolbox is filled with tools","The metal toolbox is empty"],
                            "The metal toolbox's grip and accessibility": ["The metal toolbox has a sturdy, ergonomic handle designed for easy lifting","The metal toolbox has a slippery surface or lacks a handle"],
                            "The cardboard box's material": ["The cardboard box is made of thin, corrugated material","The cardboard box is reinforced with thick layers, making it heavier than usual"],
                            "The cardboard box's inside filling": ["The cardboard box is empty","The cardboard box is filled with heavy items"],
                            "The cardboard box's grip and accessibility": ["The cardboard box has well-placed cutouts or handles","The cardboard box is large, fragile, damaged or lacks a proper grip area"],
                            "The person's physical ability and condition": ["The person has no bodily impairments and standard grip strength","The person has a condition such as arthritis or weak grip strength, affecting their ability to lift"],
                            "The person's experience with lifting objects": ["The person has average lifting ability and no specialized technique","The person is a professional weightlifter or accustomed to lifting heavy objects"]}"""
                }]

        prompt = '\nScenario: ' + sceneraio
        prompt += '\nStatement: ' + statement + '\nSentence:\n'
        count = 1
        for data in added_information:
            prompt += "#" + str(count) + " " + data + '\n'
            count += 1

        prompt += 'Oppo_statement: ' + oppo_statement + '\nSentence:\n'
        for data in oppo_added_information:
            prompt += "#" + str(count) + " " + data + '\n'
            count += 1

        message.append({"role": "user", "content": prompt})

        attempts = 0
        final_response = dict()
        while attempts < 5:
            response = ask_gpt(message, use_temp=temp, max_token=token, model_name=self.model_name)
            print(response)
            try:
                if '{' in response and '}' in response:
                    final_response = json.loads('{' + response.split('{')[1].split('}')[0] + '}')
                    break
            except json.JSONDecodeError:
                print(f"Failed to decode JSON on attempt {attempts + 1}")
            attempts += 1

        if self.extend_factors:
            attempts = 0
            message.append({"role": "assistant", "content": response})
            message.append({"role": "user",
                            "content": "Do you think that's all the factors you can conclude? Have you think about characteristics/features/advatanges/disadvantages of all the entities/conditions mentioned in the sceanrio, e.g., one condition in the scenario is eating out and have you considered the properties of restaruants. If you think there are additional factors, regenrate. Otherwise, keep your answer."})

            while attempts < 5:
                response = ask_gpt(message, use_temp=temp, max_token=token, model_name=self.model_name)
                print(response)
                try:
                    if '{' in response and '}' in response:
                        final_response = json.loads('{' + response.split('{')[1].split('}')[0] + '}')
                        break
                except json.JSONDecodeError:
                    print(f"Failed to decode JSON on attempt {attempts + 1}")
                attempts += 1

            del message[-2]
            del message[-1]

            attempts = 0
            message.append({"role": "assistant", "content": response})
            message.append({"role": "user",
                            "content": "You should then rerank the factors based on the IMPORTANCE for the decision and you can keep at most 7 factors and its corresponding values. The factors should not talk exactly about the differences in the two statements. You should also make sure the factors are DIVERSIFIED about different entities/conditions mentioned in the sceanrio. Ensure each factor focuses on a unique aspect and avoid creating factors that are similar to one another. Ensure that each factor's value MUST directly reference specific entities/conditions mentioned in the statements, avoiding vague terms like 'the object'"})
            while attempts < 5:
                response = ask_gpt(message, use_temp=temp, max_token=token, model_name=self.model_name)
                print(response)
                try:
                    if '{' in response and '}' in response:
                        final_response = json.loads('{' + response.split('{')[1].split('}')[0] + '}')
                        break
                except json.JSONDecodeError:
                    print(f"Failed to decode JSON on attempt {attempts + 1}")
                attempts += 1

            del message[-2]
            del message[-1]


        attempts = 0
        message.append({"role": "assistant", "content": response})
        if self.data_type == "common2sense":
            message.append({"role": "user",
                            "content": "Do you think that's all the possible values for each factor? If one possible situation is mentioned (e.g., 'The bottle is lighter than the bowling ball'), ensure the opposite perspective is also mentioend as a value('The bottle is heavier than the bowling ball'). Each value must explicitly reference the specific elements in the statements, avoiding vague terms like 'the object' that could apply to either statement. Regenerate all the values for a factor if necessary opposite values are missing or the values are vague. Do not create values that imply extremes, such as 'moderate' or 'extreme'."})
        else:
            message.append({"role": "user",
                            "content": "Do you think that's the complete possible values for each factor, e.g, if you have mentioned that a bottle is lighter than a bowling ball, you should also mention a bottle is heavier than a bowling ball.\nEnsure that each factor's value MUST directly reference specific elements mentioned in the statements, avoiding vague terms like 'the object' that could mean either of the choices in the statement.\nDo not generate new values for a factor unless you think there are additional values you must cover and do not generate a value that talks about moderate or extremely. Do not generate values that are partial subsets or duplicates of those already listed."})

        while attempts < 5:
            response = ask_gpt(message, use_temp=temp, max_token=1024, model_name=self.model_name)
            print(response)
            try:
                if '{' in response and '}' in response:
                    final_response = json.loads('{' + response.split('{')[1].split('}')[0] + '}')
                    break
            except json.JSONDecodeError:
                print(f"Failed to decode JSON on attempt {attempts + 1}")
            attempts += 1

        return final_response



    def key_factor_outcome_mapping(self, scenario, statement, oppo_statement, key, value, temp=0.15):

        message = [
            {
                "role": "system",
                "content": """Decide whether each of the given values supports Statement 1 or Statement 2. The statement and opposite statement are provided, along with a list of values. Determine which statement each value supports. You must follow the below structure to just generate json with an explanation.
                                For example:
                                Explanation: <explanations for how each value supports which statement>
                                List: {
                                "value 1":<output only Statement 1, Statement 2, Both, or Neither>,
                                "value 2":<output only Statement 1, Statement 2, Both, or Neither>}"""
            },
            {
                "role": "user",
                "content": """Here is a scenario: take a series of ballet classes. I have a list of values related to the nature of an injury and two statements. I'd like you to determine which statement each value supports.
                        Values:
                        * The injury is minor and temporary
                        * The injury is severe and permanent
                        * The injury is chronic but manageable
                        * The injury is unpredictable and fluctuating
                        Statements:
                        * Statement 1: You would complete the classes without making any adjustments for your injury.
                        * Statement 2: You would adjust the class schedule to accommodate your injury and complete it.
                        Please analyze each value and determine which statement it supports."""
            },
            {
                "role": "assistant",
                "content": """Explanation: In the context of attending ballet classes, a minor and temporary injury is unlikely to require adjustments, supporting the ability to complete classes as is (Statement 1). Conversely, a severe and permanent injury would necessitate modifications to accommodate the condition, aligning with Statement 2. A chronic but manageable injury could support both approaches, as the individual might complete the classes without changes (Statement 1) or make occasional adjustments (Statement 2). However, an unpredictable and fluctuating injury would introduce variability, making adjustments essential to participation, thus supporting Statement 2.
                List: {
                        "The injury is minor and temporary": "Statement 1",
                        "The injury is severe and permanent": "Statement 2",
                        "The injury is chronic but manageable": "Both",
                        "The injury is unpredictable and fluctuating": "Statement 2"
                        }"""
            }
        ]

        if scenario and not scenario[0].islower():
            scenario_temp = scenario.lower()
        else:
            scenario_temp = scenario
        if scenario_temp.endswith('.'):
            scenario_temp = scenario_temp[:-1]

        if key and not key[0].islower():
            key_temp = key.lower()
        else:
            key_temp = key
        if key_temp.endswith('.'):
            key_temp = key_temp[:-1]

        prompt = "Here is a scenario: " + scenario_temp + ". I have a list of values related to " + key_temp + " and two statements. I'd like you to determine which statement each value supports.\nValues:\n"
        for v in value:
            prompt += "* " + v + "\n"
        prompt += "Statements:\n* Statement 1: " + statement + "\n* Statement 2: " + oppo_statement + '\nPlease analyze each value and determine which statement it supports.'
        message.append({"role": "user", "content": prompt})

        attempts = 0
        final_response = dict()
        while attempts < 5:
            if attempts == 1:
                message.append({"role": "assistant", "content": response})
                message.append({"role": "user",
                                "content": "Your response cannot be loaded as json. Regenerate, remember to use double quotation marks."})
            response = ask_gpt(message, use_temp=temp, max_token=768, model_name=self.model_name)
            try:
                if '{' in response and '}' in response:
                    final_response = json.loads('{' + response.split('{')[1].split('}')[0] + '}')
                    break
            except json.JSONDecodeError:
                print(f"Failed to decode JSON on attempt {attempts + 1}")
            attempts += 1

        return final_response, response

    def prune_factors(self, structured_factors, factor_implication_dict):
        factor_implication_structure_dict = defaultdict(list)
        for key, value in structured_factors.items():
            temp = []
            for v in value:
                if v in factor_implication_dict.keys():
                    temp.append(factor_implication_dict[v])
            if len(set(temp)) == 1:
                for v in value:
                    if v in factor_implication_dict.keys():
                        del factor_implication_dict[v]
            else:
                for v in value:
                    if v != '' and v in factor_implication_dict.keys():
                        factor_implication_structure_dict[key].append(v)

        return factor_implication_structure_dict, factor_implication_dict

    def run_scenario_analysis(self, scenario, statement, oppo_statement):

        obj = {
            "scenario": scenario,
            "statement": statement,
            "opposite_statement": oppo_statement,
        }

        obj['generated_added_information'] = []
        obj['oppo_generated_added_information'] = []
        obj['generated_added_information_explanation'] = []
        obj['oppo_generated_added_information_explanation'] = []

        obj["structured_factors_list"] = []

        for _ in range(0, self.num_combined_factor_structure):
            # additional sentence generation
            add_sample_list = []
            oppo_add_sample_list = []

            add_sample_exp_list = []
            oppo_add_sample_exp_list = []

            # generate additional sentences for each direction
            for _ in range(0, self.num_added_sentences):
                add_sample_list, add_sample_exp_list, oppo_add_sample_list, oppo_add_sample_exp_list = self.generate_additional_sentence_hypothesis(
                    scenario, statement, oppo_statement, add_sample_list, oppo_add_sample_list, add_sample_exp_list,
                    oppo_add_sample_exp_list)
                print("Generated {} additional sentences for each direction".format(str(len(add_sample_list))))

            obj['generated_added_information'] += add_sample_list
            obj['oppo_generated_added_information'] += oppo_add_sample_list
            obj['generated_added_information_explanation'] += add_sample_exp_list
            obj['oppo_generated_added_information_explanation'] += oppo_add_sample_exp_list

            # summarize factors
            structured_factors = self.run_structured_factors_generation(add_sample_list,
                                                                        oppo_add_sample_list, scenario,
                                                                        statement, oppo_statement, temp=0.15)
            obj["structured_factors_list"].append(structured_factors)


        if self.num_combined_factor_structure > 1:
            # NOT IMPLEMENTED IN THE CODE: you should implement a factor_summerization function if you need to combine several runs to summerize factors.
            obj["structured_factors"] = self.factor_summerization(scenario, statement, oppo_statement,
                                                           obj["structured_factors_list"])
        else:
            obj["structured_factors"] = obj["structured_factors_list"][0]

        print("Final structured factors:\n{}".format(obj["structured_factors"]))

        if self.factor_outcome_judgement:
            factor_outcome_mapping_dict = defaultdict(list)
            factor_outcome_mapping_list = []
            obj["factor_outcome_mapping"] = dict()

            for _ in range(0, self.num_factor_outcome_mapping):
                for key, value in obj["structured_factors"].items():
                    if self.num_factor_outcome_mapping > 1:
                        temp_dict, response = self.key_factor_outcome_mapping(scenario, statement, oppo_statement, key,
                                                                          value,0.7)
                    elif self.num_factor_outcome_mapping == 1:
                        temp_dict, response = self.key_factor_outcome_mapping(scenario, statement, oppo_statement, key,
                                                                              value)
                    for key, value in temp_dict.items():
                        if value == 'Both':
                            factor_outcome_mapping_dict[key].append('Neutral')
                        elif value == 'Statement 1' or value == 'Statement 2':
                            factor_outcome_mapping_dict[key].append(value)
                        elif value == 'Neither':
                            factor_outcome_mapping_dict[key].append('Neither')
                    factor_outcome_mapping_list.append(response)

            structure_list_flat = []
            for key, value in obj["structured_factors"].items():
                for v in value:
                    structure_list_flat.append(v)

            for sent, value in factor_outcome_mapping_dict.items():
                value_set = set(factor_outcome_mapping_dict[sent])
                if sent in structure_list_flat:
                    visited = 0
                    if "Neither" not in value_set:
                        for v in value_set:
                            if (factor_outcome_mapping_dict[sent].count(v) / len(
                                    factor_outcome_mapping_dict[sent])) >= 2 / 3:
                                obj["factor_outcome_mapping"][sent] = v
                                visited = 1
                                break

                    if visited == 0:
                        for key, values in obj['structured_factors'].items():
                            if sent in values:
                                obj['structured_factors'][key].remove(sent)

            print("Finished Condition Factor Mapping.")

        # prune factors
        obj["old_structured_factor_and_mapping"] = [obj['structured_factors'], obj["factor_outcome_mapping"]]
        obj["structured_factors"], obj["factor_outcome_mapping"] = self.prune_factors(obj['structured_factors'],
                                                                                      obj["factor_outcome_mapping"])


        return obj

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-70B-Instruct", help="select the model name")
    parser.add_argument("--dataset_name", type=str, default="plasma", help="dataset name")
    parser.add_argument("--dataset_file_dic", type=str, default="/BIRD/data/", help="data file dictionary")
    parser.add_argument("--save_file_dic", type=str, default="/BIRD/results/",help="save file dictionary")
    parser.add_argument("--start", type=int, default=0, help="start instance")
    parser.add_argument("--end", type=int, default=10000, help="end instance")
    parser.add_argument("--extend_factors", type=bool, default=True, help="extend and extract the most important factors")
    parser.add_argument("--generate_detailed_factors", type=bool, default=False, help="generate more detailed factors")

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
        print(df['scenario'])
        scenario = df['scenario']
        statement = df['statement']
        oppo_statement = df['opposite_statement']

        obj = pipe.run_scenario_analysis(scenario, statement, oppo_statement)
        out_objs.append(obj)

        json.dump(out_objs, open(args.save_file_dic + args.dataset_name + "_" + args.model_name.replace("/", "-") + "_" + str(args.start) + "_w_factors.json",
            "w"), indent=4)

        print('\n')


