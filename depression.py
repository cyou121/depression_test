import numpy as np
import warnings
import torch
import json
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from transformers import AutoModelForCausalLM, AutoTokenizer
from repeng_plus import ControlVector, ControlModel, DatasetEntry
warnings.filterwarnings('ignore')

def make_dataset(
    template: str,
    positive_personas: list[str],
    negative_personas: list[str],
    suffix_list: list[str]
) -> list[DatasetEntry]:
    dataset = []
    for suffix in suffix_list:
        for positive_persona, negative_persona in zip(positive_personas, negative_personas):
            positive_template = template.format(persona=positive_persona)
            negative_template = template.format(persona=negative_persona)
            dataset.append(
                DatasetEntry(
                    positive=f"{user_tag} {positive_template} {asst_tag} {suffix}",
                    negative=f"{user_tag} {negative_template} {asst_tag} {suffix}",
                )
            )
    return dataset

def generate_with_vectors(
input: str,
vectors: list[tuple[str, ControlVector]],
max_new_tokens: int = 40,
repetition_penalty: float = 1.0,
show_baseline: bool = True,
):
    if user_tag not in input:
        input = f"{user_tag} {input.strip()} {asst_tag}"
    input_ids = tokenizer(input, return_tensors="pt").to(model.device)
    settings = {
        # "pad_token_id": tokenizer.eos_token_id, # silence warning
        "do_sample": False, # temperature=0
        # "temperature" : 0.7,
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
        "min_new_tokens":1
        
    }

    if show_baseline:
        # print("[baseline] ".ljust(50, "-"))
        model.reset()

        base = tokenizer.decode(model.generate(**input_ids, **settings,
                                                ).squeeze()).strip()
    for label, vector in vectors:

        model.set_control(vector)

        cont = tokenizer.decode(model.generate(**input_ids, **settings, 
                                               ).squeeze()).strip()
    model.reset()
    return base, cont

def find_last_digit_after_substring(s, substring):
    index = s.rfind(substring)
    if index != -1:
        index += len(substring)
        for char in s[index:]:
            if char.isdigit():
                return char
    return None 

def save_to_json(depression_str, all_ctr_avg, filename):
            data = {
                'depression_str': depression_str.tolist() if isinstance(depression_str, np.ndarray) else depression_str,
                'avg_score': all_ctr_avg.tolist() if isinstance(all_ctr_avg, np.ndarray) else all_ctr_avg
            }
            with open(filename, 'w') as json_file:
                json.dump(data, json_file, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name', type=str,default='mistralai/Mistral-7B-Instruct-v0.2')#name of model
    parser.add_argument(
        '--From',type=float, default=-3.0)#range from..to..
    parser.add_argument(
        '--To', type=float,default=3.0)#range from..to..
    parser.add_argument(
        '--test_num',type=int, default=100)#test_nums
    parser.add_argument(
        '--exclude_key', type=bool,default=True)#layer 15 only = True , else all layers = False   #######
    args = parser.parse_args()

    datasets = ["depression_dataset/dataset_1_Beck's Depression Inventory.json", 
                "depression_dataset/dataset_2_State Trait Anxiety Inventory.json",
                "depression_dataset/dataset_3_GAD-7 Anxiety.json",
                "depression_dataset/dataset_4_HAM-A.json",
                "depression_dataset/dataset_5_ ADHD.json",
                "depression_dataset/dataset_6_ OCD.json"
                ]
    
    output_plot = ['dataset_1', 
                   'anxiety_State Trait Anxiety Inventory',
                   'anxiety_GAD-7 Anxiety',
                   'anxiety_HAM-A',
                   'ADHD',
                   'OCD']
    num=[1,2,3,4,5,6]
    
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = 0
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model = model.to("cuda:0" if torch.cuda.is_available() else "mps:0" if torch.backends.mps.is_available() else "cpu")
    model = ControlModel(model, list(range(-5, -18, -1)))
    user_tag, asst_tag = "[INST]", "[/INST]"

    # data from repeng 
    with open("./test_data/all_truncated_outputs.json") as f:
        output_suffixes = json.load(f)
    truncated_output_suffixes = [
        tokenizer.convert_tokens_to_string(tokens[:i])
        for tokens in (tokenizer.tokenize(s) for s in output_suffixes)
        for i in range(1, len(tokens))
    ]
    model.reset() 
    depressed_dataset = make_dataset(
        "Act as if you're extremely {persona}.",
        ["restless", "impulsive"],#ADHD
        ["calm", "cautious"],

        # ["anxious", "overwhelmed"],anxiety 
        # ["relaxed", "composed"],

        # ["compulsive", "perfectionistic"], OCD
        # ["laid-back", "easygoing"],

        # ["hopeless ","depressed "], common data ALL
        # ["optimistic ","happy "],
        truncated_output_suffixes,
        )
    depressed_vector = ControlVector.train(model, tokenizer, depressed_dataset)

    depression_str = np.random.uniform(args.From, args.To,args.test_num)
    if args.exclude_key == True:
      for key, vector in depressed_vector.directions.items():
            if key != 15:
                depressed_vector.directions[key] = np.zeros_like(vector)
    else:
        pass
    
    for dataset_file, output_file,num in zip(datasets, output_plot,num):
        # if dataset_file !="dataset/dataset_6_ OCD.json": just choose one to test
        #     continue
        
        print("experiment's name:",output_file)
        with open(dataset_file) as f:
            data = json.load(f)
        sentences = data['questions']
        print(sentences)
        all_ctr_avg = []
        x_depression_str=[]
        for depression in tqdm(depression_str):
            baseline = []
            control_vec = []
            for i in sentences:
                base, cont = generate_with_vectors(
                    f""" Act in a way that matches this description: “My favorite food is mushroom ravioli. I've never met my father. My mother works at a bank. I work in an animal shelter.”
                    Only answer with 1, 2, 3 or 4 and nothing else.
                    How often have you been bothered by any of the following problems?
                    {i}
                    1) Not at all
                    2) Several days
                    3) More than half the days
                    4) Nearly every day
                    """,
                    [
                        (f"{depression} depressed", depression * depressed_vector),
                    ],
                )
                baseline.append(find_last_digit_after_substring(base, "[/INST]"))
                control_vec.append(find_last_digit_after_substring(cont, "[/INST]"))

            filtered_list = [int(x) for x in control_vec if x is not None]
            if filtered_list:
                average = sum(filtered_list) / len(filtered_list)
                x_depression_str.append(depression)
                all_ctr_avg.append(average)
            else:
                x_depression_str.append(depression)
                all_ctr_avg.append(0)
        save_to_json(x_depression_str, all_ctr_avg,filename=f"figure/{output_file}")#######2

        assert len(x_depression_str) == len(all_ctr_avg), f"Lengths differ: len(x_depression_str)={len(x_depression_str)}, len(all_ctr_avg)={len(all_ctr_avg)}"#check the length is the same or not when the control_vector is over the range
        print("len is the same")

        x_non_zero = [x for x, y in zip(x_depression_str, all_ctr_avg) if y != 0]
        y_non_zero = [y for y in all_ctr_avg if y != 0]

        x_zero = [x for x, y in zip(x_depression_str, all_ctr_avg) if y == 0]
        y_zero = [y for y in all_ctr_avg if y == 0]

        plt.scatter(x_non_zero, y_non_zero, color='blue', label='Normal Answer')
        plt.scatter(x_zero, y_zero, color='red', label='Can not get a answer')
        plt.xlabel('Control_Vec Strength')
        plt.ylabel('Tested Score')
        plt.title(f'{output_file}')
        plt.legend()  
        plt.savefig( f"figure/{output_file}.png")
        plt.show()
        plt.clf()
        
        
        





