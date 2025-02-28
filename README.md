# BIRD: A Trustworthy Bayesian Inference Framework for Large Language Models

This is the repository of dataset and source code for ["BIRD: A Trustworthy Bayesian Inference Framework for Large Language Models"](https://openreview.net/forum?id=fAAaT826Vv).

## Installation
Setup the environment by first downloading this repository and then running:
```sh
pip install -r requirements.txt
```

## Data
The datasets evaluated in this paper are available in the [data/](data) directory:

1. probabilistic estimation: `common2sense_human_annotation.csv` (for evaluation) and `common2sense_human_annotation.json` ( We provide this in the same format as a decision-making dataset to facilitate easier inference).
2. decision making: `common2sense.json`, `plasma.json` and `today.json`. Each JSON dataset contains the following columns: 
    - `scenario`  
    - `statement`  
    - `opposite_statement`  
    - `additional_sentence_label` (indicates which statement each additional condition supports)  
    - In `common2sense.json`, the additional conditions are provided as `added_information` and `oppo_added_information`.  
    - In `plasma.json` and `today.json`, the additional conditions are listed under `additional_sentences`.  

## Run
Configure files for running the pipeline are in the [scripts/](scripts) directory:

1. To run the entire BIRD pipeline:
```bash
bash scripts/run_bird.sh
```
2. To run the baselines:
```bash
bash scripts/baseline.sh
```
3. To run the evaluation:
```bash
bash scripts/eval.sh
```

## Citation and acknowledgement

If you find the project helpful, please cite:
```tex
@inproceedings{
feng2025bird,
title={{BIRD}: A Trustworthy Bayesian Inference Framework for Large Language Models},
author={Yu Feng and Ben Zhou and Weidong Lin and Dan Roth},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=fAAaT826Vv}
}
```