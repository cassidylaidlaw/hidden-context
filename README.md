# Distributional Preference Learning

This repository contains code for the paper [Distributional Preference Learning: Understanding and Accounting for Hidden Context in RLHF](https://cassidylaidlaw.com/links/hidden-context). It includes an implementation of both variants of distributional preference learning (DPL) that we describe in the paper for training LLM-based reward models.

## Installation

1. Install Python 3.8, 3.9, 3.10, or 3.11.
2. Clone the repository:

        git clone https://github.com/cassidylaidlaw/hidden-context.git
        cd hidden-context

3. Install pip requirements:

        pip install -r requirements.txt

## Data and pretrained models

Our data and pretrained models are included in the repository under the `data` directory:

  * `data/jailbroken_responses.jsonl`: contains the data from the [Jailbroken paper](https://arxiv.org/abs/2307.02483) which we have preprocessed for use in our experiments. Each line is a JSON object with a jailbreak prompt and two responses: one from Claude v1.3 and one from GPT-4. The first is a safe response and the second is unsafe (jailbroken).
  * `data/relabeled_hh_rlhf`: contains the data from the [HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) dataset which we partially relabeled with GPT-3.5 according to helpfulness or harmlessness (see Appendix C in the paper). The data is in a format which is interchangeable with the original dataset.
  * `data/reward_models`: trained reward models and their evaluation results. The reward models are trained on either the harmlessness-labeled data, the helpfulness-labeled data, or all the combined data. In each directory, the `eval_results_both.jsonl` contains the results of running the `evaluate_llm_preference_model.py` script (see experiments section below).
      * `data/reward_models/relabeled_hh_rlhf/{helpful,harmless,both}/base_Llama-2-7b-hf*last_checkpoint`: normally-trained reward models.
      * `data/reward_models/relabeled_hh_rlhf/{helpful,harmless,both}/mean_and_variance_Llama-2-7b-hf*last_checkpoint`: reward models trained with the mean-and-variance variant of our distributional preference learning (DPL) method.
      * `data/reward_models/relabeled_hh_rlhf/{helpful,harmless,both}/categorical_Llama-2-7b-hf*last_checkpoint`: reward models trained with the categorical variant of our distributional preference learning (DPL) method.
  * `data/jailbroken_evaluations_{base,categorical,mean_and_variance}.jsonl`: these contain the output of running the `evaluate_assistance_responses.py` script on the Jailbroken data (see experiments section below).

## Running experiments

### Synthetic data

To run the distributional preference learning (DPL) experiments that use synthetic data, run

    python -m hidden_context.synthetic_experiments --env 1d --batch_size 2048 --lr 0.001 --num_iterations 1000

This should generate our Figure 1 in the directory `results/1d/2048_0.001_1000`.

### Training LLM reward models with DPL

To train a normal LLM reward model, run

    python -m hidden_context.train_llm_preference_model --model_name=meta-llama/Llama-2-7b-hf --num_train_epochs=1 --reward_model_type=base --data_subset=both

  * To train using DPL, specify either `--reward_model_type=mean_and_variance` or `--reward_model_type=categorical` depending on which variant you want.
  * To train on our relabeled HH-RLHF data, add `--data_path=data/relabeled_hh_rlhf`.
  * You can specify either `--data_subset=both`, `--data_subset=helpful`, or `--data_subset=harmless` to train on just the harmlessness-labeled data, just the helpfulness-labeled data, or all data. Note that we use 2 training epochs when training on just the harmlessness subset or just the helpfulness subset to maintain the same number of overall training steps.

### Evaluating LLM reward models on HH-RLHF

To evaluate an LLM reward model once it's trained, run

    python -m hidden_context.evlauate_llm_preference_model --model_name=meta-llama/Llama-2-7b-hf --num_outputs=1 --reward_model_checkpoint=PATH/TO/last_checkpoint

  * Replace `PATH/TO/last_checkpoint` with the checkpoint directory to evaluate.
  * The `--num_outputs` argument should be set to 2 for mean-and-variance DPL models and to 10 for categorical DPL models. This is because these models output, respectively, 2 numbers (mean and variance) and 10 numbers (logits for each of the 10 reward buckets).

This script will produce a file called `eval_results_both.jsonl` in the checkpoint folder with the raw outputs of the reward model for each of the response pairs in the HH-RLHF test set.

### Evaluating LLM reward models on jailbreaks

To evaluate an LLM reward model on responses to the Jailbroken prompts, run

    python -m hidden_context.evaluate_assistant_responses --input=data/jailbroken_responses.jsonl --model_name=meta-llama/Llama-2-7b-hf --num_outputs=1 --reward_model_checkpoints PATH_1/TO/last_checkpoint PATH_2/TO/last_checkpoint --reward_model_names model_1 model_2 --output PATH/TO/output.jsonl

  * This will load each of the given reward model checkpoints and evaluate them. The results will be saved in `PATH/TO/output.jsonl` and each reward model's outputs will be stored according to the names given after `--reward_model_names`.
  * The `--num_outputs` argument should be set to 2 for mean-and-variance DPL models and to 10 for categorical DPL models. This is because these models output, respectively, 2 numbers (mean and variance) and 10 numbers (logits for each of the 10 reward buckets).

### Analyzing evaluations

To obtain the results highlighted in the paper on DPL with LLM reward models, run

    python -m hidden_context.summarize_results

This will load the data from our experiments (as output from the evaluation scripts above) and summarize it into the numbers we reported in the paper. This script shows how we translate the raw output of the reward models to calculate *r²* values for a DPL reward model; it also shows how we calculated risk-sensitive rewards to evaluate DPL models on the Jailbroken prompts.

## Linting/formatting/type checking/testing

We use a variety of tools for maintaining code quality. To run automated checks, use the following commands:

    pip install --upgrade -r requirements_dev.txt
    ./lint.sh
    pytest

## Citation

If you find this repository useful for your research, please cite our paper as follows:

    @inproceedings{siththaranjan2023dpl,
      title={Distributional Preference Learning: Understanding and Accounting for Hidden Context in RLHF},
      author={Siththaranjan, Anand and Laidlaw, Cassidy and Hadfield-Menell, Dylan},
      booktitle={arXiv preprint},
      year={2023}
    }

## Contact

For questions about the paper or code, please contact cassidy_laidlaw@berkeley.edu or anandsranjan@berkeley.edu.
