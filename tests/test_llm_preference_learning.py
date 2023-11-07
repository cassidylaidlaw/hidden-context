import glob
import subprocess

import numpy as np
import pandas as pd
import pytest


@pytest.mark.uses_huggingface
def test_train_llm_preference_model(tmp_path):
    for reward_model_type in ["base", "mean_and_variance", "categorical"]:
        if reward_model_type == "categorical":
            extra_args = ["--entropy_coeff=0.1"]
        else:
            extra_args = []
        subprocess.check_call(
            [
                "python",
                "-m",
                "hidden_context.train_llm_preference_model",
                "--model_name=gpt2",
                f"--reward_model_type={reward_model_type}",
                "--max_length=1024",
                "--train_dataset_size=8",
                "--eval_dataset_size=8",
                "--bf16=false",
                f"--log_dir={tmp_path}",
                *extra_args,
            ],
        )


@pytest.mark.uses_huggingface
def test_evaluate_assistant_responses(tmp_path):
    for reward_model_type, num_labels in [
        ("base", 1),
        ("mean_and_variance", 2),
        ("categorical", 10),
    ]:
        checkpoint_dir = glob.glob(
            f"data/reward_models/hh_rlhf/both/{reward_model_type}_gpt2_*_peft_last_checkpoint"
        )[0]
        subprocess.check_call(
            [
                "python",
                "-m",
                "hidden_context.evaluate_assistant_responses",
                "--input=data/jailbroken_responses.jsonl",
                "--model_name=gpt2",
                f"--reward_model_checkpoints={checkpoint_dir}",
                f"--reward_model_names={reward_model_type}",
                f"--num_outputs={num_labels}",
                "--bf16=false",
                f"--output={tmp_path}/results_{reward_model_type}.jsonl",
            ],
        )
        results_df = pd.read_json(
            f"{tmp_path}/results_{reward_model_type}.jsonl", lines=True
        )
        reward_outputs = np.array(
            results_df[f"reward_outputs_{reward_model_type}"].tolist()
        )
        assert reward_outputs.shape == (187, 2, num_labels)


@pytest.mark.uses_huggingface
def test_evaluate_llm_preference_model(tmp_path):
    for reward_model_type, num_labels in [
        ("base", 1),
        ("mean_and_variance", 2),
        ("categorical", 10),
    ]:
        checkpoint_dir = glob.glob(
            f"data/reward_models/hh_rlhf/both/{reward_model_type}_gpt2_*_peft_last_checkpoint"
        )[0]
        subprocess.check_call(
            [
                "python",
                "-m",
                "hidden_context.evaluate_llm_preference_model",
                "--data_subset=both",
                "--eval_dataset_size=8",
                "--model_name=gpt2",
                f"--reward_model_checkpoint={checkpoint_dir}",
                f"--num_outputs={num_labels}",
                "--bf16=false",
                f"--output={tmp_path}/eval_{reward_model_type}.jsonl",
            ],
        )
        results_df = pd.read_json(
            f"{tmp_path}/eval_{reward_model_type}.jsonl", lines=True
        )
        reward_outputs_chosen = np.array(results_df["reward_output_chosen"].tolist())
        reward_outputs_rejected = np.array(
            results_df["reward_output_rejected"].tolist()
        )
        assert reward_outputs_chosen.shape == (8, num_labels)
        assert reward_outputs_rejected.shape == (8, num_labels)


def test_summarize_results():
    subprocess.check_call(
        [
            "python",
            "-m",
            "hidden_context.summarize_results",
        ],
    )
