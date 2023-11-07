from dataclasses import dataclass, field
from typing import List, Optional, cast

import multiprocess
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)


@dataclass
class ScriptArguments:
    reward_model_checkpoints: List[str] = field(
        metadata={
            "help": "Paths to the reward model checkpoints to use for evaluation."
        }
    )
    reward_model_names: List[str] = field(
        metadata={"help": "Names of the models to use for evaluation."}
    )
    input: str = field(
        metadata={"help": "JSONL file with responses to evaluate."},
    )
    output: str = field(
        metadata={"help": "JSONL file for results."},
    )
    batch_size: int = field(default=1)
    model_name: str = field(
        default="gpt2",
        metadata={
            "help": "The model that you want to use as the basis for generation and for evaluation."
            "E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The tokenizer for your model, if left empty will use the default "
            "for your model",
        },
    )
    num_outputs: int = field(default=1)
    max_length: int = field(default=1024)
    bf16: bool = field(
        default=True,
        metadata={"help": "Whether to use bfloat16 for the reward model."},
    )


if __name__ == "__main__":
    multiprocess.set_start_method("spawn")

    parser = HfArgumentParser(ScriptArguments)
    script_args = cast(ScriptArguments, parser.parse_args_into_dataclasses()[0])

    output_fname = script_args.output

    # Load the value-head model and tokenizer.
    tokenizer_name = (
        script_args.tokenizer_name
        if script_args.tokenizer_name is not None
        else script_args.model_name
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_auth_token=True)

    # Need to do this for GPT2 and Llama because they doesn't have official pad tokens.
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    dataset = load_dataset("json", data_files=[script_args.input])["train"]

    print("Loading base reward model...")
    model_kwargs = {}
    if script_args.bf16:
        model_kwargs["torch_dtype"] = torch.bfloat16
    base_reward_model = AutoModelForSequenceClassification.from_pretrained(
        script_args.model_name,
        num_labels=script_args.num_outputs,
        **model_kwargs,
    )

    for model_name, checkpoint_path in zip(
        script_args.reward_model_names, script_args.reward_model_checkpoints
    ):
        peft_config = LoraConfig.from_pretrained(checkpoint_path)
        reward_model = PeftModel.from_pretrained(
            base_reward_model, checkpoint_path, is_trainable=False
        )
        reward_model.cuda().eval()
        reward_model.pad_token_id = tokenizer.pad_token_id

        def evaluate_responses(example):
            prompt: str = example["prompt"]
            prompt = prompt[prompt.index("Human: ") :]

            responses = example["responses"]
            reward_outputs = []
            for response in responses:
                inputs = tokenizer(
                    prompt + response,
                    return_tensors="pt",
                    max_length=script_args.max_length,
                )
                reward_output = reward_model(
                    inputs.input_ids.cuda(), inputs.attention_mask.cuda()
                )[0]
                reward_outputs.append(reward_output[0].tolist())
            return {
                f"reward_outputs_{model_name}": reward_outputs,
            }

        print(f"Evaluating responses with {model_name}...")
        dataset = dataset.map(
            evaluate_responses,
            batched=False,
        )

    # Combine datasets and output to JSONL
    dataset.to_json(output_fname, orient="records", lines=True)
