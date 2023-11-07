import os
from dataclasses import dataclass, field
from typing import Optional, cast

import torch
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)

from .train_llm_preference_model import (
    DataSubset,
    HHRLHFPreprocessor,
    get_hh_rlhf_dataset,
)


@dataclass
class ScriptArguments:
    reward_model_checkpoint: str = field(
        metadata={"help": "Path to the trained reward model checkpoint."}
    )
    output: Optional[str] = field(
        default=None,
        metadata={"help": "JSONL file where results will be stored."},
    )
    batch_size: Optional[int] = field(default=1)
    model_name: Optional[str] = field(default="gpt2")
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The tokenizer for your model, if left empty will use the default "
            "for your model",
        },
    )
    data_path: str = field(
        default="Anthropic/hh-rlhf",
    )
    data_subset: str = field(
        default="both",
        metadata={
            "help": "Which subset of the data to use. You can choose between 'both', "
            "'helpful', or 'harmless'."
        },
    )
    eval_dataset_size: int = field(
        default=0,
        metadata={"help": "The size of the subset of the eval data to use"},
    )
    num_outputs: int = field(
        default=1,
        metadata={"help": "The number of outputs from the model."},
    )
    max_length: int = field(default=1024)
    bf16: bool = field(
        default=True,
        metadata={"help": "Whether to use bfloat16 precision."},
    )


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

    data_subset = cast(DataSubset, script_args.data_subset)

    output_fname = script_args.output
    if output_fname is None:
        output_fname = os.path.join(
            script_args.reward_model_checkpoint, f"eval_results_{data_subset}.jsonl"
        )

    eval_dataset = get_hh_rlhf_dataset(
        data_subset,
        "test",
        script_args.eval_dataset_size,
        data_path=script_args.data_path,
    )

    # Load the value-head model and tokenizer.
    tokenizer_name = (
        script_args.tokenizer_name
        if script_args.tokenizer_name is not None
        else script_args.model_name
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_auth_token=True)

    peft_config = LoraConfig.from_pretrained(script_args.reward_model_checkpoint)
    model_kwargs = {}
    if script_args.bf16:
        model_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForSequenceClassification.from_pretrained(
        script_args.model_name,
        num_labels=script_args.num_outputs,
        **model_kwargs,
    )
    model = PeftModel.from_pretrained(
        model, script_args.reward_model_checkpoint, is_trainable=False
    )
    model = model.to("cuda").eval()

    # Need to do this for GPT2 and Llama because they doesn't have official pad tokens.
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "right"

    num_proc = 24  # Can adjust to be higher if you have more processors.

    eval_dataset = eval_dataset.map(
        HHRLHFPreprocessor(tokenizer, padding=True, max_length=script_args.max_length),
        batched=True,
        num_proc=num_proc,
    )
    eval_dataset = eval_dataset.filter(
        lambda x: len(x["input_ids_chosen"]) <= script_args.max_length
        and len(x["input_ids_rejected"]) <= script_args.max_length
    )

    def compute_predictions(example):
        output = {}
        for key in ["chosen", "rejected"]:
            batch = tokenizer.pad(
                {
                    "input_ids": example[f"input_ids_{key}"],
                },
                padding=True,
                max_length=script_args.max_length,
                pad_to_multiple_of=64,
                return_tensors="pt",
            )
            with torch.no_grad():
                output[f"reward_output_{key}"] = model(
                    input_ids=batch["input_ids"].to("cuda"),
                    attention_mask=batch["attention_mask"].to("cuda"),
                )[0].tolist()
        return output

    eval_results = eval_dataset.map(
        compute_predictions,
        remove_columns=[
            "input_ids_chosen",
            "input_ids_rejected",
            "attention_mask_chosen",
            "attention_mask_rejected",
        ],
        batched=True,
        batch_size=script_args.batch_size,
    )
    eval_results.to_json(output_fname, orient="records", lines=True)
