from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional, Type, Union, cast

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from datasets import Dataset, concatenate_datasets, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import EvalPrediction
from transformers.utils import PaddingStrategy
from typing_extensions import Literal, TypeAlias

RewardModelType: TypeAlias = Literal["base", "mean_and_variance", "categorical"]
DataSubset: TypeAlias = Literal["both", "helpful", "harmless"]


@dataclass
class ScriptArguments:
    local_rank: int = field(default=-1, metadata={"help": "Used for multi-gpu"})
    resume_from_checkpoint: bool = field(
        default=False,
        metadata={"help": "If you want to resume training where it left off."},
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to deepspeed config if using deepspeed. You may need this "
            "if the model that you want to train doesn't fit on a single GPU."
        },
    )
    per_device_train_batch_size: int = field(default=2)
    per_device_eval_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=1)
    learning_rate: float = field(default=3e-6)
    weight_decay: float = field(default=0.001)
    model_name: str = field(
        default="gpt2",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. "
            "E.g. gpt2, gpt2-xl, bert, etc."
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
    reward_model_type: str = field(
        default="base",
        metadata={
            "help": "The type of reward model to use. You can choose between "
            "'base', 'mean_and_variance', or 'categorical'."
        },
    )
    num_atoms: int = field(
        default=10,
        metadata={
            "help": "The number of atoms to use for the categorical reward model."
        },
    )
    entropy_coeff: float = field(
        default=0.1,
        metadata={"help": "The entropy coefficient for the categorical reward model."},
    )
    variance_penalty: float = field(
        default=0.0,
        metadata={
            "help": "The variance penalty for the mean and variance reward model."
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The tokenizer for your model, if left empty will use the default "
            "for your model",
        },
    )
    bf16: bool = field(
        default=True,
        metadata={
            "help": "This essentially cuts the training time in half if you want to "
            "sacrifice a little precision and have a supported GPU."
        },
    )
    num_train_epochs: int = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    train_dataset_size: int = field(
        default=0,
        metadata={"help": "The size of the subset of the training data to use"},
    )
    eval_dataset_size: int = field(
        default=0,
        metadata={"help": "The size of the subset of the eval data to use"},
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: str = field(
        default="adamw_hf",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: str = field(
        default="cosine",
        metadata={"help": "The lr scheduler"},
    )
    max_length: int = field(default=1024)
    eval_first_step: bool = field(
        default=False,
        metadata={"help": "Whether to run eval after the first step"},
    )
    log_dir: str = field(default="data/reward_models/hh_rlhf")


class HHRLHFPreprocessor(object):
    def __init__(self, tokenizer, **tokenizer_kwargs):
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = tokenizer_kwargs

    def __call__(self, examples):
        new_examples: dict = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
            tokenized_chosen = self.tokenizer(chosen, **self.tokenizer_kwargs)
            tokenized_rejected = self.tokenizer(rejected, **self.tokenizer_kwargs)

            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(
                tokenized_chosen["attention_mask"]
            )
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(
                tokenized_rejected["attention_mask"]
            )

        return new_examples


def get_step_decay_lr_lambda(current_step: int, *, num_training_steps: int):
    if current_step < num_training_steps // 3:
        return 1.0
    elif current_step < (2 * num_training_steps) // 3:
        return 0.1
    else:
        return 0.01


def get_cosine_decay_lr_lambda(current_step: int, *, num_training_steps: int):
    return 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * current_step / num_training_steps))


class RewardTrainer(Trainer):
    def __init__(self, *args, lr_lambda=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr_lambda = lr_lambda

    @classmethod
    def per_sample_loss(cls, rewards_chosen, rewards_rejected):
        return -nn.functional.logsigmoid(rewards_chosen - rewards_rejected)

    def loss(self, rewards_chosen, rewards_rejected):
        return torch.mean(self.per_sample_loss(rewards_chosen, rewards_rejected))

    def compute_loss(self, model, inputs, return_outputs=False):
        all_rewards = model(
            torch.concatenate(
                [
                    inputs["input_ids_chosen"],
                    inputs["input_ids_rejected"],
                ],
                dim=0,
            ),
            torch.concatenate(
                [
                    inputs["attention_mask_chosen"],
                    inputs["attention_mask_rejected"],
                ],
                dim=0,
            ),
        )[0]
        all_rewards = all_rewards.reshape(2, -1, all_rewards.shape[-1])
        rewards_chosen = all_rewards[0]
        rewards_rejected = all_rewards[1]
        loss = self.loss(rewards_chosen, rewards_rejected)
        if return_outputs:
            return loss, {
                "rewards_chosen": rewards_chosen,
                "rewards_rejected": rewards_rejected,
            }
        return loss

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if self.lr_lambda is not None:
            lr_lambda = partial(
                self.lr_lambda,
                num_training_steps=num_training_steps,
            )
            self.lr_scheduler = LambdaLR(optimizer, lr_lambda)
            return self.lr_scheduler
        else:
            return super().create_scheduler(num_training_steps, optimizer)

    @classmethod
    def compute_metrics(cls, eval_prediction: EvalPrediction):
        rewards_chosen, rewards_rejected = eval_prediction.predictions
        rewards_chosen = torch.from_numpy(rewards_chosen)
        rewards_rejected = torch.from_numpy(rewards_rejected)

        loss = cls.per_sample_loss(rewards_chosen, rewards_rejected)
        accuracy = torch.mean((loss < np.log(2)).float())

        return {
            "loss": loss.mean().item(),
            "accuracy": accuracy.item(),
        }


class MeanAndVarianceRewardTrainer(RewardTrainer):
    def __init__(self, *args, variance_penalty: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.variance_penalty = variance_penalty

    @classmethod
    def per_sample_loss(cls, rewards_chosen, rewards_rejected):
        mean_chosen = rewards_chosen[:, 0]
        std_chosen = F.softplus(rewards_chosen[:, 1])
        mean_rejected = rewards_rejected[:, 0]
        std_rejected = F.softplus(rewards_rejected[:, 1])

        diff_mean = mean_chosen - mean_rejected
        var_combined = std_chosen**2 + std_rejected**2
        z = diff_mean / torch.sqrt(var_combined)
        return F.softplus(-z * np.sqrt(2 * np.pi))

    def loss(self, rewards_chosen, rewards_rejected):
        std_chosen = F.softplus(rewards_chosen[:, 1])
        std_rejected = F.softplus(rewards_rejected[:, 1])
        variance_loss = (std_chosen**2 + std_rejected**2).mean()

        log_loss = self.per_sample_loss(rewards_chosen, rewards_rejected).mean()

        if self.model.training:
            return log_loss + self.variance_penalty * variance_loss
        else:
            return log_loss


class CategoricalRewardTrainer(RewardTrainer):
    def __init__(self, *args, entropy_coeff: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.entropy_coeff = entropy_coeff

    @classmethod
    def per_sample_loss(cls, rewards_chosen, rewards_rejected):
        num_atoms = rewards_chosen.size()[1]
        device = rewards_chosen.device

        comparison_matrix = torch.empty(
            (num_atoms, num_atoms),
            device=device,
            dtype=rewards_chosen.dtype,
        )
        atom_values = torch.linspace(0, 1, num_atoms, device=device)
        comparison_matrix[:] = atom_values[None, :] > atom_values[:, None]
        comparison_matrix[atom_values[None, :] == atom_values[:, None]] = 0.5

        dist_rejected = rewards_rejected.softmax(1)
        dist_chosen = rewards_chosen.softmax(1)
        prob_chosen = ((dist_rejected @ comparison_matrix) * dist_chosen).sum(dim=1)
        return -prob_chosen.log()

    def loss(self, rewards_chosen, rewards_rejected):
        dist_rejected = rewards_rejected.softmax(1)
        dist_chosen = rewards_chosen.softmax(1)
        mean_dist = torch.concatenate(
            [dist_chosen, dist_rejected],
            dim=0,
        ).mean(dim=0)
        entropy_loss = torch.sum(mean_dist * mean_dist.log())

        log_loss = self.per_sample_loss(rewards_chosen, rewards_rejected).mean()

        if self.model.training:
            return log_loss + self.entropy_coeff * entropy_loss
        else:
            return log_loss


def get_hh_rlhf_dataset(
    data_subset: DataSubset,
    split: Literal["train", "test"],
    dataset_size: int = 0,
    data_path="Anthropic/hh-rlhf",
) -> Dataset:
    datasets: List[Dataset] = []
    if data_path == "Anthropic/hh-rlhf":
        if data_subset == "harmless" or data_subset == "both":
            datasets.append(
                load_dataset(
                    "Anthropic/hh-rlhf", data_dir="harmless-base", split=split
                ).map(lambda data: {"data_subset": "harmless"})
            )
        if data_subset == "helpful" or data_subset == "both":
            datasets.append(
                load_dataset(
                    "Anthropic/hh-rlhf", data_dir="helpful-base", split=split
                ).map(lambda data: {"data_subset": "helpful"})
            )
    else:
        datasets.append(
            load_dataset(data_path, split=split).map(
                lambda data: {"data_subset": data_subset}
            )
        )

    if dataset_size:
        datasets = [
            dataset.select(range(dataset_size // len(datasets))) for dataset in datasets
        ]

    return concatenate_datasets(datasets)


trainer_classes: Dict[RewardModelType, Type[RewardTrainer]] = {
    "base": RewardTrainer,
    "mean_and_variance": MeanAndVarianceRewardTrainer,
    "categorical": CategoricalRewardTrainer,
}


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

    data_subset = cast(DataSubset, script_args.data_subset)
    train_dataset = get_hh_rlhf_dataset(
        data_subset,
        "train",
        script_args.train_dataset_size,
        data_path=script_args.data_path,
    )
    eval_dataset = get_hh_rlhf_dataset(
        data_subset,
        "test",
        script_args.eval_dataset_size,
        data_path=script_args.data_path,
    )

    reward_model_type = cast(RewardModelType, script_args.reward_model_type)

    # Define the training args. Needs to be done before the model is loaded if you
    # are using deepspeed.
    model_name_split = script_args.model_name.split("/")[-1]
    output_name = (
        f"{script_args.log_dir}/{data_subset}/"
        f"{reward_model_type}_{model_name_split}"
        f"__{script_args.train_dataset_size}_{script_args.learning_rate}"
        f"_{script_args.lr_scheduler_type}_{script_args.num_train_epochs}"
    )
    if reward_model_type == "categorical":
        output_name += f"_{script_args.num_atoms}_{script_args.entropy_coeff}"
    elif reward_model_type == "mean_and_variance":
        output_name += f"_{script_args.variance_penalty}"

    trainer_kwargs: Dict[str, Any] = {}
    if script_args.lr_scheduler_type == "step":
        lr_scheduler_type = "constant"
        trainer_kwargs["lr_lambda"] = get_step_decay_lr_lambda
    elif script_args.lr_scheduler_type == "cosine":
        lr_scheduler_type = "constant"
        trainer_kwargs["lr_lambda"] = get_cosine_decay_lr_lambda
    else:
        lr_scheduler_type = script_args.lr_scheduler_type

    training_args = TrainingArguments(
        output_dir=output_name,
        learning_rate=script_args.learning_rate,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        num_train_epochs=script_args.num_train_epochs,
        weight_decay=script_args.weight_decay,
        evaluation_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        deepspeed=script_args.deepspeed,
        local_rank=script_args.local_rank,
        remove_unused_columns=False,
        label_names=[],
        bf16=script_args.bf16,
        logging_strategy="steps",
        logging_steps=10,
        optim=script_args.optim,
        lr_scheduler_type=lr_scheduler_type,
    )
    # Load the value-head model and tokenizer.
    tokenizer_name = (
        script_args.tokenizer_name
        if script_args.tokenizer_name is not None
        else script_args.model_name
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_auth_token=True)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )

    torch.set_anomaly_enabled(True)

    trainer_class = trainer_classes[reward_model_type]
    if reward_model_type == "base":
        num_labels = 1
    elif reward_model_type == "mean_and_variance":
        num_labels = 2
        trainer_kwargs["variance_penalty"] = script_args.variance_penalty
    elif reward_model_type == "categorical":
        num_labels = script_args.num_atoms
        trainer_kwargs["entropy_coeff"] = script_args.entropy_coeff

    model = AutoModelForSequenceClassification.from_pretrained(
        script_args.model_name, num_labels=num_labels, torch_dtype=torch.bfloat16
    )
    # We multiply the final linear layer's weights by 0.01 because this seems to
    # significantly stabilize training and lead to better optimization of the loss.
    model.score.weight.data *= 0.01
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Need to do this for GPT2 and Llama because they doesn't have official pad tokens.
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "right"

    model.config.use_cache = not script_args.gradient_checkpointing
    num_proc = 24  # Can adjust to be higher if you have more processors.
    original_columns = train_dataset.column_names

    train_dataset = train_dataset.map(
        HHRLHFPreprocessor(tokenizer),
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    train_dataset = train_dataset.filter(
        lambda x: len(x["input_ids_chosen"]) <= script_args.max_length
        and len(x["input_ids_rejected"]) <= script_args.max_length
    )

    eval_dataset = eval_dataset.map(
        HHRLHFPreprocessor(tokenizer),
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    eval_dataset = eval_dataset.filter(
        lambda x: len(x["input_ids_chosen"]) <= script_args.max_length
        and len(x["input_ids_rejected"]) <= script_args.max_length
    )

    # We need to define a special data collator that batches the data in our j vs k format.
    @dataclass
    class RewardDataCollatorWithPadding:
        tokenizer: PreTrainedTokenizerBase
        padding: Union[bool, str, PaddingStrategy] = True
        max_length: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None
        return_tensors: str = "pt"

        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
            features_chosen = []
            features_rejected = []
            for feature in features:
                features_chosen.append(
                    {
                        "input_ids": feature["input_ids_chosen"],
                        "attention_mask": feature["attention_mask_chosen"],
                    }
                )
                features_rejected.append(
                    {
                        "input_ids": feature["input_ids_rejected"],
                        "attention_mask": feature["attention_mask_rejected"],
                    }
                )
            batch = self.tokenizer.pad(
                features_chosen + features_rejected,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
            input_ids = batch["input_ids"].view(2, -1, batch["input_ids"].shape[-1])
            attention_mask = batch["attention_mask"].view(
                2, -1, batch["attention_mask"].shape[-1]
            )
            return {
                "input_ids_chosen": input_ids[0],
                "attention_mask_chosen": attention_mask[0],
                "input_ids_rejected": input_ids[1],
                "attention_mask_rejected": attention_mask[1],
                "return_loss": True,
            }

    # Train the model.
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=trainer_class.compute_metrics,
        data_collator=RewardDataCollatorWithPadding(
            tokenizer=tokenizer,
            max_length=script_args.max_length,
            pad_to_multiple_of=64,
        ),
        **trainer_kwargs,
    )

    trainer.train(script_args.resume_from_checkpoint)

    print("Saving last checkpoint of the model")
    model.save_pretrained(output_name + "_peft_last_checkpoint")
