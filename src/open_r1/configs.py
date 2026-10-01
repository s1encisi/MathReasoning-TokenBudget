# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Any, Literal

import trl
from open_r1.i18n import t


@dataclass
class DatasetConfig:
    """Configuration for a dataset in a mixture."""

    id: str
    config: str | None = None
    split: str = "train"
    columns: list[str] | None = None
    weight: float | None = None


@dataclass
class DatasetMixtureConfig:
    """Configuration for a mixture of datasets."""

    datasets: list[DatasetConfig]
    seed: int = 0
    test_split_size: float | None = None


@dataclass
class ScriptArguments(trl.ScriptArguments):
    """
    Extended version of ScriptArguments with support for dataset mixtures.

    Args:
        dataset_mixture (`dict[str, Any]` or `None`, *optional*, defaults to `None`):
            Configuration for creating dataset mixtures with advanced options.
            Format:
              dataset_mixture:
                datasets:
                  - id: dataset_id1
                    config: config_name
                    columns:
                      - col1
                      - col2
                    weight: 0.5
                  - id: dataset_id2
                    config: config_name
                    columns:
                      - col1
                      - col2
                    weight: 0.5
                seed: 42
                test_split_size: 0.1
    """

    # Override the dataset_name to make it optional
    dataset_name: str | None = field(
        default=None, metadata={"help": t("cli.configs.script.dataset_name")}
    )
    dataset_mixture: dict[str, Any] | None = field(
        default=None,
        metadata={"help": t("cli.configs.script.dataset_mixture")},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.dataset_mixture is None:
            raise ValueError(t("error.configs.dataset_source_required"))

        if self.dataset_mixture is not None:
            if not isinstance(self.dataset_mixture, dict) or "datasets" not in self.dataset_mixture:
                raise ValueError(t("error.configs.dataset_mixture_shape"))

            datasets_list = []
            datasets_data = self.dataset_mixture.get("datasets", [])

            if isinstance(datasets_data, list):
                for dataset_config in datasets_data:
                    datasets_list.append(
                        DatasetConfig(
                            id=dataset_config.get("id"),
                            config=dataset_config.get("config"),
                            split=dataset_config.get("split", "train"),
                            columns=dataset_config.get("columns"),
                            weight=dataset_config.get("weight", 1.0),
                        )
                    )
            else:
                raise ValueError(t("error.configs.datasets_must_be_list"))

            self.dataset_mixture = DatasetMixtureConfig(
                datasets=datasets_list,
                seed=self.dataset_mixture.get("seed", 0),
                test_split_size=self.dataset_mixture.get("test_split_size", None),
            )

            # Check that column names are consistent across all dataset configs
            columns_sets = [set(dataset.columns) for dataset in datasets_list if dataset.columns is not None]
            if columns_sets:
                first_columns = columns_sets[0]
                if not all(columns == first_columns for columns in columns_sets):
                    raise ValueError(
                        t("error.configs.columns_inconsistent", columns=[list(cols) for cols in columns_sets])
                    )


# TODO: add the shared options with a mixin to reduce code duplication
@dataclass
class GRPOConfig(trl.GRPOConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=list,
        metadata={"help": t("cli.configs.grpo.benchmarks")},
    )
    callbacks: list[str] = field(
        default_factory=list,
        metadata={"help": t("cli.configs.grpo.callbacks")},
    )
    chat_template: str | None = field(default=None, metadata={"help": t("cli.configs.grpo.chat_template")})
    hub_model_revision: str | None = field(
        default="main", metadata={"help": t("cli.configs.grpo.hub_model_revision")}
    )
    num_completions_to_print: int = field(
        default=0, metadata={"help": t("cli.configs.grpo.num_completions_to_print")}
    )
    overwrite_hub_revision: bool = field(
        default=False, metadata={"help": t("cli.configs.grpo.overwrite_hub_revision")}
    )
    push_to_hub_revision: bool = field(default=False, metadata={"help": t("cli.configs.grpo.push_to_hub_revision")})
    system_prompt: str | None = field(
        default=None,
        metadata={"help": t("cli.configs.grpo.system_prompt")},
    )
    wandb_log_unique_prompts: bool = field(
        default=True,
        metadata={"help": t("cli.configs.grpo.wandb_log_unique_prompts")},
    )
    wandb_entity: str | None = field(
        default=None,
        metadata={"help": t("cli.configs.grpo.wandb_entity")},
    )
    wandb_project: str | None = field(
        default=None,
        metadata={"help": t("cli.configs.grpo.wandb_project")},
    )
    wandb_run_group: str | None = field(
        default=None,
        metadata={"help": t("cli.configs.grpo.wandb_run_group")},
    )


@dataclass
class SFTConfig(trl.SFTConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=list,
        metadata={"help": t("cli.configs.sft.benchmarks")},
    )
    callbacks: list[str] = field(
        default_factory=list,
        metadata={"help": t("cli.configs.sft.callbacks")},
    )
    chat_template: str | None = field(default=None, metadata={"help": t("cli.configs.sft.chat_template")})
    system_prompt: str | None = field(
        default=None,
        metadata={"help": t("cli.configs.sft.system_prompt")},
    )
    hub_model_revision: str | None = field(
        default="main",
        metadata={"help": t("cli.configs.sft.hub_model_revision")},
    )
    overwrite_hub_revision: bool = field(
        default=False, metadata={"help": t("cli.configs.sft.overwrite_hub_revision")}
    )
    push_to_hub_revision: bool = field(default=False, metadata={"help": t("cli.configs.sft.push_to_hub_revision")})
    wandb_entity: str | None = field(
        default=None,
        metadata={"help": t("cli.configs.sft.wandb_entity")},
    )
    wandb_project: str | None = field(
        default=None,
        metadata={"help": t("cli.configs.sft.wandb_project")},
    )
    wandb_run_group: str | None = field(
        default=None,
        metadata={"help": t("cli.configs.sft.wandb_run_group")},
    )


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', 'tag_count', 'code', 'ioi_code', 'code_format', 'soft_overlong_punishment'.
        cosine_min_value_wrong (`float`):
            Minimum reward for cosine scaling for wrong answers.
        cosine_max_value_wrong (`float`):
            Maximum reward for cosine scaling for wrong answers.
        cosine_min_value_correct (`float`):
            Minimum reward for cosine scaling for correct answers.
        cosine_max_value_correct (`float`):
            Maximum reward for cosine scaling for correct answers.
        cosine_max_len (`int`):
            Maximum length for cosine scaling.
        code_language (`str`):
            Language for code format reward.
        max_completion_len (`int`):
            Maximum number of tokens in completion.
        soft_punish_cache (`int`):
            Minimum number of tokens in completion.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "tag_count"],
        metadata={"help": t("cli.configs.grpo_script.reward_funcs")},
    )
    cosine_min_value_wrong: float = field(
        default=0.0,
        metadata={"help": t("cli.configs.grpo_script.cosine_min_value_wrong")},
    )
    cosine_max_value_wrong: float = field(
        default=-0.5,
        metadata={"help": t("cli.configs.grpo_script.cosine_max_value_wrong")},
    )
    cosine_min_value_correct: float = field(
        default=0.5,
        metadata={"help": t("cli.configs.grpo_script.cosine_min_value_correct")},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": t("cli.configs.grpo_script.cosine_max_value_correct")},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": t("cli.configs.grpo_script.cosine_max_len")},
    )
    repetition_n_grams: int = field(
        default=3,
        metadata={"help": t("cli.configs.grpo_script.repetition_n_grams")},
    )
    repetition_max_penalty: float = field(
        default=-1.0,
        metadata={"help": t("cli.configs.grpo_script.repetition_max_penalty")},
    )
    code_language: str = field(
        default="python",
        # '(?:python|cpp)'
        metadata={
            "help": t("cli.configs.grpo_script.code_language"),
            "choices": ["python", "javascript", "r", "java", "bash", "cpp"],
        },
    )
    code_eval_test_batch_size: int = field(
        default=1,
        metadata={"help": t("cli.configs.grpo_script.code_eval_test_batch_size")},
    )
    code_eval_scoring_mode: Literal["pass_fail", "partial", "weighted_sum"] = field(
        default="weighted_sum",
        metadata={"help": t("cli.configs.grpo_script.code_eval_scoring_mode")},
    )
    parallel_code_exec_per_proc: int = field(
        default=2,
        metadata={"help": t("cli.configs.grpo_script.parallel_code_exec_per_proc")},
    )

    dataset_prompt_column: str = field(
        default="prompt",
        metadata={"help": t("cli.configs.grpo_script.dataset_prompt_column")},
    )

    e2b_router_url: str | None = field(
        default=None,
        metadata={"help": t("cli.configs.grpo_script.e2b_router_url")},
    )

    morph_router_url: str | None = field(
        default=None,
        metadata={"help": t("cli.configs.grpo_script.morph_router_url")},
    )

    code_provider: str | None = field(
        default="e2b",
        metadata={
            "help": t("cli.configs.grpo_script.code_provider"),
            "choices": ["e2b", "local", "morph"],
        },
    )

    ioi_provider: str | None = field(
        default="piston",
        metadata={
            "help": t("cli.configs.grpo_script.ioi_provider"),
            "choices": ["piston", "morph"],
        },
    )

    max_completion_len: int = field(
        default=16384,
        metadata={"help": t("cli.configs.grpo_script.max_completion_len")},
    )
    soft_punish_cache: int = field(
        default=4096,
        metadata={"help": t("cli.configs.grpo_script.soft_punish_cache")},
    )
