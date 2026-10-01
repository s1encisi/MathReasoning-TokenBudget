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


from distilabel.llms import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import StepResources
from distilabel.steps.tasks import TextGeneration
from open_r1.i18n import t


def build_distilabel_pipeline(
    model: str,
    base_url: str = "http://localhost:8000/v1",
    prompt_column: str | None = None,
    prompt_template: str = "{{ instruction }}",
    temperature: float | None = None,
    top_p: float | None = None,
    max_new_tokens: int = 8192,
    num_generations: int = 1,
    input_batch_size: int = 64,
    client_replicas: int = 1,
    timeout: int = 900,
    retries: int = 0,
) -> Pipeline:
    generation_kwargs = {"max_new_tokens": max_new_tokens}

    if temperature is not None:
        generation_kwargs["temperature"] = temperature

    if top_p is not None:
        generation_kwargs["top_p"] = top_p

    with Pipeline().ray() as pipeline:
        TextGeneration(
            llm=OpenAILLM(
                base_url=base_url,
                api_key="something",
                model=model,
                timeout=timeout,
                max_retries=retries,
                generation_kwargs=generation_kwargs,
            ),
            template=prompt_template,
            input_mappings=({"instruction": prompt_column} if prompt_column is not None else {}),
            input_batch_size=input_batch_size,
            num_generations=num_generations,
            group_generations=True,
            resources=StepResources(replicas=client_replicas),
        )

    return pipeline


if __name__ == "__main__":
    from datasets import load_dataset

    from open_r1.i18n.argparse_help import make_parser

    parser = make_parser(description=t("cli.generate.description"))
    parser.add_argument(
        "--hf-dataset",
        type=str,
        required=True,
        help=t("cli.generate.help.hf_dataset"),
    )
    parser.add_argument(
        "--hf-dataset-config",
        type=str,
        required=False,
        help=t("cli.generate.help.hf_dataset_config"),
    )
    parser.add_argument(
        "--hf-dataset-split",
        type=str,
        default="train",
        help=t("cli.generate.help.hf_dataset_split"),
    )
    parser.add_argument(
        "--prompt-column",
        type=str,
        default="prompt",
        help=t("cli.generate.help.prompt_column"),
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="{{ instruction }}",
        help=t("cli.generate.help.prompt_template"),
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=t("cli.generate.help.model"),
    )
    parser.add_argument(
        "--vllm-server-url",
        type=str,
        default="http://localhost:8000/v1",
        help=t("cli.generate.help.vllm_server_url"),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help=t("cli.generate.help.temperature"),
    )
    parser.add_argument(
        "--top-p",
        type=float,
        help=t("cli.generate.help.top_p"),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8192,
        help=t("cli.generate.help.max_new_tokens"),
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=1,
        help=t("cli.generate.help.num_generations"),
    )
    parser.add_argument(
        "--input-batch-size",
        type=int,
        default=64,
        help=t("cli.generate.help.input_batch_size"),
    )
    parser.add_argument(
        "--client-replicas",
        type=int,
        default=1,
        help=t("cli.generate.help.client_replicas"),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help=t("cli.generate.help.timeout"),
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=0,
        help=t("cli.generate.help.retries"),
    )
    parser.add_argument(
        "--hf-output-dataset",
        type=str,
        required=False,
        help=t("cli.generate.help.hf_output_dataset"),
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help=t("cli.generate.help.private"),
    )

    args = parser.parse_args()

    print(f"\n{t('msg.generate.running_with_arguments')}")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    print(
        t(
            "msg.generate.loading_dataset",
            dataset=args.hf_dataset,
            config=args.hf_dataset_config,
            split=args.hf_dataset_split,
        )
    )
    dataset = load_dataset(args.hf_dataset, args.hf_dataset_config, split=args.hf_dataset_split)
    print(t("msg.generate.dataset_loaded"))

    pipeline = build_distilabel_pipeline(
        model=args.model,
        base_url=args.vllm_server_url,
        prompt_template=args.prompt_template,
        prompt_column=args.prompt_column,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        num_generations=args.num_generations,
        input_batch_size=args.input_batch_size,
        client_replicas=args.client_replicas,
        timeout=args.timeout,
        retries=args.retries,
    )

    print(t("msg.generate.running_pipeline"))
    distiset = pipeline.run(
        dataset=dataset,
        dataset_batch_size=args.input_batch_size * 1000,
        use_cache=False,
    )
    print(t("msg.generate.pipeline_finished"))

    if args.hf_output_dataset:
        print(t("msg.generate.pushing_dataset", repo=args.hf_output_dataset))
        distiset.push_to_hub(args.hf_output_dataset, private=args.private)
        print(t("msg.generate.dataset_pushed"))
