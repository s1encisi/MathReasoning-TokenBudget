import argparse
from transformers import AutoConfig
from math import gcd

from open_r1.i18n import t


def get_tensor_parallel_size(model_name: str, revision: str = None, default_tp: int = 8) -> int:
    try:
        config = AutoConfig.from_pretrained(model_name, revision=revision, trust_remote_code=True)
        num_heads = getattr(config, 'num_attention_heads', None)

        if num_heads is not None and num_heads % default_tp != 0:
            tp = gcd(num_heads, default_tp)
            return max(tp, 1)
        else:
            return default_tp
    except Exception as e:
        print(t("msg.tensor_parallel.config_failed", model=model_name, revision=revision, error=e))
        return default_tp


if __name__ == "__main__":
    from open_r1.i18n.argparse_help import make_parser

    parser = make_parser()
    parser.add_argument("--model_name", type=str, required=True, help=t("cli.tensor_parallel.help.model_name"))
    parser.add_argument("--revision", type=str, default=None, help=t("cli.tensor_parallel.help.revision"))
    parser.add_argument("--default_tp", type=int, default=8, help=t("cli.tensor_parallel.help.default_tp"))

    args = parser.parse_args()

    tp = get_tensor_parallel_size(args.model_name, args.revision, args.default_tp)
    print(tp)
