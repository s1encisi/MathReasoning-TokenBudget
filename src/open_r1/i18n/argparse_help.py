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
"""适配中文的 `argparse` 帮助信息格式化器。

标准库 `argparse.HelpFormatter` 一律使用 `len()` 计算宽度，中文帮助文本会被
低估显示宽度，导致「选项列与说明列重叠」；换行也只能在半角空格处断开，
中文长句会超出终端宽度。本模块的 `LocaleAwareHelpFormatter` 做了三处修正：

* 用 `display_width()` 计算选项名（invocation）宽度并做填充；
* 用 `wrap()` 做东亚宽度感知的换行，并处理中文避头尾标点；
* 终端宽度取自 `shutil.get_terminal_size()`，随窗口自适应。
"""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

from .catalog import t
from .formatting import display_width, pad, wrap


# argparse 的抑制常量，`action.help is SUPPRESS` 表示该选项不出现在帮助里。
_SUPPRESS = argparse.SUPPRESS

# argparse 内部的固定文案由 `argparse._`（gettext）渲染，无法按实例覆盖。
# 这里把已知的英文原文映射到语言资源键，未收录的原文保持原样。
# 注意：Python 3.13 起部分文案改为具名占位符并去掉了标题后的冒号，
# 因此 3.10-3.12 与 3.13+ 两种写法都要登记。
_ARGPARSE_KEY_BY_TEXT = {
    "usage: ": "usage",
    "options": "options",  # 3.13+
    "optional arguments:": "optional_arguments",  # <= 3.12
    "positional arguments": "positional_arguments",  # 3.13+
    "positional arguments:": "positional_arguments",  # <= 3.12
    "%(heading)s:": "heading",
    "subcommands": "subcommands",
    "show this help message and exit": "help",
    " (default: %(default)s)": "default_suffix",
    "%(prog)s: error: %(message)s\n": "error",
    "%(prog)s: warning: %(message)s\n": "warning",
    "unrecognized arguments: %s": "unrecognized_arguments",
    "the following arguments are required: %s": "required_arguments",
    "one of the arguments %s is required": "one_of_required",
    "expected one argument": "expected_one_argument",  # 3.13+
    "argument %s: expected one argument": "expected_one_argument",  # <= 3.12
    "expected at least one argument": "expected_at_least_one",
    "expected at most one argument": "expected_at_most_one",
    "invalid choice: %(value)r (choose from %(choices)s)": "invalid_choice",  # 3.13+
    "invalid choice: %r (choose from %s)": "invalid_choice",  # <= 3.12
    "invalid %(type)s value: %(value)r": "invalid_value",  # 3.13+
    "argument %s: invalid %s value: %r": "invalid_value",  # <= 3.12
    "argument %(argument_name)s: %(message)s": "argument_prefix",  # 3.13+
    "not allowed with argument %s": "mutually_exclusive",  # 3.13+
    "argument %s: not allowed with argument %s": "mutually_exclusive",  # <= 3.12
    "ambiguous option: %(option)s could match %(matches)s": "ambiguous_option",
    "ignored explicit argument %r": "ignored_explicit_argument",
}


def _localized_argparse_text(text: str) -> str:
    """把 argparse 内置文案翻译成当前语言；未收录的原文原样返回。"""
    key = _ARGPARSE_KEY_BY_TEXT.get(text)
    if key is None:
        return text
    translated = t(f"argparse.{key}")
    return translated if translated != f"argparse.{key}" else text


@contextmanager
def _localized_argparse() -> Iterator[None]:
    """在受限范围内替换 `argparse._`，避免影响进程内其它解析器的输出。"""
    original = getattr(argparse, "_", None)
    argparse._ = _localized_argparse_text  # argparse 的 gettext 钩子，必须按名替换
    try:
        yield
    finally:
        argparse._ = original


class LocaleAwareHelpFormatter(argparse.HelpFormatter):
    """按显示宽度对齐与换行的帮助格式化器。"""

    def __init__(
        self,
        prog: str,
        indent_increment: int = 2,
        max_help_position: int = 24,
        width: int | None = None,
    ) -> None:
        if width is None:
            width = max(shutil.get_terminal_size().columns - 2, 40)
        super().__init__(prog, indent_increment=indent_increment, max_help_position=max_help_position, width=width)

    # -- 宽度修正 ---------------------------------------------------------
    def add_arguments(self, actions: Sequence[argparse.Action]) -> None:
        """重新按显示宽度统计最长的选项名，避免中文下低估列宽。"""
        super().add_arguments(actions)
        widths = [
            display_width(self._format_action_invocation(action)) for action in actions if action.help is not _SUPPRESS
        ]
        if widths:
            self._action_max_length = max(self._action_max_length, max(widths))

    def _display_width(self, text: str) -> int:
        return display_width(text)

    # -- 换行修正 ---------------------------------------------------------
    def _split_lines(self, text: str, width: int) -> list:
        return wrap(text, width)

    def _fill_text(self, text: str, width: int, indent: str) -> str:
        lines = wrap(text, max(width - display_width(indent), 2))
        return "\n".join(indent + line for line in lines)

    # -- 对齐修正 ---------------------------------------------------------
    def _format_action(self, action: argparse.Action) -> str:
        help_position = min(self._action_max_length + 2, self._max_help_position)
        help_width = max(self._width - help_position, 11)
        action_width = help_position - self._current_indent - 2
        action_header = self._format_action_invocation(action)

        if not action.help:
            action_header = f"{' ' * self._current_indent}{action_header}\n"
            indent_first = 0
        elif display_width(action_header) <= action_width:
            # 用显示宽度填充，中文选项名也能与说明列对齐
            action_header = f"{' ' * self._current_indent}{pad(action_header, action_width)}  "
            indent_first = 0
        else:
            action_header = f"{' ' * self._current_indent}{action_header}\n"
            indent_first = help_position

        parts = [action_header]

        if action.help and action.help.strip():
            help_text = self._expand_help(action)
            if help_text:
                help_lines = self._split_lines(help_text, help_width)
                parts.append(f"{' ' * indent_first}{help_lines[0]}\n")
                for line in help_lines[1:]:
                    parts.append(f"{' ' * help_position}{line}\n")
        elif not action_header.endswith("\n"):
            parts.append("\n")

        for subaction in self._iter_indented_subactions(action):
            parts.append(self._format_action(subaction))

        return self._join_parts(parts)

    def format_help(self) -> str:
        with _localized_argparse():
            return super().format_help()


class LocaleAwareArgumentParser(argparse.ArgumentParser):
    """在生成帮助与报错信息时使用本地化文案的解析器。

    分组标题（如 `options`）在构造期就已生成，因此构造过程本身也要置于
    本地化上下文中；解析期抛出的错误同理。
    """

    def __init__(self, *args, **kwargs) -> None:
        with _localized_argparse():
            super().__init__(*args, **kwargs)
        # `-h/--help` 的说明由 argparse 在构造期写入，这里显式覆盖
        for action in self._actions:
            if isinstance(action, argparse._HelpAction):
                action.help = t("argparse.help")

    def parse_known_args(self, args=None, namespace=None):
        with _localized_argparse():
            return super().parse_known_args(args, namespace)

    def format_help(self) -> str:
        with _localized_argparse():
            return super().format_help()

    def format_usage(self) -> str:
        with _localized_argparse():
            return super().format_usage()

    def error(self, message: str) -> None:
        with _localized_argparse():
            super().error(message)


def make_parser(
    description: str | None = None,
    prog: str | None = None,
    epilog: str | None = None,
) -> LocaleAwareArgumentParser:
    """创建一个已绑定中文友好格式化器的 `ArgumentParser`。"""
    return LocaleAwareArgumentParser(
        prog=prog,
        description=description,
        epilog=epilog,
        formatter_class=LocaleAwareHelpFormatter,
    )
