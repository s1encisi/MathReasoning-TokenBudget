# coding=utf-8
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
"""国际化（i18n）模块的单元测试。

只依赖标准库，可在没有 GPU / trl / vLLM 的环境下运行。
"""

import datetime
import unittest
from contextlib import contextmanager

from open_r1.i18n import (
    DEFAULT_LOCALE,
    LocaleAwareHelpFormatter,
    available_locales,
    collation_key,
    display_width,
    format_compact_number,
    format_currency,
    format_date,
    format_datetime,
    format_duration,
    format_int,
    format_number,
    format_percent,
    format_time,
    get_locale,
    missing_keys,
    normalize_locale,
    pad,
    render_table,
    set_locale,
    sort_by_locale,
    t,
    wrap,
)
from open_r1.i18n.catalog import load_catalog


ZH = "zh_CN"
EN = "en_US"


class TestCatalog(unittest.TestCase):
    def setUp(self):
        self._previous = get_locale()
        set_locale(ZH)

    def tearDown(self):
        set_locale(self._previous)

    def test_locales_present(self):
        self.assertIn(ZH, available_locales())
        self.assertIn(EN, available_locales())

    def test_zh_catalog_is_complete(self):
        self.assertEqual(missing_keys(ZH, EN), [])

    def test_en_catalog_is_complete(self):
        self.assertEqual(missing_keys(EN, ZH), [])

    def test_lookup_renders_placeholders(self):
        self.assertEqual(t("error.evaluation.unknown_benchmark", benchmark="aime24"), "未知的基准测试：aime24")

    def test_missing_key_falls_back_to_key(self):
        self.assertEqual(t("this.key.does.not.exist"), "this.key.does.not.exist")

    def test_bad_placeholders_do_not_raise(self):
        # 传入的参数与模板不匹配时应原样返回模板，而不是抛异常
        self.assertEqual(t("error.evaluation.unknown_benchmark", wrong=1), "未知的基准测试：{benchmark}")

    def test_switch_locale(self):
        set_locale(EN)
        self.assertEqual(t("error.evaluation.unknown_benchmark", benchmark="aime24"), "Unknown benchmark aime24")
        set_locale(ZH)
        self.assertEqual(t("error.evaluation.unknown_benchmark", benchmark="aime24"), "未知的基准测试：aime24")

    def test_all_prose_values_are_translated(self):
        """`argparse` / `cli` / `error` / `log` / `msg` 下的文案必须含 CJK 字符。

        `fmt.*` 是分隔符、模板与代码表，天然不含中文，不在此断言范围内。
        """
        catalog = load_catalog(ZH)
        untranslated = []
        for key, value in catalog.items():
            if not key.startswith(("argparse.", "cli.", "error.", "log.", "msg.")):
                continue
            if not isinstance(value, str):
                continue
            # 汉字、CJK 标点（如「：」U+FF1A）与全角符号都算作已中文化
            if not any(
                "\u4e00" <= char <= "\u9fff" or "\u3000" <= char <= "\u303f" or "\uff00" <= char <= "\uffef"
                for char in value
            ):
                untranslated.append(key)
        self.assertEqual(untranslated, [], f"以下键仍为英文：{untranslated}")

    def test_normalize_locale(self):
        self.assertEqual(normalize_locale("zh"), ZH)
        self.assertEqual(normalize_locale("zh-CN"), ZH)
        self.assertEqual(normalize_locale("zh_CN.UTF-8"), ZH)
        self.assertEqual(normalize_locale("en"), EN)
        self.assertEqual(normalize_locale("en_US.UTF-8"), EN)
        # `C` / `POSIX` 只是占位值，不应静默退回英文
        self.assertEqual(normalize_locale("C"), DEFAULT_LOCALE)
        self.assertEqual(normalize_locale("C.UTF-8"), DEFAULT_LOCALE)
        self.assertEqual(normalize_locale(None), DEFAULT_LOCALE)


class TestFormatting(unittest.TestCase):
    def setUp(self):
        self._previous = get_locale()
        set_locale(ZH)

    def tearDown(self):
        set_locale(self._previous)

    def test_display_width(self):
        self.assertEqual(display_width("abc"), 3)
        self.assertEqual(display_width("中文"), 4)
        self.assertEqual(display_width("中文abc"), 7)

    def test_pad_aligns_by_display_width(self):
        self.assertEqual(display_width(pad("中文", 10)), 10)
        self.assertEqual(display_width(pad("中文", 10, align="right")), 10)
        self.assertEqual(display_width(pad("中文", 11, align="center")), 11)

    def test_wrap_respects_display_width(self):
        lines = wrap("中文中文中文中文中文中文", 8)
        self.assertTrue(all(display_width(line) <= 8 for line in lines))
        self.assertEqual("".join(lines), "中文中文中文中文中文中文")

    def test_wrap_mixed_script(self):
        lines = wrap("训练完成后运行 AIME 基准测试。", 12)
        self.assertTrue(all(display_width(line) <= 12 for line in lines))

    def test_wrap_no_line_starts_with_forbidden_punctuation(self):
        forbidden = "。，、；：？！）】》」』,.!?:;)]}"
        text = "先写一句话，然后再写另一句话，最后再写一句。"
        for line in wrap(text, 8):
            if line:
                self.assertNotIn(line[0], forbidden)

    def test_number_and_currency(self):
        self.assertEqual(format_number(1234567.891), "1,234,567.89")
        self.assertEqual(format_int(1234567), "1,234,567")
        self.assertEqual(format_percent(0.1534), "15.34%")
        self.assertEqual(format_currency(1234.5), "¥1,234.50")
        self.assertEqual(format_compact_number(235000000), "2.35亿")
        self.assertEqual(format_duration(12.345), "12.35 秒")

    def test_localized_number_and_currency_in_english(self):
        set_locale(EN)
        self.assertEqual(format_currency(1234.5), "$1,234.50")
        self.assertEqual(format_compact_number(235000000), "235.00M")
        set_locale(ZH)

    def test_date_and_time(self):
        moment = datetime.datetime(2026, 9, 30, 9, 11, 24)
        self.assertEqual(format_date(moment), "2026年9月30日")
        self.assertEqual(format_date(moment, style="short"), "2026-09-30")
        self.assertEqual(format_time(moment), "09:11:24")
        self.assertEqual(format_datetime(moment), "2026-09-30 09:11:24")
        self.assertEqual(format_datetime(moment, style="long"), "2026年9月30日 09:11:24")

    def test_render_table_columns_align(self):
        table = render_table(["样本数", "执行耗时（秒）"], [["16", "12.34"], ["256", "120.55"]])
        lines = table.splitlines()
        widths = {display_width(line) for line in lines}
        self.assertEqual(len(widths), 1, "表格每一行的显示宽度必须一致")

    def test_render_table_respects_max_width(self):
        headers = ["样本数", "执行耗时（秒）", "平均奖励", "最小奖励", "最大奖励"]
        rows = [["16", "12.34", "0.9812", "0.5", "1.0"], ["256", "120.55", "0.9735", "0.2", "1.0"]]
        table = render_table(headers, rows, max_width=44)
        self.assertTrue(all(display_width(line) <= 44 for line in table.splitlines()))

    def test_sort_by_locale_is_total_and_stable(self):
        items = ["北京", "上海", "apple", "广州", "Banana"]
        ordered = sort_by_locale(items)
        self.assertEqual(sorted(ordered), sorted(items))
        self.assertEqual(sort_by_locale(items), sort_by_locale(items))
        self.assertIsInstance(collation_key("中文"), tuple)


class TestCliHelpFormatter(unittest.TestCase):
    def setUp(self):
        self._previous = get_locale()
        set_locale(ZH)

    def tearDown(self):
        set_locale(self._previous)

    def test_help_columns_align_for_chinese(self):
        import argparse

        parser = argparse.ArgumentParser(prog="demo", formatter_class=LocaleAwareHelpFormatter)
        parser.add_argument("--model", type=str, help="生成所用的模型名称")
        parser.add_argument("--dataset-name", type=str, help="要加载的 HuggingFace 数据集")
        parser.add_argument("--max-new-tokens", type=int, help="生成的最大新 token 数")
        usage = parser.format_usage()
        for line in parser.format_help().splitlines():
            self.assertLessEqual(display_width(line), 120)
        self.assertIn("demo", usage)

    def test_builtin_argparse_strings_are_localized(self):
        from open_r1.i18n.argparse_help import make_parser

        parser = make_parser(prog="demo", description="演示")
        parser.add_argument("--model", type=str, required=True, help="生成所用的模型名称")
        help_text = parser.format_help()
        self.assertIn("用法：", help_text)
        self.assertIn("显示此帮助信息并退出", help_text)

    def test_parse_error_message_is_localized(self):
        from open_r1.i18n.argparse_help import make_parser

        parser = make_parser(prog="demo")
        parser.add_argument("--model", type=str, required=True, help="生成所用的模型名称")
        with self.assertRaises(SystemExit):
            with _capture_stderr() as stream:
                parser.parse_args([])
        self.assertIn("缺少以下必需参数", stream.getvalue())


@contextmanager
def _capture_stderr():
    """捕获写入 stderr 的内容，用于断言 argparse 的错误输出。"""
    import io
    from contextlib import redirect_stderr

    buffer = io.StringIO()
    with redirect_stderr(buffer):
        yield buffer


if __name__ == "__main__":
    unittest.main()
