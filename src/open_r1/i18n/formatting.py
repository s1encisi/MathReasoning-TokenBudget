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
"""面向中文场景的格式化与排版工具。

涵盖四类能力：

1. **数值 / 货币 / 百分比 / 日期时间**：全部由语言资源中的 `fmt.*` 配置驱动，
   新增语言时无需改代码。
2. **排序（collation）**：提供 `collation_key()`，优先使用系统区域排序规则，
   不可用时降级为 Unicode 码位排序。
3. **东亚宽度计算**：`display_width()` 依据 `unicodedata.east_asian_width` 判定
   全角/半角，解决中英混排下 `len()` 低估显示宽度导致的对齐错位。
4. **自动换行与表格排版**：`wrap()` 按显示宽度断行并处理中文「避头尾」标点，
   `render_table()` 按显示宽度对齐列宽并支持总宽自适应压缩。
"""

from __future__ import annotations

import datetime as _dt
import locale as _locale
import math
import unicodedata
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from .catalog import config, get_locale


# `str.maketrans` 不适用，这里直接给出日期模板支持的占位符。
_DATE_TOKENS = ("YYYY", "YY", "MM", "M", "DD", "D", "HH", "hh", "mm", "ss", "WD")

# 中文排版「避头尾」规则：这些字符不允许出现在行首。
_DEFAULT_LINE_START_FORBIDDEN = "。．，、；：？！）”』」】》〉」’”%‰、…～·,.!?:;)]}>/\\-=+|~"
# 这些字符不允许出现在行尾。
_DEFAULT_LINE_END_FORBIDDEN = "（『「【《〈‘“([{</\\-=+|~$￥"

_ZH_COLLATION_LOCALE_CANDIDATES = (
    "zh_CN.UTF-8",
    "zh_CN.utf8",
    "zh_CN",
    "Chinese (Simplified)_China.936",
    "Chinese_China.936",
    "zh-CN",
)


def _line_start_forbidden() -> str:
    value = config("fmt.wrap.line_start_forbidden", None)
    return value if isinstance(value, str) and value else _DEFAULT_LINE_START_FORBIDDEN


def _line_end_forbidden() -> str:
    value = config("fmt.wrap.line_end_forbidden", None)
    return value if isinstance(value, str) and value else _DEFAULT_LINE_END_FORBIDDEN


# ---------------------------------------------------------------------------
# 显示宽度
# ---------------------------------------------------------------------------


def display_width(text: str, ambiguous_wide: Optional[bool] = None) -> int:
    """计算字符串在等宽终端下的显示宽度（全角字符计 2）。

    Args:
        text: 待测量的字符串。
        ambiguous_wide: `East Asian Ambiguous` 字符是否按全角处理。
            为 `None` 时读取语言资源中的 `fmt.width.ambiguous_wide`。
    """
    if ambiguous_wide is None:
        ambiguous_wide = bool(config("fmt.width.ambiguous_wide", False))
    total = 0
    for char in text:
        if unicodedata.combining(char):
            continue
        category = unicodedata.east_asian_width(char)
        if category in ("W", "F"):
            total += 2
        elif category == "A":
            total += 2 if ambiguous_wide else 1
        else:
            total += 1
    return total


def pad(text: str, width: int, align: str = "left", fillchar: str = " ") -> str:
    """按显示宽度填充字符串，使中英混排时列对齐。

    Args:
        text: 原始文本。
        width: 目标显示宽度；小于文本自身宽度时不做裁剪。
        align: `left` / `right` / `center`。
        fillchar: 填充字符，长度必须为 1。
    """
    if len(fillchar) != 1:
        raise ValueError("fillchar must be exactly one character")
    gap = width - display_width(text)
    if gap <= 0:
        return text
    if align == "right":
        return fillchar * gap + text
    if align == "center":
        left = gap // 2
        return fillchar * left + text + fillchar * (gap - left)
    return text + fillchar * gap


# ---------------------------------------------------------------------------
# 换行
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> List[Tuple[str, bool]]:
    """切分为不可分割的排版单元。

    返回 `(token, breakable_before)` 列表：连续拉丁词/数字整体不可断，
    单个中日韩字符之间可断，空格为天然断点。
    """
    tokens: List[Tuple[str, bool]] = []
    is_wide = lambda ch: unicodedata.east_asian_width(ch) in ("W", "F")  # noqa: E731
    buffer: List[str] = []

    def flush() -> None:
        if buffer:
            tokens.append(("".join(buffer), True))
            buffer.clear()

    # URL、路径、版本号等不应在标点处断开，因此把这些字符也并入「词」的范畴
    unbreakable_punctuation = "_-.~:/?#[]@!$&'()*+,;=%"
    for char in text:
        if char.isspace():
            flush()
            tokens.append((char, True))
        elif is_wide(char):
            flush()
            tokens.append((char, True))
        elif char.isalnum() or char in unbreakable_punctuation:
            buffer.append(char)
        else:
            flush()
            tokens.append((char, False))
    flush()
    return tokens


def wrap(text: str, width: int) -> List[str]:
    """按显示宽度换行，并处理中文避头尾标点。

    Args:
        text: 待换行文本。
        width: 单行最大显示宽度（至少为 2）。
    """
    width = max(int(width), 2)
    start_forbidden = _line_start_forbidden()
    end_forbidden = _line_end_forbidden()

    lines: List[str] = []
    current: List[str] = []
    current_width = 0

    def flush_line() -> None:
        nonlocal current, current_width
        if current:
            lines.append("".join(current).rstrip())
            current = []
            current_width = 0

    for token, breakable in _tokenize(text):
        token_width = display_width(token)
        if token.isspace():
            if current and current_width + 1 <= width:
                current.append(token)
                current_width += 1
            continue
        if current_width + token_width > width:
            if not current:
                # 单个单元就超宽：硬断，避免死循环
                if token_width > width and len(token) > 1:
                    remaining = token
                    while remaining:
                        take_width = 0
                        take = ""
                        for index, char in enumerate(remaining):
                            char_width = display_width(char)
                            if take_width + char_width > width and take:
                                break
                            take_width += char_width
                            take = remaining[: index + 1]
                        lines.append(take)
                        remaining = remaining[len(take) :]
                    continue
                current.append(token)
                current_width = token_width
                flush_line()
                continue
            # 避头尾：若下一行以禁则标点开头，则把前一字符挤到当前行
            if token in start_forbidden and current_width + token_width <= width + 1:
                current.append(token)
                current_width += token_width
                continue
            # 避头尾：若当前行以禁则标点结尾，把它一起挪到下一行
            if current and current[-1] in end_forbidden:
                carry = current.pop()
                carry_width = display_width(carry)
                flush_line()
                current.append(carry + token)
                current_width = carry_width + token_width
                continue
            if breakable:
                flush_line()
                if token in start_forbidden:
                    # 行首禁则字符尽力不单独断到下一行开头，这里只能保留
                    pass
                current.append(token)
                current_width = token_width
                continue
            flush_line()
            current.append(token)
            current_width = token_width
            continue
        current.append(token)
        current_width += token_width

    flush_line()
    return lines or [""]


def wrap_block(text: str, width: int, initial_indent: str = "", subsequent_indent: str = "") -> str:
    """换行并把结果拼成带缩进的整块文本。"""
    lines = wrap(text, max(width - display_width(initial_indent), 2))
    rendered = []
    for index, line in enumerate(lines):
        prefix = initial_indent if index == 0 else subsequent_indent
        rendered.append(prefix + line)
    return "\n".join(rendered)


# ---------------------------------------------------------------------------
# 表格排版
# ---------------------------------------------------------------------------

_ALIGN_LEFT, _ALIGN_RIGHT, _ALIGN_CENTER = "left", "right", "center"


def render_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    aligns: Optional[Sequence[str]] = None,
    max_width: Optional[int] = None,
) -> str:
    """按显示宽度渲染对齐的 ASCII 表格，并在总宽超限时自动折行压缩。

    Args:
        headers: 表头文本。
        rows: 数据行，元素会被 `str()` 转换。
        aligns: 每列对齐方式（`left` / `right` / `center`），缺省靠左。
        max_width: 表格总显示宽度上限；超出时按比例收缩各列并折行。
    """
    headers = [str(item) for item in headers]
    column_count = len(headers)
    cells = [[str(item) for item in row] for row in rows]
    aligns = list(aligns or []) + [_ALIGN_LEFT] * (column_count - len(aligns or []))

    widths = [
        max([display_width(headers[index])] + [display_width(row[index]) for row in cells] + [1])
        for index in range(column_count)
    ]
    # 每列左右各 1 个空格，外框每列 1 个竖线，再加左右边框
    overhead = 3 * column_count + 1
    if max_width is not None and sum(widths) + overhead > max_width:
        budget = max(max_width - overhead, column_count * 2)
        floor = max(budget // max(column_count, 1), 2)
        # 按比例收缩，保证每列不小于 floor
        total = sum(widths)
        scaled = [max(int(round(width * budget / total)), floor) for width in widths]
        delta = sum(scaled) - budget
        index = 0
        while delta > 0 and column_count:
            column = index % column_count
            if scaled[column] > floor:
                scaled[column] -= 1
                delta -= 1
            index += 1
            if index > column_count * 64:  # 防止极端输入下的长循环
                break
        widths = scaled

    def render_row(values: Sequence[str], align: str) -> str:
        wrapped = [wrap(value, widths[index]) for index, value in enumerate(values)]
        height = max(len(column) for column in wrapped)
        lines = []
        for line_index in range(height):
            parts = []
            for column_index in range(column_count):
                cell = wrapped[column_index][line_index] if line_index < len(wrapped[column_index]) else ""
                parts.append(pad(cell, widths[column_index], aligns[column_index]))
            lines.append("| " + " | ".join(parts) + " |")
        return "\n".join(lines)

    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    header_border = "+" + "+".join("=" * (width + 2) for width in widths) + "+"
    parts = [border, render_row(headers, _ALIGN_LEFT), header_border]
    for row in cells:
        parts.append(render_row(row, _ALIGN_LEFT))
    parts.append(border)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 数字 / 货币 / 百分比
# ---------------------------------------------------------------------------


def _separators() -> Tuple[str, str, int]:
    group_sep = config("fmt.number.group_sep", ",")
    decimal_sep = config("fmt.number.decimal_sep", ".")
    group_size = int(config("fmt.number.group_size", 3) or 3)
    return str(group_sep), str(decimal_sep), max(group_size, 1)


def _group_integer(digits: str, group_sep: str, group_size: int) -> str:
    chunks = []
    while len(digits) > group_size:
        chunks.append(digits[-group_size:])
        digits = digits[:-group_size]
    chunks.append(digits)
    return group_sep.join(reversed(chunks))


def format_number(value: Any, decimals: Optional[int] = None, group: bool = True) -> str:
    """格式化数字，使用当前语言的千分位与小数点符号。

    Args:
        value: 数值或可转换为 float 的对象。
        decimals: 小数位数；为 `None` 时读取 `fmt.number.decimals`（默认 2）。
        group: 是否启用千分位分组。
    """
    if decimals is None:
        decimals = int(config("fmt.number.decimals", 2) or 0)
    group_sep, decimal_sep, group_size = _separators()
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number) or math.isinf(number):
        return str(number)
    sign = "-" if number < 0 else ""
    rounded = f"{abs(number):.{max(decimals, 0)}f}"
    integer_part, _, fraction_part = rounded.partition(".")
    if group:
        integer_part = _group_integer(integer_part, group_sep, group_size)
    result = integer_part if not fraction_part else f"{integer_part}{decimal_sep}{fraction_part}"
    return f"{sign}{result}"


def format_int(value: Any, group: bool = True) -> str:
    """格式化整数。"""
    group_sep, _, group_size = _separators()
    try:
        number = int(value)
    except (TypeError, ValueError):
        return str(value)
    digits = str(abs(number))
    if group:
        digits = _group_integer(digits, group_sep, group_size)
    return f"{'-' if number < 0 else ''}{digits}"


def format_compact_number(value: Any, decimals: Optional[int] = None) -> str:
    """按当前语言的习惯做数量级缩写（中文：万 / 亿；英文：K / M / B）。"""
    units = config("fmt.number.compact_units", []) or []
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    magnitude = abs(number)
    for unit in units:
        threshold = float(unit.get("threshold", 0))
        if magnitude >= threshold:
            scaled = number / threshold
            return f"{format_number(scaled, decimals=decimals if decimals is not None else 2)}{unit.get('suffix', '')}"
    return format_number(number, decimals=decimals if decimals is not None else 0)


def format_percent(value: Any, decimals: Optional[int] = None) -> str:
    """格式化百分比。入参为比率（0.1534 -> `15.34%`）。"""
    pattern = str(config("fmt.percent.pattern", "{value}%"))
    if decimals is None:
        decimals = int(config("fmt.percent.decimals", 2) or 0)
    try:
        number = float(value) * 100
    except (TypeError, ValueError):
        return pattern.format(value=value)
    return pattern.format(value=format_number(number, decimals=decimals))


def format_currency(value: Any, currency: Optional[str] = None, decimals: Optional[int] = None) -> str:
    """格式化货币金额，符号与位置由语言资源决定。"""
    symbols = config("fmt.currency.symbols", {}) or {}
    currency = currency or str(config("fmt.currency.default", "CNY"))
    symbol = symbols.get(currency, currency)
    pattern = str(config("fmt.currency.pattern", "{symbol}{value}"))
    if decimals is None:
        decimals = int(config("fmt.currency.decimals", 2) or 0)
    return pattern.format(symbol=symbol, currency=currency, value=format_number(value, decimals=decimals))


# ---------------------------------------------------------------------------
# 日期与时间
# ---------------------------------------------------------------------------

_WEEKDAY_FIELDS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


def _render_pattern(pattern: str, moment: _dt.datetime) -> str:
    weekday_names = config("fmt.date.weekday_names", []) or []
    weekday = weekday_names[moment.weekday()] if len(weekday_names) == 7 else str(moment.weekday())
    values = {
        "YYYY": f"{moment.year:04d}",
        "YY": f"{moment.year % 100:02d}",
        "MM": f"{moment.month:02d}",
        "M": str(moment.month),
        "DD": f"{moment.day:02d}",
        "D": str(moment.day),
        "HH": f"{moment.hour:02d}",
        "hh": f"{moment.hour % 12 or 12:02d}",
        "mm": f"{moment.minute:02d}",
        "ss": f"{moment.second:02d}",
        "WD": weekday,
    }
    result = pattern
    for token in _DATE_TOKENS:
        result = result.replace("{" + token + "}", values[token])
    return result


def _coerce_datetime(value: Any) -> _dt.datetime:
    if isinstance(value, _dt.datetime):
        return value
    if isinstance(value, _dt.date):
        return _dt.datetime(value.year, value.month, value.day)
    if isinstance(value, (int, float)):
        return _dt.datetime.fromtimestamp(value)
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return _dt.datetime.fromisoformat(text.replace(" ", "T", 1))
    raise TypeError(f"unsupported datetime value: {value!r}")


def format_datetime(value: Any, style: str = "short") -> str:
    """格式化日期时间。`style` 为 `short` / `long`。"""
    pattern = str(config(f"fmt.datetime.{style}", config("fmt.datetime.short", "{YYYY}-{MM}-{DD} {HH}:{mm}:{ss}")))
    return _render_pattern(pattern, _coerce_datetime(value))


def format_date(value: Any, style: str = "long") -> str:
    """格式化日期。中文默认 `2026年9月30日` 形式。"""
    pattern = str(config(f"fmt.date.{style}", config("fmt.date.short", "{YYYY}-{MM}-{DD}")))
    return _render_pattern(pattern, _coerce_datetime(value))


def format_time(value: Any, style: str = "medium") -> str:
    """格式化时间。"""
    pattern = str(config(f"fmt.time.{style}", "{HH}:{mm}:{ss}"))
    return _render_pattern(pattern, _coerce_datetime(value))


def format_duration(seconds: Any, decimals: Optional[int] = None) -> str:
    """格式化时长，单位取自语言资源（中文：`秒`）。"""
    unit = str(config("fmt.units.second", "s"))
    return f"{format_number(seconds, decimals=decimals if decimals is not None else 2)} {unit}".strip()


# ---------------------------------------------------------------------------
# 排序
# ---------------------------------------------------------------------------


def _posix_locale() -> Optional[str]:
    """尝试切到当前语言对应的系统区域；失败返回 `None`。"""
    if get_locale().lower().startswith("zh"):
        candidates = _ZH_COLLATION_LOCALE_CANDIDATES
    else:
        candidates = ("en_US.UTF-8", "en_US.utf8", "en_US", "C")
    for candidate in candidates:
        try:
            _locale.setlocale(_locale.LC_COLLATE, candidate)
            return candidate
        except _locale.Error:
            continue
    return None


def collation_key(text: Any) -> Tuple[int, str]:
    """生成可用于 `sorted(key=...)` 的排序键。

    优先使用系统 `LC_COLLATE` 的 `strxfrm`（Linux 上 `zh_CN.UTF-8` 通常为拼音序）；
    系统区域不可用时降级为「NFKC 归一 + 大小写折叠 + Unicode 码位」序。
    """
    value = "" if text is None else str(text)
    if _posix_locale():
        try:
            return (0, _locale.strxfrm(value))
        except _locale.Error:
            pass
    return (1, unicodedata.normalize("NFKC", value).casefold())


def sort_by_locale(items: Iterable[Any], key=None, reverse: bool = False) -> List[Any]:
    """按当前语言的排序规则排序，返回新列表。"""
    if key is None:
        return sorted(items, key=collation_key, reverse=reverse)
    return sorted(items, key=lambda item: collation_key(key(item)), reverse=reverse)
