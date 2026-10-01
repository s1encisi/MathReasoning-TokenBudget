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
"""`open_r1` 的国际化（i18n）支持。

典型用法：

```python
from open_r1.i18n import t

raise ValueError(t("error.evaluation.unknown_benchmark", benchmark=name))
```

语言选择优先级：`OPENR1_LOCALE` > `LC_ALL` > `LC_MESSAGES` > `LANG` > `zh_CN`。
资源文件位于 `open_r1/i18n/locales/<locale>.json`。
"""

from .argparse_help import LocaleAwareArgumentParser, LocaleAwareHelpFormatter, make_parser
from .catalog import (
    DEFAULT_LOCALE,
    FALLBACK_LOCALE,
    available_locales,
    config,
    get_locale,
    missing_keys,
    normalize_locale,
    resolve_locale,
    set_locale,
    t,
)
from .formatting import (
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
    pad,
    render_table,
    sort_by_locale,
    wrap,
    wrap_block,
)


__all__ = [
    "DEFAULT_LOCALE",
    "FALLBACK_LOCALE",
    "LocaleAwareArgumentParser",
    "LocaleAwareHelpFormatter",
    "available_locales",
    "collation_key",
    "config",
    "display_width",
    "format_compact_number",
    "format_currency",
    "format_date",
    "format_datetime",
    "format_duration",
    "format_int",
    "format_number",
    "format_percent",
    "format_time",
    "get_locale",
    "make_parser",
    "missing_keys",
    "normalize_locale",
    "pad",
    "render_table",
    "resolve_locale",
    "set_locale",
    "sort_by_locale",
    "t",
    "wrap",
    "wrap_block",
]
