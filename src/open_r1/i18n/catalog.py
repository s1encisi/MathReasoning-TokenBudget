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
"""语言资源目录（catalog）的加载、查询与回退。

资源文件位于 `open_r1/i18n/locales/<locale>.json`，键名采用点分命名空间
（例如 `cli.generate.help.model`、`error.configs.dataset_source_required`）。

设计约束：

* 仅依赖标准库，避免在训练/推理环境中引入额外依赖。
* 缺省语言为 `zh_CN`，回退语言为 `en_US`；任何缺失键都会逐级回退，
  最终回退到键名本身，保证不会因为缺翻译而抛异常。
* 目录值可以是字符串，也可以是任意 JSON 结构（数字/布尔/列表），
  后者通过 `config()` 读取，用于驱动日期、数字、货币与排版规则。
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any


# 缺省语言与回退语言。回退语言必须始终存在，否则缺键时无兜底。
DEFAULT_LOCALE = "zh_CN"
FALLBACK_LOCALE = "en_US"

# 语言选择优先级：显式环境变量优先，其次系统环境变量。
LOCALE_ENV_VARS = ("OPENR1_LOCALE", "LC_ALL", "LC_MESSAGES", "LANG")

# 常见的语言标签别名 -> 规范化的资源文件名。
_LOCALE_ALIASES = {
    "zh": "zh_CN",
    "zh_cn": "zh_CN",
    "zh-cn": "zh_CN",
    "zh_hans": "zh_CN",
    "zh_hans_cn": "zh_CN",
    "zh-hans": "zh_CN",
    "zh_sg": "zh_CN",
    "zh_tw": "zh_TW",
    "zh_hk": "zh_HK",
    "en": "en_US",
    "en_us": "en_US",
    "en-us": "en_US",
    "en_gb": "en_US",
    # `C` / `POSIX` 只是「未指定区域」的占位值，不应被当作英语偏好，
    # 否则在 `LANG=C.UTF-8` 的训练集群上会静默退回英文。
    "c": DEFAULT_LOCALE,
    "posix": DEFAULT_LOCALE,
}

_LOCALES_DIR = Path(__file__).parent / "locales"
_LOCK = threading.RLock()

# 扁平化后的目录缓存：locale -> {dotted.key: value}
_CATALOGS: dict[str, dict[str, Any]] = {}
_ACTIVE_LOCALE = DEFAULT_LOCALE


def normalize_locale(code: str | None) -> str:
    """把各种写法的语言标签规范化为资源文件名（如 `zh_CN`）。

    空值、无法识别的标签以及 `C` / `POSIX` 这类「未指定区域」的占位值，
    一律回落到缺省语言 `zh_CN`，确保中文化结果可预期。
    """
    if not code:
        return DEFAULT_LOCALE
    # `zh_CN.UTF-8` / `en_US.UTF-8` 这类带编码后缀的写法需先剥离后缀
    base = str(code).strip().replace("-", "_")
    if "." in base:
        base = base.split(".", 1)[0]
    if "@" in base:  # 例如 `sr_RS@latin`
        base = base.split("@", 1)[0]
    key = base.lower()
    if key in _LOCALE_ALIASES:
        return _LOCALE_ALIASES[key]
    # 形如 `zh_CN` 的规范写法，按分段首字母大写 / 后段大写还原
    if "_" in base:
        language, _, territory = base.partition("_")
        return f"{language.lower()}_{territory.upper()}"
    return DEFAULT_LOCALE


def available_locales() -> list[str]:
    """返回磁盘上实际存在的语言资源列表（已排序）。"""
    if not _LOCALES_DIR.is_dir():
        return []
    return sorted(p.stem for p in _LOCALES_DIR.glob("*.json"))


def _flatten(mapping: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """把嵌套字典压平为点分键。

    * 以 `_` 开头的键视为注释，忽略。
    * 含 `"_opaque": true` 的字典不再展开，整体作为一个值保存
      （例如 `fmt.currency.symbols` 这类「代码 -> 值」映射表）。
    """
    flat: dict[str, Any] = {}
    for key, value in mapping.items():
        if key.startswith("_"):
            continue
        dotted = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and not value.get("_opaque"):
            flat.update(_flatten(value, dotted))
        else:
            if isinstance(value, dict):
                value = {sub_key: sub_value for sub_key, sub_value in value.items() if not sub_key.startswith("_")}
            flat[dotted] = value
    return flat


def load_catalog(locale: str) -> dict[str, Any]:
    """加载（并缓存）指定语言的扁平化目录。文件缺失时返回空字典。"""
    locale = normalize_locale(locale)
    with _LOCK:
        if locale in _CATALOGS:
            return _CATALOGS[locale]
        path = _LOCALES_DIR / f"{locale}.json"
        catalog: dict[str, Any] = {}
        if path.is_file():
            try:
                with path.open("r", encoding="utf-8") as handle:
                    catalog = _flatten(json.load(handle))
            except (OSError, ValueError):
                # 资源文件损坏时降级为空目录，由调用方回退到 en_US
                catalog = {}
        _CATALOGS[locale] = catalog
        return catalog


def resolve_locale() -> str:
    """按环境变量探测当前应使用的语言。"""
    for env_var in LOCALE_ENV_VARS:
        value = os.environ.get(env_var)
        if value:
            return normalize_locale(value)
    return DEFAULT_LOCALE


def get_locale() -> str:
    """返回当前生效的语言代码。"""
    with _LOCK:
        return _ACTIVE_LOCALE


def set_locale(locale: str | None) -> str:
    """切换当前语言并返回切换后的语言代码。"""
    global _ACTIVE_LOCALE
    with _LOCK:
        _ACTIVE_LOCALE = normalize_locale(locale)
        return _ACTIVE_LOCALE


def _lookup(locale: str, key: str) -> Any:
    catalog = load_catalog(locale)
    if key in catalog:
        return catalog[key]
    if locale != FALLBACK_LOCALE:
        fallback = load_catalog(FALLBACK_LOCALE)
        if key in fallback:
            return fallback[key]
    return None


def t(key: str, locale: str | None = None, **params: Any) -> str:
    """按键取本地化文案，支持命名占位符。

    查找顺序：指定/当前语言 -> `en_US` -> 键名本身。
    占位符渲染失败时原样返回未渲染的模板，绝不抛异常。
    """
    target = normalize_locale(locale) if locale else get_locale()
    value = _lookup(target, key)
    if value is None:
        return key
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    if params:
        try:
            text = text.format(**params)
        except (KeyError, IndexError, ValueError):
            # 占位符与传入参数不匹配时保留模板，便于定位翻译错误
            pass
    return text


def config(key: str, default: Any = None, locale: str | None = None) -> Any:
    """读取非字符串型的本地化配置（数字/布尔/列表/映射）。"""
    target = normalize_locale(locale) if locale else get_locale()
    value = _lookup(target, key)
    if value is None:
        return default
    return value


def missing_keys(locale: str, reference: str = FALLBACK_LOCALE) -> list[str]:
    """列出 `locale` 相对 `reference` 缺失的键，用于翻译完整性检查。"""
    reference_catalog = load_catalog(reference)
    target_catalog = load_catalog(locale)
    return sorted(set(reference_catalog) - set(target_catalog))


# 模块首次导入时按环境确定语言；后续可由调用方用 set_locale() 覆盖。
set_locale(resolve_locale())
