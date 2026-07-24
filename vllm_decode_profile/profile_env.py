"""Environment parsing shared by online and offline profiling helpers."""

from __future__ import annotations

import os


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise RuntimeError(f"{name} must be a boolean, got {value!r}")


def env_non_negative_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {value!r}") from exc
    if parsed < 0:
        raise RuntimeError(f"{name} must be non-negative, got {parsed}")
    return parsed


def env_non_negative_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number, got {value!r}") from exc
    if parsed < 0:
        raise RuntimeError(f"{name} must be non-negative, got {parsed}")
    return parsed


def env_optional_positive_int(
    name: str,
    fallback_name: str | None = None,
) -> int | None:
    value = os.environ.get(name)
    selected_name = name
    if value is None and fallback_name is not None:
        value = os.environ.get(fallback_name)
        selected_name = fallback_name
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise RuntimeError(
            f"{selected_name} must be an integer, got {value!r}"
        ) from exc
    if parsed == 0:
        return None
    if parsed < 0:
        raise RuntimeError(
            f"{selected_name} must be non-negative, got {parsed}"
        )
    return parsed
