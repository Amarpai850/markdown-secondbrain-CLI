"""ANSI terminal colorization and keyword highlighting."""

from __future__ import annotations

import os
import sys

_VTP_ENABLED = False


def _enable_windows_vtp() -> bool:
    """Enable Virtual Terminal Processing on Windows console without 3rd-party dependencies."""
    global _VTP_ENABLED
    if _VTP_ENABLED or sys.platform != "win32":
        return True

    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        # STD_OUTPUT_HANDLE = -11
        h_out = kernel32.GetStdHandle(-11)
        if h_out == 0 or h_out == -1:
            return False

        mode = ctypes.c_ulong()
        if not kernel32.GetConsoleMode(h_out, ctypes.byref(mode)):
            return False

        # ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        new_mode = mode.value | 0x0004
        if kernel32.SetConsoleMode(h_out, new_mode):
            _VTP_ENABLED = True
            return True
    except Exception:
        pass
    return False


def colors_enabled() -> bool:
    """Check if ANSI color output should be enabled.

    Honors NO_COLOR, checks stdout isatty, and enables Windows VTP.
    """
    if "NO_COLOR" in os.environ:
        return False
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False
    if sys.platform == "win32":
        return _enable_windows_vtp()
    return True


# ANSI escape sequences
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
MAGENTA = "\033[35m"
RED = "\033[31m"
BOLD_YELLOW = "\033[1;33m"


def style(text: str, color_code: str) -> str:
    """Apply ANSI style if coloring is supported, else return plain text."""
    if not colors_enabled():
        return text
    return f"{color_code}{text}{RESET}"


def cyan(text: str) -> str:
    return style(text, CYAN)


def yellow(text: str) -> str:
    return style(text, YELLOW)


def green(text: str) -> str:
    return style(text, GREEN)


def magenta(text: str) -> str:
    return style(text, MAGENTA)


def dim(text: str) -> str:
    return style(text, DIM)


def bold(text: str) -> str:
    return style(text, BOLD)


def bold_yellow(text: str) -> str:
    return style(text, BOLD_YELLOW)


def red(text: str) -> str:
    return style(text, RED)


def highlight_snippet(snippet: str, spans: list[tuple[int, int]]) -> str:
    """Inject ANSI bold yellow highlighting into snippet at specified character spans."""
    if not colors_enabled() or not spans:
        return snippet

    # Merge overlapping or adjacent spans
    sorted_spans = sorted(spans, key=lambda s: s[0])
    merged: list[tuple[int, int]] = []
    for s, e in sorted_spans:
        if not merged:
            merged.append((s, e))
        else:
            prev_s, prev_e = merged[-1]
            if s <= prev_e:
                merged[-1] = (prev_s, max(prev_e, e))
            else:
                merged.append((s, e))

    result_parts: list[str] = []
    cursor = 0
    for s, e in merged:
        if s > cursor:
            result_parts.append(snippet[cursor:s])
        match_text = snippet[s:e]
        result_parts.append(f"{BOLD_YELLOW}{match_text}{RESET}")
        cursor = e

    if cursor < len(snippet):
        result_parts.append(snippet[cursor:])

    return "".join(result_parts)
