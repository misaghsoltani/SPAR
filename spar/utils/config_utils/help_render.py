"""Helpers for rendering Hydra CLI help output."""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import os
from pathlib import Path
import re
import sys
from typing import TYPE_CHECKING, get_args

from omegaconf import DictConfig, ListConfig, OmegaConf

from . import config_schema as schema

if TYPE_CHECKING:
    from collections.abc import Iterable
    from re import Match, Pattern
    from typing import Literal, TypeAlias, TypeGuard

    from typing_extensions import LiteralString


Scalar: TypeAlias = str | int | float | bool | os.PathLike[str] | None
Node: TypeAlias = Scalar | dict[str, "Node"] | list["Node"] | tuple["Node", ...]


def _is_nested_node(value: Node) -> TypeGuard[dict[str, Node] | list[Node] | tuple[Node, ...]]:
    """Return whether a rendered node contains nested values."""
    return isinstance(value, (dict, list, tuple))


def _is_scalar_node(value: Node) -> TypeGuard[Scalar]:
    """Return whether a rendered node can be formatted as a scalar."""
    return not _is_nested_node(value)


def _lower_option(value: str) -> str:
    """Return the case-folded sorting key for one option."""
    return value.lower()


RESET: Literal["\x1b[0m"] = "\x1b[0m"
ANSI_RE: Pattern[str] = re.compile(r"\x1b\[[0-9;]*m")
MAX_WIDTH: int = 120
CONFIG_DIR: Path = Path(__file__).resolve().parents[2] / "configs"


@lru_cache(maxsize=1)
def _color_enabled() -> bool:
    """Return True if the current environment supports ANSI colors.

    Heuristics:
    - Respect NO_COLOR to disable unconditionally.
    - Respect ``CLICOLOR=0`` to disable color and ``CLICOLOR_FORCE`` or ``FORCE_COLOR`` to enable it.
    - Disable on dumb terminals and most CI unless forced.
    - Enable when stdout is a TTY.
    """
    # Explicit opt-out takes precedence
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("CLICOLOR", "").strip() == "0":
        return False
    # Explicit opt-in can override non-tty
    if os.environ.get("CLICOLOR_FORCE", "") == "1":
        return True
    if os.environ.get("FORCE_COLOR", "") in {"1", "2", "3", "true", "TRUE"}:
        return True
    term: str = os.environ.get("TERM", "")
    if term.lower() == "dumb":
        return False
    # Disable color for noninteractive CI output unless the caller forces it.
    if os.environ.get("CI") and not sys.stdout.isatty():
        return False
    return sys.stdout.isatty()


def _paint(text: str, *codes: int) -> str:
    if not codes or not _color_enabled():
        return text
    code_str: str = ";".join(str(code) for code in codes)
    return f"\x1b[{code_str}m{text}{RESET}"


def _strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def _visible_len(text: str) -> int:
    # Account for our escaping of '$' -> '$$' for Hydra string.Template
    cleaned: str = _strip_ansi(text).replace("$$", "$")
    return len(cleaned)


def _ljust(text: str, width: int) -> str:
    return text + " " * max(width - _visible_len(text), 0)


def _wrap_text(text: str, max_visible: int) -> list[str]:
    """Word-wrap an ANSI-colored string to a target visible width.

    Text breaks on whitespace. Long tokens are hard-wrapped.
    """
    if _visible_len(text) <= max_visible:
        return [text]
    parts: list[str] = re.split(r"(\s+)", text)
    lines: list[str] = []
    cur: str = ""
    for part in parts:
        if _visible_len(cur) + _visible_len(part) <= max_visible:
            cur += part
            continue
        if _visible_len(part) > max_visible:
            token: str = part
            while _visible_len(token) > max_visible:
                i: int = 0
                vis: int = 0
                while i < len(token) and vis < max_visible:
                    if token[i] == "\x1b":
                        m: Match[str] | None = ANSI_RE.match(token, i)
                        if m:
                            i = m.end()
                            continue
                    vis += 1
                    i += 1
                segment: str = token[:i]
                if cur:
                    lines.append(cur)
                    cur = ""
                lines.append(segment)
                token = token[i:]
            if token:
                cur = token
        else:
            if cur:
                lines.append(cur)
            cur = part
    if cur:
        lines.append(cur)
    return [ln.rstrip() for ln in lines]


def _escape_template_dollars(s: str) -> str:
    return s.replace("$", "$$")


def _panel(title: str, lines: list[str], color: int, *, padding: int = 1, wrap: bool = True) -> str:
    if not lines:
        lines = [""]

    # Escape dollars first for width computations to stay consistent
    safe_lines: list[str] = [_escape_template_dollars(line) for line in lines]
    # Wrap content to keep total width under MAX_WIDTH (unless explicitly pre-wrapped)
    target_content_width: int = max(10, MAX_WIDTH - 2 - 2 * padding)
    wrapped_lines: list[str] = []
    if wrap:
        for ln in safe_lines:
            wrapped_lines.extend(_wrap_text(ln, target_content_width))
    else:
        wrapped_lines = safe_lines

    content_width: int = max((_visible_len(line) for line in wrapped_lines), default=0)
    title_segment: str = f" {title} "
    inner_width: int = max(content_width + 2 * padding, _visible_len(title_segment))
    inner_width = min(inner_width, MAX_WIDTH - 2)  # keep panel within max width

    pad_total: int = inner_width - _visible_len(title_segment)
    left_pad: int = max(pad_total // 2, 0)
    right_pad: int = max(pad_total - left_pad, 0)

    top_plain: str = f"╭{'─' * left_pad}{title_segment}{'─' * right_pad}╮"
    bottom_plain: LiteralString = f"╰{'─' * inner_width}╯"

    top: str = _paint(top_plain, color, 1)
    bottom: str = _paint(bottom_plain, color, 1)

    left_border: str = _paint("│", color, 1)
    right_border: str = _paint("│", color, 1)
    pad_str: str = " " * padding
    body: list[str] = [
        f"{left_border}{pad_str}{_ljust(line, inner_width - 2 * padding)}{pad_str}{right_border}"
        for line in wrapped_lines
    ]
    return "\n".join([top, *body, bottom])


def _sanitize(value: str | None) -> str | None:
    if value is None:
        return None

    text: str = value.strip()
    if not text or text in {"???", "None", "null"} or "annotation=NoneType" in text:
        return None
    return text


@lru_cache(maxsize=1)
def _group_index() -> dict[str, list[str]]:
    groups: dict[str, set[str]] = defaultdict(set)
    if not CONFIG_DIR.exists():
        return {}

    for yaml_path in CONFIG_DIR.rglob("*.yaml"):
        rel: Path = yaml_path.relative_to(CONFIG_DIR)
        if not rel.parts:
            continue
        if any(part.startswith(".") for part in rel.parts):
            continue
        if rel.parts[0] == "__pycache__":
            continue

        group_parts: tuple[str, ...] = rel.parts[:-1]
        if not group_parts:
            continue
        group: str = "/".join(group_parts)
        groups[group].add(yaml_path.stem)

    return {group: sorted(options) for group, options in sorted(groups.items())}


def _sort_default_first(options: Iterable[str]) -> list[str]:
    opts: list[str] = list(options)
    default_present: bool = any(o == "default" for o in opts)
    others: list[str] = sorted([o for o in opts if o != "default"], key=_lower_option)
    return (["default", *others]) if default_present else others


def _format_options_wrapped(options: Iterable[str], width: int, highlights: set[str] | None = None) -> list[str]:
    """Pack comma-separated options into wrapped lines without trailing commas."""
    highlight_lower: set[str] = {h.lower() for h in highlights} if highlights else set()
    ordered: list[str] = _sort_default_first(options)

    lines: list[str] = []
    cur: str = ""
    for opt in ordered:
        token: str = _paint(opt, 92, 1) if opt.lower() in highlight_lower else opt
        seg: str = token if not cur else f", {token}"
        if not cur:
            cur = seg
            continue
        if _visible_len(cur) + _visible_len(seg) <= width:
            cur += seg
        else:
            lines.append(cur)
            cur = token
    if cur:
        lines.append(cur)
    return lines or [""]


def _coerce_to_node(value: Node | DictConfig | ListConfig | None) -> Node | None:
    if value is None:
        return None

    if isinstance(value, (DictConfig, ListConfig)):
        try:
            plain = OmegaConf.to_container(value, resolve=False)
        except Exception:
            return _format_value(str(value))

        if isinstance(plain, dict):
            return {str(k): _coerce_to_node(v) for k, v in plain.items()}
        if isinstance(plain, list):
            return [_coerce_to_node(v) for v in plain]
        if isinstance(plain, tuple):
            return tuple(_coerce_to_node(v) for v in plain)
        # Scalars
        return plain

    if isinstance(value, dict):
        return {k: _coerce_to_node(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_coerce_to_node(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_coerce_to_node(v) for v in value)

    # Scalars (str/int/float/bool/PathLike) fall through here
    return value


def _schema_groups_for_stage(stage: str, index: dict[str, list[str]]) -> list[str]:
    """Resolve relevant config groups from dataclass schema metadata."""
    ordered_schema_groups: list[str] = []
    for cls in get_args(schema.SPARConfig):
        # Only consider dataclasses
        fields = cls.__dataclass_fields__
        # if not isinstance(fields, dict):
        #     continue
        stage_field = fields.get("stage")
        if stage_field is None:
            continue
        default_val = getattr(stage_field, "default", None)
        if isinstance(default_val, str) and default_val == stage:
            # Preserve field order and include only known top-level groups.
            ordered_schema_groups.extend(name for name in fields if name in index and "/" not in name)
            break

    return ordered_schema_groups


def _stage_defaults_groups(stage_file: Path, index: dict[str, list[str]]) -> list[str]:
    """Resolve relevant config groups from a stage defaults file."""
    ordered: list[str] = []
    seen: set[str] = set()
    if stage_file.exists():
        cfg: DictConfig | ListConfig = OmegaConf.load(stage_file)
        defaults = cfg.get("defaults") if isinstance(cfg, DictConfig) else None
        if isinstance(defaults, list):
            for entry in defaults:
                for key in entry:
                    # Normalize rename syntax: group@name -> group
                    base_key: str = key.split("@", 1)[0]
                    if base_key in index and "/" not in base_key and base_key not in seen:
                        ordered.append(base_key)
                        seen.add(base_key)

    return ordered


@lru_cache(maxsize=64)
def _relevant_groups_for_stage(stage: str | None) -> list[str]:
    """Determine relevant config groups for a stage dynamically from configs, preserving a sensible order.

    Strategy:
    - Build the set of available top-level groups from CONFIG_DIR.
    - If a stage is provided and a corresponding configs/stage/<stage>.yaml exists,
      read its `defaults` list and collect the keys in order of appearance when they
      correspond to available groups.
    - Append any remaining available groups (not referenced in defaults) in alphabetical order.
    - If no stage or no defaults available, return all available groups in alphabetical order.
    """
    index: dict[str, list[str]] = _group_index()
    available_top: list[str] = sorted(g for g in index if "/" not in g)

    s: str | None = stage or None
    if not s:
        return available_top

    try:
        ordered_schema_groups: list[str] = _schema_groups_for_stage(s, index)
        if ordered_schema_groups:
            return ordered_schema_groups
    except Exception:
        # Ignore schema import/introspection errors and fall back to YAML defaults logic
        pass

    stage_file: Path = CONFIG_DIR / "stage" / f"{s}.yaml"
    try:
        ordered: list[str] = _stage_defaults_groups(stage_file, index)
    except Exception:
        # Fall back to all available if anything goes wrong
        return available_top

    # Return groups collected from the defaults list without reordering them.
    if ordered:
        return ordered
    # As a final fallback for known stages, derive core groups from BaseSPARConfig
    try:
        base_fields = getattr(schema.BaseSPARConfig, "__dataclass_fields__", {})
        if not isinstance(base_fields, dict):
            return available_top
        core_ordered: list[str] = [name for name in base_fields if name in index and "/" not in name]
        if core_ordered:
            return core_ordered
    except Exception:
        pass
    # Last resort: show all available groups
    return available_top


def render_usage(
    stage_value: str | None, env_value: str | None, stage_choice: str | None, env_choice: str | None
) -> str:
    """Render a small usage panel for the CLI.

    Args:
        stage_value: The configured default stage value (may be None or placeholder-like strings).
        env_value: The configured default environment value (may be None or placeholder-like strings).
        stage_choice: The resolved/selected stage choice if available.
        env_choice: The resolved/selected environment choice if available.

    Returns:
        A multi-line string containing an ANSI-styled usage panel.
    """
    stage_token: str = _sanitize(stage_choice) or _sanitize(stage_value) or _paint("<required>", 93, 1)
    env_token: str = _sanitize(env_choice) or _sanitize(env_value) or _paint("<required>", 93, 1)

    # Main flow
    main_cmd: str = (
        f"{_paint('spar', 36)} "
        f"{_paint('env', 94)}={_paint(env_token, 92)} "
        f"{_paint('stage', 94)}={_paint(stage_token, 92)} "
        f"[{_paint('KEY', 94)}={_paint('VALUE', 92)} …]"
    )

    # Experiment flow (note the leading '+')
    exp_env: str = _sanitize(env_choice) or _sanitize(env_value) or "env"
    exp_cmd: str = (
        f"{_paint('spar', 36)} "
        f"+{_paint('experiment', 94)}={_paint(f'{exp_env}/<experiment>', 92)} "
        f"[{_paint('KEY', 94)}={_paint('VALUE', 92)} …]"
    )

    lines: list[str] = [main_cmd, _paint("Override nested values with dotted keys (e.g. train.lr=3e-4)", 90), exp_cmd]
    return _panel("Usage", lines, color=33)


def render_context(
    stage_value: str | None, env_value: str | None, stage_choice: str | None, env_choice: str | None
) -> str:
    """Render a panel describing the current configuration context.

    Args:
        stage_value: The configured default stage value (may be None or placeholder-like strings).
        env_value: The configured default environment value (may be None or placeholder-like strings).
        stage_choice: The resolved/selected stage choice if available.
        env_choice: The resolved/selected environment choice if available.

    Returns:
        A multi-line string containing an ANSI-styled context panel.
    """
    stage_clean: str | None = _sanitize(stage_value)
    env_clean: str | None = _sanitize(env_value)
    choice_stage: str | None = _sanitize(stage_choice)
    choice_env: str | None = _sanitize(env_choice)

    env_display: str = choice_env or env_clean or _paint("<required>", 93, 1)
    stage_display: str = choice_stage or stage_clean or _paint("<required>", 93, 1)

    lines: list[str] = [f"Stage group : {_paint(stage_display, 96) if choice_stage else stage_display}"]
    if stage_clean and stage_clean != choice_stage:
        lines.append(f"Stage target: {_paint(stage_clean, 96)}")
    lines.append(f"Environment: {_paint(env_display, 96) if choice_env else env_display}")

    return _panel("Context", lines, color=32)


def render_examples(
    stage_value: str | None, env_value: str | None, stage_choice: str | None, env_choice: str | None
) -> str:
    """Render example commands and help snippets.

    Args:
        stage_value: The configured default stage value (may be None or placeholder-like strings).
        env_value: The configured default environment value (may be None or placeholder-like strings).
        stage_choice: The resolved/selected stage choice if available.
        env_choice: The resolved/selected environment choice if available.

    Returns:
        A multi-line string containing an ANSI-styled panel with example commands.
    """
    stage_clean: str | None = _sanitize(stage_value)
    env_clean: str | None = _sanitize(env_value)
    choice_stage: str | None = _sanitize(stage_choice)
    choice_env: str | None = _sanitize(env_choice)

    env_placeholder: str = choice_env or env_clean or "<env>"
    stage_placeholder: str = choice_stage or stage_clean or "<stage>"

    actions: list[tuple[str, str]] = []

    if choice_stage and choice_env:
        actions.extend((
            ("Run current stage", f"spar env={choice_env} stage={choice_stage}"),
            ("Inspect defaults", f"spar env={choice_env} stage={choice_stage} --info defaults"),
            ("Dump resolved config", f"spar env={choice_env} stage={choice_stage} --cfg=all --resolve"),
        ))
    elif choice_stage:
        actions.extend((
            ("Run stage", f"spar env={env_placeholder} stage={choice_stage}"),
            ("Inspect defaults", f"spar stage={choice_stage} --info defaults"),
        ))
    elif stage_clean:
        actions.extend((
            ("Run stage", f"spar env={env_placeholder} stage={stage_clean}"),
            ("Inspect defaults", f"spar stage={stage_clean} --info defaults"),
        ))
    else:
        actions.extend((
            ("Single run", f"spar env={env_placeholder} stage={stage_placeholder}"),
            ("Explore stages", "spar --info defaults-tree"),
        ))

    # Add experiment examples listed for the selected environment.
    groups: dict[str, list[str]] = _group_index()
    if env_placeholder and groups.get(f"experiment/{env_placeholder}"):
        # Show at most two experiment examples.
        exps: list[str] = groups.get(f"experiment/{env_placeholder}") or []
        if exps:
            example_exp: str = exps[0]
            actions.append(("Run experiment", f"spar +experiment={env_placeholder}/{example_exp}"))
            if len(exps) > 1:
                # Use dedicated command to list experiments only
                if choice_env or env_clean:
                    actions.append(("List experiments", f"spar-experiments env={env_placeholder}"))
                else:
                    actions.append(("List experiments", "spar-experiments env=<env>"))

    actions.append(("Hydra flags", "spar --hydra-help"))

    lines: list[str] = [f"• {_ljust(_paint(f'{label}:', 95), 24)} {_paint(command, 36)}" for label, command in actions]
    return _panel("Example actions", lines, color=35)


def render_group_tree(
    stage_value: str | None, env_value: str | None, stage_choice: str | None, env_choice: str | None
) -> str:
    """Render a compact catalog of available config groups and options.

    Args:
        stage_value: The configured default stage value (may be None or placeholder-like strings).
        env_value: The configured default environment value (may be None or placeholder-like strings).
        stage_choice: The resolved/selected stage choice if available.
        env_choice: The resolved/selected environment choice if available.

    Returns:
        A multi-line string containing an ANSI-styled table of config groups and key options.
    """
    _ = (env_value, env_choice)
    stage_clean: str | None = _sanitize(stage_value)
    choice_stage: str | None = _sanitize(stage_choice)

    groups: dict[str, list[str]] = _group_index()

    # rows of (group_name, options_list)
    rows: list[tuple[str, list[str]]] = []
    stage_key: str | None = choice_stage or stage_clean
    allowed: list[str] = _relevant_groups_for_stage(stage_key)
    for group_name in allowed:
        opts_opt: list[str] | None = groups.get(group_name)
        if not opts_opt:
            continue
        opts: list[str] = opts_opt or []
        rows.append((group_name, _sort_default_first(opts)))

    if not rows:
        return _panel("Config catalog", ["No config groups found."], color=34)

    col0_width: int = max((_visible_len(name) for name, _ in rows), default=5)
    col0_width = max(col0_width, len("Group"))

    header: str = f"{_ljust(_paint('Group', 90), col0_width)} │ {_paint('Key options', 90)}"
    # Compute panel content width for padding=1 to prevent panel-level wrapping
    content_width: int = max(10, MAX_WIDTH - 2 - 2 * 1)
    avail_second: int = max(10, content_width - col0_width - 3)
    divider: LiteralString = f"{'─' * col0_width}─┼─{'─' * avail_second}"

    table_lines: list[str] = [header, divider]
    # Wrap second column to stay within panel content width using option-aware packing
    for name, items in rows:
        label: str = _paint(name, 94, 1) if "/" not in name else _paint(name, 94)
        chunks: list[str] = _format_options_wrapped(items, avail_second, None)
        for i, chunk in enumerate(chunks):
            if i == 0:
                table_lines.append(f"{_ljust(label, col0_width)} │ {chunk}")
            else:
                table_lines.append(f"{_ljust('', col0_width)} │ {chunk}")

    return _panel("Config catalog", table_lines, color=34, wrap=False)


def _format_value(val: bool | int | float | str | os.PathLike[str] | None) -> str:
    """Colorize scalar values for readability."""
    if isinstance(val, bool):
        return _paint(str(val).lower(), 33)
    if val is None:
        return _paint("null", 90)
    if isinstance(val, str) and val.strip() == "???":
        return _paint("<required>", 93, 1)
    if isinstance(val, (int, float)):
        return _paint(str(val), 96)
    return _paint(str(val), 92)


def _format_mapping(mapping: Node | None, *, indent: int = 0) -> list[str]:
    """Render a nested mapping/list in a YAML-like style with colored keys/values."""
    pad: str = "  " * indent
    lines: list[str] = []
    if mapping is None:
        return [pad + _paint("null", 90)]
    if isinstance(mapping, dict):
        if not mapping:
            return [pad + _paint("{}", 90)]

        for k in sorted(mapping.keys(), key=str):
            v = mapping[k]
            key_str: str = _paint(f"{k!s}:", 94)
            if _is_scalar_node(v):
                lines.append(pad + f"{key_str} {_format_value(v)}")
            else:
                lines.append(pad + key_str)
                lines.extend(_format_mapping(v, indent=indent + 1))
        return lines
    if isinstance(mapping, (list, tuple)):
        if not mapping:
            return [pad + _paint("[]", 90)]

        for item in mapping:
            bullet: str = f"{pad}{_paint('-', 90)} "
            if _is_scalar_node(item):
                lines.append(bullet + _format_value(item))
            else:
                lines.append(bullet.rstrip())
                lines.extend(_format_mapping(item, indent=indent + 1))
        return lines
    return [pad + _format_value(mapping)]


def render_resolved_defaults(
    save_dir: Scalar | None,
    debug: bool | None,
    stage_choice: str | None,
    env_block: Node | DictConfig | ListConfig | None,
) -> str:
    """Render a rich panel for the key resolved defaults.

    This intentionally focuses on the most useful, high-signal fields. It does not attempt to dump
    the entire config (which can be obtained with `--cfg=all --resolve`).
    """
    lines: list[str] = []
    lines.extend((
        f"{_paint('save_dir', 94)}: {_format_value(save_dir)}",
        f"{_paint('debug', 94)}: {_format_value(debug)}",
    ))
    if stage_choice:
        lines.append(f"{_paint('stage', 94)}: {_paint(stage_choice, 92)}")
    # Environment block (nested)
    lines.append(_paint("env:", 94))
    # Coerce to builtin containers if needed
    env_obj: Node | None = _coerce_to_node(env_block)
    if not isinstance(env_obj, (dict, list, tuple)):
        env_obj = {}
    lines.extend(_format_mapping(env_obj, indent=1))

    return _panel("Resolved defaults", lines, color=34)


def flag(s: str) -> str:
    """Style a command-line flag."""
    return _paint(s, 96)


def render_common_flags() -> str:
    """Render a concise, curated set of common Hydra flags with styling."""
    lines: list[str] = []
    app_help = "Application's help"
    hydra_help = "Hydra's help"
    version_help = "Show Hydra's version and exit"

    lines.extend((
        f"{flag('--help,-h')} : {_paint(app_help, 90)}",
        f"{flag('--hydra-help')} : {_paint(hydra_help, 90)}",
        f"{flag('--version')} : {_paint(version_help, 90)}",
        f"{flag('--cfg,-c')} : {_paint('Show config instead of running [job|hydra|all]', 90)}",
        f"{flag('--resolve')} : {_paint('With --cfg, resolve interpolations before printing', 90)}",
        f"{flag('--package,-p')} : {_paint('Config package to show', 90)}",
        f"{flag('--run,-r')} : {_paint('Run a job', 90)}",
        f"{flag('--multirun,-m')} : {_paint('Run multiple jobs (launcher/sweeper)', 90)}",
        f"{flag('--shell-completion,-sc')} : {_paint('Install or uninstall shell completion', 90)}",
        "",
    ))
    # Auto-detect user's shell (bash/zsh/fish) and show only relevant commands
    shell: str = os.environ.get("SHELL", "").split("/")[-1].lower()
    bash_install_cmd = 'eval "$(spar -sc install=bash)"'
    bash_uninstall_cmd = 'eval "$(spar -sc uninstall=bash)"'
    bashcompinit_cmd = "autoload -U +X bashcompinit && bashcompinit"
    fish_install_cmd = "spar -sc install=fish | source"
    fish_uninstall_cmd = "spar -sc uninstall=fish | source"
    lines.append(_paint("Shell completion", 95))
    if shell == "bash":
        lines.extend((
            _paint("  Bash - Install:", 90),
            f"  {_paint(bash_install_cmd, 92)}",
            _paint("  Bash - Uninstall:", 90),
            f"  {_paint(bash_uninstall_cmd, 92)}",
        ))
    elif shell == "zsh":
        # Zsh uses bash completion via bashcompinit
        lines.extend((
            _paint("  Zsh - Install:", 90),
            f"  {_paint(bashcompinit_cmd, 92)}",
            f"  {_paint(bash_install_cmd, 92)}",
            _paint("  Zsh - Uninstall:", 90),
            f"  {_paint(bashcompinit_cmd, 92)}",
            f"  {_paint(bash_uninstall_cmd, 92)}",
        ))
    elif shell == "fish":
        lines.extend((
            _paint("  Fish - Install:", 90),
            f"  {_paint(fish_install_cmd, 92)}",
            _paint("  Fish - Uninstall:", 90),
            f"  {_paint(fish_uninstall_cmd, 92)}",
        ))
    else:  # other shells
        lines.extend((
            f"{_paint('  Note:', 90)}{_paint(' For Zsh, run:', 90)}",
            f"  {_paint(bashcompinit_cmd, 92)}",
            f"  {_paint(bash_install_cmd, 92)}",
        ))

    lines.extend(("", f"{_paint('Overrides', 95)}{_paint(': KEY=VALUE (use dotted keys for nested)', 90)}"))

    return _panel("Common flags", lines, color=34)


def render_experiments(env_value: str | None, env_choice: str | None) -> str:
    """Render experiments for the selected env in a separate table."""
    env_clean: str | None = _sanitize(env_value)
    choice_env: str | None = _sanitize(env_choice)
    env_key: str | None = choice_env or env_clean
    groups: dict[str, list[str]] = _group_index()
    if not env_key:
        return _panel("Experiments", ["Set env=<env> to see experiments."], color=34)
    exp_group: str = f"experiment/{env_key}"
    options_opt: list[str] | None = groups.get(exp_group)
    if not options_opt:
        return _panel("Experiments", [f"No experiments under {exp_group} found."], color=34)

    col0: str = _paint("Group", 90)
    col1: str = _paint("Key options", 90)
    col0_width: int = max(_visible_len(exp_group), _visible_len(col0))
    header: str = f"{_ljust(col0, col0_width)} │ {col1}"
    # Match panel's content width (padding=1)
    content_width: int = max(10, MAX_WIDTH - 2 - 2 * 1)
    avail_second: int = max(10, content_width - col0_width - 3)
    divider: LiteralString = f"{'─' * col0_width}─┼─{'─' * avail_second}"
    items: list[str] = _sort_default_first(options_opt or [])
    chunks: list[str] = _format_options_wrapped(items, avail_second, None)

    lines: list[str] = [header, divider]
    label: str = _paint(exp_group, 94)
    for i, chunk in enumerate(chunks):
        if i == 0:
            lines.append(f"{_ljust(label, col0_width)} │ {chunk}")
        else:
            lines.append(f"{_ljust('', col0_width)} │ {chunk}")

    return _panel("Experiments", lines, color=34, wrap=False)


def render_footer() -> str:
    """Render a small footer with links and tips."""
    return "\n".join([
        _paint("Powered by Hydra (https://hydra.cc)", 2),
        _paint("Tip: spar --hydra-help for Hydra-specific flags", 2),
    ])


def render_header(title: str) -> str:
    """Render a header string, adding color when supported."""
    return _paint(title, 1, 36)


def get_group_index() -> dict[str, list[str]]:
    """Expose the cached config group index for other helpers."""
    return {group: list(options) for group, options in _group_index().items()}
