from __future__ import annotations

import sys

from spar.utils.config_utils.help_render import render_experiments


def main(argv: list[str] | None = None) -> None:
    """Print experiments table for an env.

    Usage:
      spar-experiments env=<env>

    If env is omitted or unknown, a hint panel will be shown.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    env_val: str | None = None
    # Parse key=value arguments.
    for arg in argv:
        if "=" in arg:
            k, v = arg.split("=", 1)
            if k == "env":
                env_val = v

    out: str = render_experiments(env_val, env_val)
    print(out)


if __name__ == "__main__":
    main()
