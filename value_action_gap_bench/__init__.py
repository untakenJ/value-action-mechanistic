"""Value-Action Gap benchmark (ValueActionLens, arXiv:2501.15463)."""

__all__ = ["main"]


def main(*args, **kwargs):
    from value_action_gap_bench.run import main as _main

    return _main(*args, **kwargs)
