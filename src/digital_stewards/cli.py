"""digital-stewards <command>

    train | eval | scan        the OUT/IN classifier (see classifier.py)
    dashboard add|serve|list   Steward Desk: recording + live steward log, FIA-style penalties
"""
import sys


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "dashboard":
        from .dashboard.server import main as dash
        return dash(argv[1:])
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__); return
    from .classifier import main as clf
    clf(argv)
