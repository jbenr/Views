"""Live signal monitoring for the book.

A registry of promoted ("live") setups, an append-only audit ledger of
analysis runs, and a Dash dashboard to view and refresh them.

    python -m dashboard.registry --promote book.curve.tens_10s30s
    python -m dashboard.app

See dashboard/README.md for the full workflow.
"""
