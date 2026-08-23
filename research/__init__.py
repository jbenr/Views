"""Reusable research methods for discovering macro relative-value alpha.

The package is deliberately separate from :mod:`backtest`.  Research asks
whether a stated relationship exists and under which conditions; a backtest
then evaluates a frozen candidate trade.

The three first-class research paths are:

* :class:`DislocationStudy` -- short-horizon conditional dislocations.
* :class:`PairRVStudy` / :class:`PCRelativeValueStudy` -- weighted
  relative-value packages.
* :class:`FairValueStudy` -- declared or searched multi-factor fair value.
"""

from .dislocation import DislocationStudy
from .fair_value import FairValueStudy
from .relative_value import PCRelativeValueStudy, PairRVStudy

__all__ = [
    "DislocationStudy",
    "FairValueStudy",
    "PairRVStudy",
    "PCRelativeValueStudy",
]
