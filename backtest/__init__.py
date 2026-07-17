from .trades import TradeDef, TradeType, Position, ClosedPosition
from .signals import (
    SignalConfig, SignalPipeline, BooleanSignalPipeline,
    generate_signals, generate_boolean_actions,
    profit_target, half_drift_residual,
)
from .sizing import DV01Map, size_dv01_neutral, size_beta_weighted, size_custom
from .engine import Engine, BacktestConfig, BacktestResult
from .metrics import compute_metrics, drawdown_series, trade_log
from .report import print_summary, summary_table, equity_curve_pd, trade_log_pd
from .portfolio import SpreadBook, BookConfig, SpreadDef
from .sweep import ParameterSweep, SweepConfig
from .lab import (
    ParamGrid, sweep_strategy, fast_scan, predict_scan, signal_matrix,
    MetricStore, gate_scan, add_gate_lift, add_predict_lift,
    gate_variant_count, parse_gate, gate_allow_mask, stateful_exit_scan,
)
