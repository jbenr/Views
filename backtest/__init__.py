from .engine import (
    TradeDef, TradeType, Position, ClosedPosition,
    SignalConfig, SignalPipeline, BooleanSignalPipeline,
    generate_signals, generate_boolean_actions,
    profit_target, half_drift_residual,
    DV01Map, size_dv01_neutral, size_beta_weighted, size_custom,
    Engine, BacktestConfig, BacktestResult,
    compute_metrics, drawdown_series, trade_log,
    print_summary, summary_table, equity_curve_pd, trade_log_pd,
    SpreadBook, BookConfig, SpreadDef,
)
from .lab import (
    ParamGrid, sweep_strategy, fast_scan, predict_scan, signal_matrix,
    MetricStore, gate_scan, add_gate_lift, add_predict_lift,
    gate_variant_count, parse_gate, gate_allow_mask, gate_percentile_rank,
    stateful_exit_scan,
    neighbor_ic_stats,
)
from .validation import (
    PBOResult,
    annualized_sharpe,
    deflated_sharpe_ratio,
    effective_number_of_trials,
    event_overlap_diagnostics,
    expected_maximum_sharpe,
    probability_of_backtest_overfitting,
    probabilistic_sharpe_ratio,
    return_moments,
)
