import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
import io
import re
from contextlib import redirect_stdout, redirect_stderr
import requests
import traceback
import logging

import pandas as pd

from copy import deepcopy
from config import config
import optimize
from optimize import DEFAULT_GRID, apply_overrides
from utils import (
    get_run_dir,
    make_run_dirs,
    get_market_week_start,
    get_market_week_end,
)

# Set up comprehensive logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('walkforward_debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class Tee(io.TextIOBase):
    """Write output to multiple streams."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()


class FilteredTee(io.TextIOBase):
    """Write output to console and selectively to a file."""

    def __init__(self, console, log_file, patterns: list[str], verbose: bool = False):
        self.console = console
        self.log_file = log_file
        self.patterns = [re.compile(p) for p in patterns]
        self.verbose = verbose
        self.buffer = ""

    def write(self, data):
        self.console.write(data)
        self.console.flush()
        if self.verbose:
            self.log_file.write(data)
            return len(data)

        for ch in data:
            self.buffer += ch
            if ch == "\n":
                line = self.buffer
                if any(p.search(line) for p in self.patterns):
                    self.log_file.write(line)
                self.buffer = ""
        return len(data)

    def flush(self):
        self.console.flush()
        if self.buffer:
            if self.verbose or any(p.search(self.buffer) for p in self.patterns):
                self.log_file.write(self.buffer)
            self.buffer = ""
        self.log_file.flush()


def format_td(td: timedelta) -> str:
    """Return HH:MM:SS string for a timedelta."""
    total = int(td.total_seconds())
    hours, rem = divmod(total, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}"


def send_discord_webhook(message: str, webhook_url: str = "https://discord.com/api/webhooks/1302792016345825350/2ukwCzoeN_WFzvfOiBvJoUfMDRIVr5PwsPbsFQsWCdHU5gNw9k2o5RCQdB_Xu9-QPe6l"):
    """Send a message to Discord webhook."""
    try:
        payload = {
            "content": message,
            "username": "Trading System Bot",
            "avatar_url": "https://cdn.discordapp.com/avatars/123456789/abcdef.png"
        }
        response = requests.post(webhook_url, json=payload)
        if response.status_code == 204:
            print("✅ Discord webhook sent successfully")
        else:
            print(f"⚠️ Discord webhook failed with status code: {response.status_code}")
    except Exception as e:
        print(f"⚠️ Discord webhook error: {e}")


def validate_file_exists(file_path: Path, description: str) -> bool:
    """Validate that a file exists and log the result."""
    exists = file_path.exists()
    if exists:
        logger.info(f"✅ {description}: {file_path} exists")
        # Check file size
        size = file_path.stat().st_size
        logger.info(f"   File size: {size} bytes")
        if size == 0:
            logger.warning(f"⚠️ {description}: {file_path} is empty!")
            return False
    else:
        logger.error(f"❌ {description}: {file_path} does not exist!")
    return exists


def parse_args():
    parser = argparse.ArgumentParser(description="Walk-forward backtest loop")
    parser.add_argument(
        "--train_window_months",
        type=int,
        default=config.get("walkforward.train_window_months"),
    )
    parser.add_argument(
        "--window_weeks",
        type=int,
        default=config.get("walkforward.window_weeks"),
    )
    parser.add_argument(
        "--stepback_weeks",
        type=int,
        default=config.get("walkforward.stepback_weeks", 12),
    )
    parser.add_argument("--run", type=str, help="Base directory for runs")
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Grid search configs before each iteration",
    )
    parser.add_argument(
        "--max_configs",
        type=int,
        default=None,
        help="Limit number of parameter sets tested when optimizing",
    )
    parser.add_argument(
        "--grid",
        choices=["fast", "mid", "full"],
        default="full",
        help="Which parameter grid to use when optimizing",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Write all output to output.txt instead of filtering",
    )
    parser.add_argument(
        "--target_trades_per_week",
        type=float,
        default=None,
        help="Desired trades per week for ranker",
    )
    parser.add_argument(
        "--min_trades_per_week",
        type=float,
        default=None,
        help="Minimum trades per week for ranker",
    )
    parser.add_argument(
        "--max_trades_per_week",
        type=float,
        default=None,
        help="Maximum trades per week for ranker",
    )
    return parser.parse_args()


def load_raw_index():
    """Return the first and last timestamps in the raw CSV.

    Only the first and last lines are read to avoid loading the entire file.
    """
    input_dir = config.get("data.input_dir")
    path = next(Path(input_dir).glob("*.csv"))

    def parse_line(line: str) -> pd.Timestamp:
        date, time, *_ = line.strip().split(",")
        return pd.to_datetime(f"{date} {time}")

    with open(path, "r") as f:
        # Skip headers or any line that does not parse as a timestamp
        first_line = f.readline()
        while first_line:
            tokens = first_line.strip().split(",")
            if (
                len(tokens) >= 2
                and tokens[0].lower() == "date"
                and tokens[1].lower() == "time"
            ):
                first_line = f.readline()
                continue
            try:
                parse_line(first_line)
                break
            except Exception:
                first_line = f.readline()
        if not first_line:
            raise ValueError("No valid timestamp lines found")

    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        end = f.tell() - 1
        pos = end
        while pos >= 0:
            f.seek(pos)
            char = f.read(1)
            if char == b"\n" and pos != end:
                break
            pos -= 1
        f.seek(pos + 1)
        last_line = f.readline().decode()

    tz = config.get("market.timezone")
    start_ts = parse_line(first_line)
    end_ts = parse_line(last_line)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize(tz)
    else:
        start_ts = start_ts.tz_convert(tz)
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize(tz)
    else:
        end_ts = end_ts.tz_convert(tz)
    return start_ts, end_ts


def validate_dataframe(df: pd.DataFrame, name: str) -> bool:
    """Validate a dataframe and log its properties."""
    if df is None or df.empty:
        logger.error(f"❌ {name}: DataFrame is None or empty!")
        return False
    
    logger.info(f"✅ {name}: DataFrame loaded successfully")
    logger.info(f"   Shape: {df.shape}")
    logger.info(f"   Columns: {list(df.columns)}")
    logger.info(f"   Index type: {type(df.index)}")
    logger.info(f"   Date range: {df.index.min()} to {df.index.max()}")
    
    # Check for common issues
    if df.isnull().any().any():
        null_counts = df.isnull().sum()
        logger.warning(f"⚠️ {name}: Found null values: {null_counts[null_counts > 0].to_dict()}")
    
    if df.duplicated().any():
        logger.warning(f"⚠️ {name}: Found {df.duplicated().sum()} duplicate rows")
    
    return True


def run_script_with_debug(script: str, args: list[str], env: dict, step_name: str) -> None:
    """Run a helper script with comprehensive debugging."""
    logger.info(f"🚀 Starting {step_name}: {script}")
    logger.info(f"   Arguments: {args}")
    logger.info(f"   Environment keys: {list(env.keys())}")
    
    run_dir = Path(env.get("RUN_ID", ""))
    log_dir = run_dir / "logs" if run_dir else None
    log_file = None
    
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{Path(script).stem}_debug.log"
        log_file = open(log_path, "w")
        logger.info(f"   Debug log: {log_path}")

    # Check if script exists
    if not Path(script).exists():
        logger.error(f"❌ Script not found: {script}")
        raise FileNotFoundError(f"Script not found: {script}")

    proc = subprocess.Popen(
        [sys.executable, script] + args,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert proc.stdout is not None
    
    output_lines = []
    try:
        for line in proc.stdout:
            output_lines.append(line.strip())
            sys.stdout.write(line)
            sys.stdout.flush()
            if log_file:
                log_file.write(line)
    except Exception as e:
        logger.error(f"❌ Error reading output from {script}: {e}")
        raise
    finally:
        if log_file:
            log_file.flush()
            log_file.close()
    
    proc.wait()
    
    if proc.returncode != 0:
        logger.error(f"❌ {step_name} failed with return code: {proc.returncode}")
        logger.error(f"   Last 10 output lines:")
        for line in output_lines[-10:]:
            logger.error(f"   {line}")
        raise subprocess.CalledProcessError(
            proc.returncode, [sys.executable, script] + args
        )
    
    logger.info(f"✅ {step_name} completed successfully")
    
    # Validate expected output files
    if step_name == "Prepare":
        validate_file_exists(run_dir / "data" / "prepared.csv", "Prepared data")
    elif step_name == "Label":
        validate_file_exists(run_dir / "data" / "labeled.csv", "Labeled data")
    elif step_name == "Train Base":
        validate_file_exists(run_dir / "data" / "signals.csv", "Signals data")
        validate_file_exists(run_dir / "data" / "probs_base.csv", "Base probabilities")
    elif step_name == "Train Meta":
        validate_file_exists(run_dir / "data" / "probs_meta.csv", "Meta probabilities")
    elif step_name == "SLTP":
        validate_file_exists(run_dir / "data" / "signals.csv", "Signals with SLTP")
    elif step_name == "Train Ranker":
        validate_file_exists(run_dir / "data" / "signals.csv", "Final signals")
    elif step_name == "Simulate":
        validate_file_exists(run_dir / "artifacts" / "sim_metrics.json", "Simulation metrics")
        validate_file_exists(run_dir / "artifacts" / "sim_trades.csv", "Simulation trades")


def run_script(script: str, args: list[str], env: dict) -> None:
    """Run a helper script and stream its output."""
    # Map script names to step names for debugging
    step_names = {
        "prepare.py": "Prepare",
        "label.py": "Label", 
        "train_base.py": "Train Base",
        "train_meta.py": "Train Meta",
        "sltp.py": "SLTP",
        "train_ranker.py": "Train Ranker",
        "simulate.py": "Simulate"
    }
    
    step_name = step_names.get(script, script)
    run_script_with_debug(script, args, env, step_name)


def append_history(base: Path, end: pd.Timestamp, metrics: dict, best: dict | None = None) -> None:
    hist_path = base / "history.csv"
    exists = hist_path.exists()
    
    logger.info(f"📊 Appending metrics to history: {hist_path}")
    logger.info(f"   Metrics: {metrics}")
    if best:
        logger.info(f"   Best config metrics: {best}")
    
    with open(hist_path, "a", newline="") as f:
        writer = csv.writer(f)
        header = [
            "end_date",
            "win_rate",
            "avg_rr",
            "total_trades",
            "profit_factor",
            "trades_per_wk",
        ]
        row = [
            end.strftime("%Y-%m-%d"),
            metrics.get("win_rate", 0),
            metrics.get("avg_rr", 0),
            metrics.get("total_trades", metrics.get("trades_per_wk", 0)),
            metrics.get("profit_factor", 0),
            metrics.get("trades_per_wk", 0),
        ]
        if best is not None:
            header.extend(["opt_win_rate", "opt_profit_factor", "opt_trades_per_wk"])
            row.extend([
                best.get("win_rate", 0),
                best.get("profit_factor", 0),
                best.get("trades_per_wk", 0),
            ])
        if not exists:
            writer.writerow(header)
        writer.writerow(row)
    
    logger.info(f"✅ History updated successfully")


def validate_simulation_results(run_dir: Path) -> bool:
    """Validate simulation results and log detailed analysis."""
    logger.info(f"🔍 Validating simulation results in: {run_dir}")
    
    # Check metrics file
    metrics_file = run_dir / "artifacts" / "sim_metrics.json"
    if not validate_file_exists(metrics_file, "Simulation metrics"):
        return False
    
    try:
        with open(metrics_file) as f:
            metrics = json.load(f)
        
        logger.info(f"📊 Simulation metrics:")
        logger.info(f"   Win rate: {metrics.get('win_rate', 0):.3f}")
        logger.info(f"   Profit factor: {metrics.get('profit_factor', 0):.3f}")
        logger.info(f"   Total trades: {metrics.get('total_trades', 0)}")
        logger.info(f"   Avg RR: {metrics.get('avg_rr', 0):.3f}")
        logger.info(f"   Trades per week: {metrics.get('trades_per_wk', 0):.1f}")
        
        # Validate metrics
        win_rate = metrics.get('win_rate', 0)
        if win_rate > 1.0:
            logger.error(f"❌ Invalid win rate: {win_rate} (should be 0-1)")
            return False
        
        if win_rate == 0 and metrics.get('total_trades', 0) > 0:
            logger.warning(f"⚠️ Zero win rate with {metrics.get('total_trades', 0)} trades - possible issue")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error reading metrics: {e}")
        return False


def main(args=None):
    if args is None:
        args = parse_args()
    
    logger.info("🚀 Starting walkforward analysis with debugging enabled")
    logger.info(f"   Arguments: {vars(args)}")
    
    if args.optimize:
        logger.info(f"Optimization enabled: searching configs ({args.grid})")
    
    base_run = Path(args.run) if args.run else Path(get_run_dir())
    base_run.mkdir(parents=True, exist_ok=True)
    logger.info(f"   Base run directory: {base_run}")

    summary_rows: list[tuple[pd.Timestamp, dict]] = []
    start_time = datetime.now()

    def print_eta(completed: int) -> None:
        elapsed = datetime.now() - start_time
        avg = elapsed / completed if completed else timedelta(0)
        remaining = avg * (args.stepback_weeks - completed)
        print(
            f"Progress: {completed}/{args.stepback_weeks} iterations, "
            f"elapsed {format_td(elapsed)}, est. remaining {format_td(remaining)}"
        )

    # Load and validate raw data
    logger.info("📊 Loading raw data index...")
    start_raw, end_raw = load_raw_index()
    logger.info(f"   Raw data range: {start_raw} to {end_raw}")
    
    tz = config.get("market.timezone")
    if start_raw.tzinfo is None:
        start_raw = start_raw.tz_localize(tz)
    else:
        start_raw = start_raw.tz_convert(tz)
    if end_raw.tzinfo is None:
        end_raw = end_raw.tz_localize(tz)
    else:
        end_raw = end_raw.tz_convert(tz)

    # Find the last complete week in the data
    last_week_start = get_market_week_start(end_raw)
    
    # If we're not at the end of a complete week, go back one week
    if end_raw - last_week_start < pd.Timedelta(days=5):
        last_week_start -= pd.DateOffset(weeks=1)
    
    logger.info(f"   Last complete week starts: {last_week_start}")
    
    # Calculate the earliest week we can start from
    earliest_possible_start = last_week_start - pd.DateOffset(weeks=args.stepback_weeks - 1)
    earliest_possible_start = earliest_possible_start - pd.DateOffset(months=args.train_window_months)
    
    if earliest_possible_start < start_raw:
        logger.warning(f"⚠️ Earliest possible start {earliest_possible_start} is before available data {start_raw}")
        logger.info("Adjusting stepback_weeks to fit available data...")
        
        # Calculate how many weeks we can actually go back
        available_weeks = (end_raw - start_raw).days // 7
        needed_weeks = args.train_window_months * 4 + args.window_weeks  # Rough estimate
        max_stepback = max(1, available_weeks - needed_weeks)
        
        if max_stepback < args.stepback_weeks:
            logger.info(f"Reducing stepback_weeks from {args.stepback_weeks} to {max_stepback}")
            args.stepback_weeks = max_stepback

    # Start from the most recent week and work backwards
    for offset in range(args.stepback_weeks):
        iter_start = datetime.now()
        logger.info(f"\n🔄 Processing iteration {offset + 1}/{args.stepback_weeks}")
        
        # Calculate the week we're processing (0 = most recent, 1 = second most recent, etc.)
        current_week_start = last_week_start - pd.DateOffset(weeks=offset)
        
        # Training period: from current_week_start back by train_window_months
        train_end = current_week_start
        train_start = train_end - pd.DateOffset(months=args.train_window_months)
        
        # Simulation period: from current_week_start forward by window_weeks
        sim_end = get_market_week_end(
            current_week_start + pd.DateOffset(weeks=args.window_weeks - 1)
        )
        
        logger.info(f"   Current week: {current_week_start.strftime('%Y-%m-%d')}")
        logger.info(f"   Training period: {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}")
        logger.info(f"   Simulation period: {train_end.strftime('%Y-%m-%d')} to {sim_end.strftime('%Y-%m-%d')}")
        
        # Verify all dates are within available data
        if train_start < start_raw:
            logger.warning(f"⚠️ Skipping week starting {current_week_start.strftime('%Y-%m-%d')}: training start {train_start.strftime('%Y-%m-%d')} is before available data")
            print_eta(offset + 1)
            continue
            
        if sim_end > end_raw:
            logger.warning(f"⚠️ Skipping week starting {current_week_start.strftime('%Y-%m-%d')}: simulation end {sim_end.strftime('%Y-%m-%d')} exceeds available data")
            print_eta(offset + 1)
            continue

        print(f"\nProcessing week {offset + 1}/{args.stepback_weeks}: {current_week_start.strftime('%Y-%m-%d')}")
        print(f"  Training: {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}")
        print(f"  Simulation: {train_end.strftime('%Y-%m-%d')} to {sim_end.strftime('%Y-%m-%d')}")

        run_dir = base_run / train_end.strftime("%Y%m%d")
        logger.info(f"   Run directory: {run_dir}")
        
        env = os.environ.copy()
        env["RUN_ID"] = str(run_dir)
        make_run_dirs(str(run_dir))

        train_end_str = train_end.strftime("%Y-%m-%d")
        start_str = train_start.strftime("%Y-%m-%d")
        end_str = sim_end.strftime("%Y-%m-%d")

        best_conf = {}
        best_metrics = None
        base_cfg = deepcopy(config._config)
        
        if args.optimize:
            logger.info("🔧 Running optimization...")
            opt_dir = run_dir / "opt_tmp"
            opt_dir.mkdir(parents=True, exist_ok=True)
            cache_dir = run_dir / "opt_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            optimize.base_config = deepcopy(config._config)
            grid = optimize.GRID_MAP.get(args.grid, DEFAULT_GRID)
            best_conf, best_metrics = optimize.search_configs(
                grid,
                start_str,
                train_end_str,
                end_str,
                str(opt_dir),
                args.max_configs,
                str(cache_dir),
            )
            with open(run_dir / "best_config.json", "w") as f:
                json.dump({"best_conf": best_conf, "best_metrics": best_metrics}, f)
            config._config = apply_overrides(base_cfg, best_conf)
            logger.info("✅ Optimization completed")

        # Run the pipeline steps with debugging
        try:
            logger.info("📊 Step 1: Preparing data...")
        run_script(
            "prepare.py",
            [
                "--start_date",
                start_str,
                "--train_end_date",
                train_end_str,
                "--end_date",
                end_str,
                "--input_dir",
                config.get("data.input_dir"),
            ],
            env,
        )
            
            logger.info("🏷️ Step 2: Labeling data...")
        run_script(
            "label.py",
            [
                "--run",
                str(run_dir),
                "--start_date",
                start_str,
                "--train_end_date",
                train_end_str,
                "--end_date",
                end_str,
            ],
            env,
        )
            
        model_args = [
            "--run",
            str(run_dir),
            "--start_date",
            start_str,
            "--train_end_date",
            train_end_str,
            "--end_date",
            end_str,
        ]
            
            logger.info("🤖 Step 3: Training base model...")
        run_script("train_base.py", model_args, env)
            
            logger.info("🤖 Step 4: Training meta model...")
        run_script("train_meta.py", model_args, env)
            
            logger.info("🎯 Step 5: Setting SL/TP...")
        run_script("sltp.py", model_args, env)
            
        ranker_args = []
        if args.target_trades_per_week is not None:
            ranker_args += [
                "--target_trades_per_week",
                str(args.target_trades_per_week),
            ]
        if args.min_trades_per_week is not None:
            ranker_args += [
                "--min_trades_per_week",
                str(args.min_trades_per_week),
            ]
        if args.max_trades_per_week is not None:
            ranker_args += [
                "--max_trades_per_week",
                str(args.max_trades_per_week),
            ]
            
            logger.info("🏆 Step 6: Training ranker...")
        run_script("train_ranker.py", model_args + ranker_args, env)

        # Verify that the simulation end date is within the prepared data range
        prepared_path = run_dir / "data" / "prepared.csv"
        if prepared_path.exists():
                logger.info("📊 Validating prepared data range...")
            df_dates = pd.read_csv(prepared_path, index_col=0, parse_dates=True, usecols=[0])
            tz = config.get("market.timezone")
            idx = df_dates.index
            # Ensure idx is a DatetimeIndex before checking timezone
            if not isinstance(idx, pd.DatetimeIndex):
                idx = pd.to_datetime(idx, utc=True)
            if idx.tz is None:
                idx = idx.tz_localize(tz)
            else:
                idx = idx.tz_convert(tz)
            max_date = idx.max()
            end_dt = pd.Timestamp(end_str, tz=tz)
            if end_dt > max_date:
                    logger.warning(f"⚠️ Adjusting simulation end from {end_str} to {max_date.strftime('%Y-%m-%d')} due to limited data")
                end_dt = max_date
                end_str = end_dt.strftime("%Y-%m-%d")

            logger.info("🎮 Step 7: Running simulation...")
        sim_args = ["--run", str(run_dir), "--start_date", train_end_str, "--end_date", end_str]
        if args.window_weeks == 1:
            sim_args += ["--test_frac", "1.0"]
        run_script("simulate.py", sim_args, env)

            # Validate simulation results
            if not validate_simulation_results(run_dir):
                logger.error(f"❌ Simulation validation failed for {run_dir}")
                continue

        metrics_file = run_dir / "artifacts" / "sim_metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
            append_history(base_run, train_end, metrics, best_metrics)
            summary_rows.append((train_end, metrics))
                logger.info(f"✅ Results added to summary: {metrics}")

        except Exception as e:
            logger.error(f"❌ Error in iteration {offset + 1}: {e}")
            logger.error(f"   Traceback: {traceback.format_exc()}")
            continue

        print_eta(offset + 1)

        if args.optimize:
            config._config = base_cfg

    if summary_rows:
        logger.info("📊 Generating final summary...")
        print("\nSummary of results:")
        summary_message = "**🎯 Walkforward Test Results**\n\n"
        
        for end, metr in sorted(summary_rows):
            trades = metr.get("total_trades", metr.get("trades_per_wk", 0))
            win_rate = metr.get('win_rate', 0) * 100
            avg_rr = metr.get('avg_rr', 0)
            
            # Format the line for console output
            result_line = (
                f"{end.strftime('%Y-%m-%d')}: Win Rate {win_rate:.1f}%, "
                f"Avg RR {avg_rr:.2f}, Trades {int(trades)}"
            )
            print(result_line)
            
            # Add to Discord message with emoji indicators
            if win_rate >= 60:
                emoji = "🟢"  # Green for high win rate
            elif win_rate >= 50:
                emoji = "🟡"  # Yellow for moderate win rate
            else:
                emoji = "🔴"  # Red for low win rate
                
            summary_message += f"{emoji} **{end.strftime('%Y-%m-%d')}**: Win Rate {win_rate:.1f}%, Avg RR {avg_rr:.2f}, Trades {int(trades)}\n"
        
        # Calculate overall statistics
        if summary_rows:
            all_win_rates = [metr.get('win_rate', 0) * 100 for _, metr in summary_rows]
            all_trades = [metr.get("total_trades", metr.get("trades_per_wk", 0)) for _, metr in summary_rows]
            all_rr = [metr.get('avg_rr', 0) for _, metr in summary_rows]
            
            avg_win_rate = sum(all_win_rates) / len(all_win_rates)
            total_trades = sum(all_trades)
            avg_rr_overall = sum(all_rr) / len(all_rr)
            weeks_above_50 = sum(1 for wr in all_win_rates if wr >= 50)
            weeks_above_60 = sum(1 for wr in all_win_rates if wr >= 60)
            
            logger.info(f"📊 Overall statistics:")
            logger.info(f"   Average win rate: {avg_win_rate:.1f}%")
            logger.info(f"   Total trades: {total_trades}")
            logger.info(f"   Average RR: {avg_rr_overall:.2f}")
            logger.info(f"   Weeks ≥50% win rate: {weeks_above_50}/{len(summary_rows)}")
            logger.info(f"   Weeks ≥60% win rate: {weeks_above_60}/{len(summary_rows)}")
            
            summary_message += f"\n**📊 Overall Statistics:**\n"
            summary_message += f"• Average Win Rate: {avg_win_rate:.1f}%\n"
            summary_message += f"• Total Trades: {total_trades}\n"
            summary_message += f"• Average RR: {avg_rr_overall:.2f}\n"
            summary_message += f"• Weeks ≥50% Win Rate: {weeks_above_50}/{len(summary_rows)}\n"
            summary_message += f"• Weeks ≥60% Win Rate: {weeks_above_60}/{len(summary_rows)}\n"
            
            # Add strategy info
            summary_message += f"\n**🎯 Strategy**: Hybrid Success + Volume Control\n"
            summary_message += f"**📅 Test Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            
            # Send to Discord
            send_discord_webhook(summary_message)
    else:
        logger.warning("⚠️ No results to summarize!")

    logger.info("🎉 Walkforward analysis completed!")


if __name__ == "__main__":
    args = parse_args()
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    patterns = [
        r"\[prepare\]",
        r"\[label\]",
        "Test Simulation Results",
        "Full Simulation Results",
        "Total Trades",
        "Win Rate",
        "Profit Factor",
        "Average RR",
    ]
    with open(output_dir / "output.txt", "w") as f:
        tee = FilteredTee(sys.stdout, f, patterns, verbose=args.verbose)
        with redirect_stdout(tee), redirect_stderr(tee):
            main(args)
