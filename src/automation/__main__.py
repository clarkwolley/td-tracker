"""Entry point for ``python -m src.automation``."""

import argparse
import logging


def main():
    parser = argparse.ArgumentParser(description="TD Tracker weekly pipeline")
    parser.add_argument(
        "--retrain", action="store_true", help="Force model retrain"
    )
    parser.add_argument(
        "--no-notify", action="store_true", help="Skip Telegram notifications"
    )
    parser.add_argument(
        "--top-n", type=int, default=20, help="Number of predictions (default: 20)"
    )
    parser.add_argument(
        "--backtest", nargs=2, type=int, metavar=("SEASON", "WEEK"),
        help="Backtest a historical week instead of running live",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # Import here to avoid slow startup for --help
    from src.automation.pipeline import run_weekly, backtest_week

    if args.backtest:
        backtest_week(
            *args.backtest,
            top_n=args.top_n,
            notify=not args.no_notify,
        )
    else:
        run_weekly(
            top_n=args.top_n,
            force_retrain=args.retrain,
            notify=not args.no_notify,
        )


if __name__ == "__main__":
    main()
