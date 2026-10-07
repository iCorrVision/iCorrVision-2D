import argparse
import sys
from pathlib import Path
import time
from contextlib import contextmanager
import threading

from icorrvision.config.loader import build_run
from icorrvision.engine.factory import build_engine
from icorrvision.io.archive.writer import write_icorr_archive, write_directory_output

EXIT_SUCCESS = 0
EXIT_ABORTED = 1
EXIT_CONFIG_ERROR = 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="icorr")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_p = subparsers.add_parser(
        "run", help="Execute a correlation run from a TOML profile"
    )

    def _expanded_path(value: str) -> Path:
        return Path(value).expanduser()

    run_p.add_argument("--config", required=True, type=_expanded_path)
    run_p.add_argument("--image-dir", type=_expanded_path, default=None)
    run_p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="key.path=value",
        help="Override a TOML value, e.g. --set 'correlation.search.method=\"pyramid\"' (strings keep their TOML quotes)",
    )
    output_group = run_p.add_mutually_exclusive_group(required=True)
    output_group.add_argument("--output-dir", type=_expanded_path)
    output_group.add_argument("--output-archive", type=_expanded_path)

    return parser


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        built = build_run(args.config, args.set, args.image_dir)
        engine = build_engine(
            built.correlation_config, built.setup_output, built.frame_loader
        )
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    with _background_status() as progress_cb:
        result = engine.run(progress_cb=progress_cb)

    if args.output_dir is not None:
        write_directory_output(result, built, args.output_dir)
    else:
        write_icorr_archive(
            result,
            built.setup_output,
            built.correlation_config,
            built.frame_loader,
            built.resolved_session.reference_frame_name,
            built.resolved_session.deformed_frames,
            args.output_archive,
        )

    return EXIT_ABORTED if result.aborted else EXIT_SUCCESS


@contextmanager
def _background_status(task_name: str = "Correlation"):
    stop_event = threading.Event()
    progress = {"done": 0, "total": 0}

    def status_loop():
        symbols = [
            "'_( o _ o)__",
            "'_( o _ o)__",
            "-_( o _ o)__",
            "-_( o _ o)__",
            " _-( o _ o)__",
            "  __( o _ o)-_",
            "  __( o _ o)_-",
            "  __( o _ o)_-",
            "  __( o _ o)_'",
            "  __( o _ o)_'",
            "  __( o _ o)_-",
            " __( o _ o)-_",
            "_-( o _ o)__",
            "-_( o _ o)__",
        ]
        i = 0
        start = time.perf_counter()
        while not stop_event.is_set():
            elapsed = time.perf_counter() - start

            hours = int(elapsed // 3600)
            minutes = int((elapsed // 60) % 60)
            second = int(elapsed % 60)
            hundreths = int((elapsed % 1) * 100)

            sym = symbols[i % len(symbols)]

            done, total = progress["done"], progress["total"]
            frame_str = f"[{done}/{total}]" if total > 0 else ""

            print(
                f"\r{task_name}{frame_str}...{sym} | time elapsed:{hours}:{minutes}:{second}:{hundreths}",
                end="",
                flush=True,
            )
            i += 1
            time.sleep(0.1)

    thread = threading.Thread(target=status_loop, daemon=True)
    thread.start()

    def update_progress(done: int, total: int):
        progress["done"] = done
        progress["total"] = total

    try:
        yield update_progress
    finally:
        stop_event.set()
        thread.join()
        print(f"\r{task_name} finished ( o u o)7")


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "run":
        return _cmd_run(args)
    parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
