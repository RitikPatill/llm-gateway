"""Rich terminal dashboard for LLM Gateway.

Run with:
    python -m llm_gateway.dashboard
    python -m llm_gateway.dashboard --url http://localhost:8000
"""

import argparse
import sys

import httpx
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def _build_stats_panel(data: dict) -> Panel:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", justify="right")
    table.add_column(justify="left")

    table.add_row("Total requests", str(data.get("total_requests", 0)))
    table.add_row("Cache hit rate", f"{data.get('hit_rate', 0.0) * 100:.1f} %")
    table.add_row("p50 latency", f"{data.get('p50_latency_ms', 0.0):.1f} ms")
    table.add_row("p95 latency", f"{data.get('p95_latency_ms', 0.0):.1f} ms")
    table.add_row("~Tokens saved", str(data.get("tokens_saved", 0)))

    return Panel(table, title="[bold green]Stats[/bold green]", border_style="green")


def _build_log_panel(data: dict) -> Panel:
    log = data.get("recent_log", [])
    table = Table(show_header=True, header_style="bold magenta", expand=True)
    table.add_column("Time", style="dim", width=10)
    table.add_column("Model")
    table.add_column("Hit", justify="center", width=5)
    table.add_column("Latency (ms)", justify="right", width=14)

    for entry in reversed(log):
        hit_text = Text("✓", style="green") if entry.get("hit") else Text("✗", style="red")
        table.add_row(
            entry.get("ts", ""),
            entry.get("model", ""),
            hit_text,
            f"{entry.get('latency_ms', 0):.1f}",
        )

    return Panel(table, title="[bold blue]Request Log (latest first)[/bold blue]", border_style="blue")


def _build_layout(data: dict) -> Layout:
    layout = Layout()
    layout.split_row(
        Layout(name="left", ratio=1),
        Layout(name="right", ratio=2),
    )
    layout["left"].update(_build_stats_panel(data))
    layout["right"].update(_build_log_panel(data))
    return layout


def main(gateway_url: str = "http://localhost:8000") -> None:
    """Poll /metrics every second and render a Rich Live layout."""
    metrics_url = gateway_url.rstrip("/") + "/metrics"

    empty: dict = {
        "total_requests": 0,
        "cache_hits": 0,
        "hit_rate": 0.0,
        "p50_latency_ms": 0.0,
        "p95_latency_ms": 0.0,
        "tokens_saved": 0,
        "recent_log": [],
    }

    with Live(_build_layout(empty), refresh_per_second=1, screen=True) as live:
        try:
            while True:
                try:
                    resp = httpx.get(metrics_url, timeout=2.0)
                    resp.raise_for_status()
                    data = resp.json()
                except Exception as exc:
                    data = dict(empty)
                    data["recent_log"] = [
                        {
                            "ts": "--:--:--",
                            "model": f"error: {exc}",
                            "hit": False,
                            "latency_ms": 0,
                        }
                    ]
                live.update(_build_layout(data))
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Gateway Rich dashboard")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Gateway base URL (default: http://localhost:8000)",
    )
    args = parser.parse_args()
    main(gateway_url=args.url)
