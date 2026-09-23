"""scavenger — Discover and curate HuggingFace datasets for training.

v0.70.5: Intelligent dataset discovery that searches HuggingFace Hub
for datasets matching your criteria, filtering by storage, quality,
recency, and relevance.

Usage:
    from hypernix.scavenger import Scavenger, ScavengerCriteria

    sc = Scavenger()
    criteria = ScavengerCriteria(
        keywords=["code", "python", "instruction"],
        max_storage_per_dataset_gb=10.0,
        max_combined_storage_gb=50.0,
        min_entries=1000,
        data_types=["jsonl", "parquet"],
        min_likes=10,
        max_age_days=365,
    )
    datasets = sc.hunt(criteria)

CLI:
    hnx scavenger --keywords code,python --max-storage 10
    hnx scavenger --keywords medical --min-likes 50 --max-age 180
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table


@dataclass
class ScavengerCriteria:
    """Criteria for filtering HuggingFace datasets.

    All fields are optional — unset fields don't filter.
    """
    keywords: list[str] = field(default_factory=list)
    task_types: list[str] = field(default_factory=list)
    data_types: list[str] = field(default_factory=list)
    max_storage_per_dataset_gb: float | None = None
    max_combined_storage_gb: float | None = None
    remaining_device_storage_gb: float | None = None
    min_entries: int | None = None
    max_entries: int | None = None
    min_likes: int | None = None
    min_downloads: int | None = None
    max_age_days: int | None = None
    tokens_per_entry: int = 200
    max_total_tokens: int | None = None

    def matches(self, info: dict[str, Any]) -> tuple[bool, str]:
        """Check if a dataset info dict matches these criteria."""
        if self.keywords:
            text = " ".join([
                info.get("id", ""),
                info.get("description", ""),
                " ".join(info.get("tags", [])),
            ]).lower()
            if not any(kw.lower() in text for kw in self.keywords):
                return False, "no keyword match"

        if self.task_types:
            tags = [t.lower() for t in info.get("tags", [])]
            if not any(tt.lower() in tags for tt in self.task_types):
                return False, "no task type match"

        if self.data_types:
            config_names = [c.get("config_name", "").lower() for c in info.get("configs", [])]
            available_types = set()
            for cn in config_names:
                for dt in ["jsonl", "parquet", "csv", "text", "json"]:
                    if dt in cn:
                        available_types.add(dt)
            if not any(dt.lower() in available_types for dt in self.data_types):
                ds_id = info.get("id", "").lower()
                if not any(dt.lower() in ds_id for dt in self.data_types):
                    return False, "no matching data type"

        size_gb = info.get("size_gb", 0)
        if self.max_storage_per_dataset_gb and size_gb > self.max_storage_per_dataset_gb:
            return False, f"size {size_gb:.1f}GB > max {self.max_storage_per_dataset_gb}GB"

        num_rows = info.get("num_rows", 0)
        if self.min_entries and num_rows < self.min_entries:
            return False, f"entries {num_rows} < min {self.min_entries}"
        if self.max_entries and num_rows > self.max_entries:
            return False, f"entries {num_rows} > max {self.max_entries}"

        likes = info.get("likes", 0)
        if self.min_likes and likes < self.min_likes:
            return False, f"likes {likes} < min {self.min_likes}"

        downloads = info.get("downloads", 0)
        if self.min_downloads and downloads < self.min_downloads:
            return False, f"downloads {downloads} < min {self.min_downloads}"

        if self.max_age_days:
            last_modified = info.get("last_modified")
            if last_modified:
                age = (datetime.now(datetime.UTC) - last_modified).days
                if age > self.max_age_days:
                    return False, f"age {age}d > max {self.max_age_days}d"

        if self.max_total_tokens and num_rows > 0:
            estimated_tokens = num_rows * self.tokens_per_entry
            if estimated_tokens > self.max_total_tokens:
                return False, f"tokens {estimated_tokens} > max {self.max_total_tokens}"

        return True, ""

    def check_storage_budget(self, datasets: list[dict[str, Any]]) -> tuple[bool, float]:
        """Check if selected datasets fit within storage constraints.
        
        Returns True if the total fits within max_combined_storage_gb (if set),
        otherwise returns False. The remaining_device_storage_gb check is also
        performed if set.
        """
        total_gb = sum(d.get("size_gb", 0) for d in datasets)

        if self.max_combined_storage_gb and total_gb > self.max_combined_storage_gb:
            return False, total_gb

        if self.remaining_device_storage_gb:
            import shutil
            free = shutil.disk_usage(".").free / (1024**3)
            if free - total_gb < self.remaining_device_storage_gb:
                return False, total_gb

        return True, total_gb


@dataclass
class Scavenger:
    """HuggingFace dataset discovery engine."""

    console: Console = field(default_factory=Console)
    cache_dir: str | None = None

    def __post_init__(self) -> None:
        if self.cache_dir is None:
            self.cache_dir = os.environ.get("HF_DATASETS_CACHE", "./hf_datasets_cache")

    def _search_hub(self, query: str, limit: int = 100) -> list[dict[str, Any]]:
        """Search HuggingFace Hub for datasets."""
        try:
            from huggingface_hub import HfApi
        except ImportError:
            self.console.print("[yellow]huggingface_hub not installed. Install: pip install huggingface_hub[/]")
            return []

        api = HfApi()
        try:
            # Through hubcompat: `direction` was removed in
            # huggingface-hub 1.0 and passing it raises TypeError at the
            # moment somebody searches, which is the first thing they do.
            from hypernix.system.hubcompat import list_datasets

            results = list_datasets(
                api,
                search=query,
                limit=limit,
                sort="downloads",
                direction=-1,
            )
        except Exception as e:
            self.console.print(f"[yellow]Hub search error: {e}[/]")
            return []

        datasets = []
        for ds in results:
            info = {
                "id": ds.id,
                "description": getattr(ds, "description", "") or "",
                "tags": list(getattr(ds, "tags", []) or []),
                "likes": getattr(ds, "likes", 0) or 0,
                "downloads": getattr(ds, "downloads", 0) or 0,
                "last_modified": getattr(ds, "last_modified", None),
                "size_gb": 0.0,
                "num_rows": 0,
                "configs": [],
            }
            datasets.append(info)

        return datasets

    def _probe_size(self, info: dict[str, Any]) -> dict[str, Any]:
        """Probe actual dataset size and row count."""
        try:
            from datasets import load_dataset_builder

            builder = load_dataset_builder(info["id"], trust_remote_code=False)
            ds_info = builder.info

            if ds_info:
                info["num_rows"] = ds_info.splits.total_num_examples if ds_info.splits else 0
                info["configs"] = [{"config_name": c} for c in (ds_info.config_name or ["default"])]

                total_bytes = 0
                if ds_info.splits:
                    for split_info in ds_info.splits.values():
                        if hasattr(split_info, "num_bytes"):
                            total_bytes += split_info.num_bytes

                if total_bytes > 0:
                    info["size_gb"] = total_bytes / (1024**3)

        except Exception:
            pass

        return info

    def hunt(
        self,
        criteria: ScavengerCriteria,
        search_limit: int = 100,
        verbose: bool = True,
    ) -> list[dict[str, Any]]:
        """Search for datasets matching criteria."""
        if verbose:
            self.console.print("[bold]🔍 Scavenger Hunt[/]")
            self.console.print(f"  Keywords: [cyan]{', '.join(criteria.keywords)}[/]" if criteria.keywords else "  Keywords: (any)")
            if criteria.max_storage_per_dataset_gb:
                self.console.print(f"  Max storage/dataset: [cyan]{criteria.max_storage_per_dataset_gb}GB[/]")

        queries = criteria.keywords if criteria.keywords else [""]

        all_results: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            disable=not verbose,
        ) as progress:
            for query in queries:
                task = progress.add_task(f"Searching for '{query}'...", total=None)
                results = self._search_hub(query, limit=search_limit)
                progress.update(task, completed=True)

                for info in results:
                    if info["id"] in seen_ids:
                        continue
                    seen_ids.add(info["id"])

                    task2 = progress.add_task(f"Probing {info['id']}...", total=None)
                    info = self._probe_size(info)
                    progress.update(task2, completed=True)

                    matches, reason = criteria.matches(info)
                    if matches:
                        info["_score"] = self._score_dataset(info, criteria)
                        all_results.append(info)
                    elif verbose:
                        self.console.print(f"  [dim]Skipped {info['id']}: {reason}[/]")

        all_results.sort(key=lambda x: x.get("_score", 0), reverse=True)

        fits, total_gb = criteria.check_storage_budget(all_results)
        if not fits:
            trimmed = []
            running_gb = 0.0
            for ds in all_results:
                ds_size = ds.get("size_gb", 0)
                if criteria.max_combined_storage_gb:
                    if running_gb + ds_size > criteria.max_combined_storage_gb:
                        continue
                running_gb += ds_size
                trimmed.append(ds)
            all_results = trimmed

            if verbose:
                self.console.print(f"[yellow]Trimmed to fit storage budget: {running_gb:.1f}GB[/]")

        if verbose:
            self.console.print(f"\n[bold]Found {len(all_results)} matching datasets[/]")

        return all_results

    def _score_dataset(self, info: dict[str, Any], criteria: ScavengerCriteria) -> float:
        """Calculate a relevance score for a dataset."""
        score = 0.0

        downloads = info.get("downloads", 0)
        score += min(20.0, downloads / 1000.0)

        likes = info.get("likes", 0)
        score += min(15.0, likes / 5.0)

        last_modified = info.get("last_modified")
        if last_modified:
            age_days = (datetime.now(datetime.UTC) - last_modified).days
            if age_days < 30:
                score += 10.0
            elif age_days < 90:
                score += 5.0
            elif age_days < 180:
                score += 2.0

        num_rows = info.get("num_rows", 0)
        if 1000 <= num_rows <= 1000000:
            score += 10.0
        elif num_rows > 1000000:
            score += 5.0

        if criteria.keywords:
            text = f"{info.get('id', '')} {info.get('description', '')}".lower()
            matches = sum(1 for kw in criteria.keywords if kw.lower() in text)
            score += matches * 5.0

        return score

    def display_results(self, datasets: list[dict[str, Any]]) -> None:
        """Display search results in a formatted table."""
        if not datasets:
            self.console.print("[yellow]No datasets found matching criteria.[/]")
            return

        table = Table(title="Scavenger Hunt Results")
        table.add_column("Rank", style="dim", justify="right")
        table.add_column("Dataset", style="cyan")
        table.add_column("Rows", style="green", justify="right")
        table.add_column("Size", style="yellow", justify="right")
        table.add_column("Likes", style="magenta", justify="right")
        table.add_column("Downloads", style="blue", justify="right")
        table.add_column("Score", style="bold white", justify="right")

        for i, ds in enumerate(datasets[:20], 1):
            size_str = f"{ds.get('size_gb', 0):.1f}GB" if ds.get('size_gb', 0) > 0 else "?"
            rows_str = f"{ds.get('num_rows', 0):,}" if ds.get('num_rows', 0) > 0 else "?"
            table.add_row(
                str(i),
                ds.get("id", "?"),
                rows_str,
                size_str,
                str(ds.get("likes", 0)),
                str(ds.get("downloads", 0)),
                f"{ds.get('_score', 0):.1f}",
            )

        self.console.print(table)

        total_gb = sum(d.get("size_gb", 0) for d in datasets)
        total_rows = sum(d.get("num_rows", 0) for d in datasets)
        self.console.print(f"\n[dim]Total: {len(datasets)} datasets | {total_rows:,} rows | {total_gb:.1f}GB[/]")

    def download_dataset(
        self,
        dataset_id: str,
        subset: str | None = None,
        split: str | None = None,
        streaming: bool = True,
    ) -> Any:
        """Download a dataset from HuggingFace Hub."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library required. Install: pip install datasets") from None

        self.console.print(f"[bold]📥 Downloading {dataset_id}...[/]")

        kwargs = {"trust_remote_code": False}
        if subset:
            kwargs["name"] = subset
        if split:
            kwargs["split"] = split
        if streaming:
            kwargs["streaming"] = True

        ds = load_dataset(dataset_id, **kwargs)
        self.console.print(f"[green]✓ Loaded {dataset_id}[/]")
        return ds


def cli_main(argv: list[str] | None = None) -> int:
    """CLI entry point for hnx scavenger."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Discover HuggingFace datasets for training",
        prog="hnx scavenger",
    )
    parser.add_argument("--keywords", type=str, help="Comma-separated keywords")
    parser.add_argument("--max-storage", type=float, help="Max storage per dataset (GB)")
    parser.add_argument("--max-total", type=float, help="Max combined storage (GB)")
    parser.add_argument("--min-entries", type=int, default=1000)
    parser.add_argument("--min-likes", type=int, default=10)
    parser.add_argument("--max-age", type=int, help="Max age in days")
    parser.add_argument("--data-type", type=str, help="Preferred data type (jsonl, parquet, csv)")
    parser.add_argument("--limit", type=int, default=50, help="Search limit per query")
    parser.add_argument("--download", type=str, help="Download a specific dataset by ID")

    args = parser.parse_args(argv)

    console = Console()

    if args.download:
        sc = Scavenger(console=console)
        try:
            ds = sc.download_dataset(args.download)
            console.print(f"[green]✓ Successfully loaded {args.download}[/]")
            if hasattr(ds, "info") and ds.info:
                console.print(f"  Splits: {list(ds.info.splits.keys()) if ds.info.splits else 'N/A'}")
        except Exception as e:
            console.print(f"[red]Download failed: {e}[/]")
            return 1
        return 0

    keywords = [k.strip() for k in args.keywords.split(",")] if args.keywords else []
    data_types = [args.data_type] if args.data_type else []

    criteria = ScavengerCriteria(
        keywords=keywords,
        max_storage_per_dataset_gb=args.max_storage,
        max_combined_storage_gb=args.max_total,
        min_entries=args.min_entries,
        min_likes=args.min_likes,
        max_age_days=args.max_age,
        data_types=data_types,
    )

    sc = Scavenger(console=console)
    datasets = sc.hunt(criteria, search_limit=args.limit)
    sc.display_results(datasets)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(cli_main())
