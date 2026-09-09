# `gather` — crawling a site into a corpus

`hnx gather` walks a site and writes what it finds as training data.
It is the scraping half of HyperNix's data tooling: `scavenger` pulls
datasets that already exist on Hugging Face, `gather` makes one out of
pages that do not.

```bash
hnx gather -W https://example.org -Q 2 -T 4 -p 1.5 -f jsonl -o ./corpus
```

`hypernix gather` and `hnx gather` are the same command. The subcommand
is optional — `hnx gather -W …` and `hnx gather crawl -W …` do the same
thing — so the short form works in a shell and the explicit form reads
better in a script.

## Flags

| Flag | Long | Meaning |
| --- | --- | --- |
| `-W` | `--site` | The site to crawl. |
| `-L` | `--list` | Several sites, comma-separated. |
| `-T` | `--threads` | Fetches in parallel. Default 1. |
| `-Q` | `--depth` | How many links deep. `0` is the seed page alone. Default 1. |
| `-p` | `--pause` | Seconds between requests **to the same host**. Default 1.0. |
| `-f` | `--format` | Output format. See below. Default `jsonl`. |
| `-o` | `--output` | Where to write. Defaults to `~/hypernix-gather`. |
| `-O` | `--header` | A provenance line written into the output; with `-C`, the archive's name instead. |
| `-C` | `--compress` | Compress the result. Needs one of `--xz`, `--7z`, `--zip`, `--gz`. |
| | | The uncompressed files are **kept** — see below. |
| `-U` | `--upload` | Upload to a GitHub or Hugging Face repo. Needs `--yes`. |
| `-u` | `--requests` | (with `probe`) How many requests to measure a host with. |

And the ones without a letter: `--no-robots`, `--any-host`,
`--include` / `--exclude` (regexes matched against the URL),
`--max-pages`, `--timeout`, `--json`, `-q` / `--quiet`, `--yes`.

## Formats

| `-f` | What you get |
| --- | --- |
| `html` | One file per page, as fetched. |
| `html-full` | Every page merged into a single document. Single-site only. |
| `html-full-wimages` | The same, with images inlined as data URIs. |
| `text` | Tags stripped, one file per page. |
| `jsonl` | One JSON object per page: url, title, text, links, status. |
| `parquet` | The same columns, as Parquet. Needs `pyarrow` or `pandas`. |
| `js` | The JavaScript each page references, **saved**. |

`html-full` merges pages into one document, which has no meaning across
several sites, so combining it with `-L` is refused rather than
producing a file that silently interleaves two corpora.

`-f js` saves JavaScript. It does not run any: there is no interpreter
in the module and no browser engine imported, and a test reads the
source on every CI run to keep it that way. A page that only renders
under JavaScript will come back thin, and that is the trade — a scraper
that executes what it downloads is running a stranger's code on your
training box.

## It asks, and it waits

Three defaults you can turn off but should not:

- **robots.txt is honoured.** `--no-robots` exists for crawling your own
  site, which is the only place it belongs.
- **There is a pause between requests.** Per host, so `-L` crawling four
  sites is not four times as slow as one.
- **A host's own `Crawl-delay` wins.** If `robots.txt` asks for 5
  seconds and you passed `-p 1`, you wait 5.

The rate limiter claims its slot inside the lock rather than after it.
Claimed after, two of `-T 8`'s threads both read the clock, both decide
now is fine, and the delay you configured is not the delay the server
sees.

Before a long crawl, ask what a host will tolerate:

```bash
hnx gather probe -W https://example.org -u 8
```

That sends a few requests, watches for `429` and `Retry-After`, and
prints the delay it would suggest. It does not crawl.

## Where it writes

Only under `-o`. A URL path is chosen by whoever wrote the page and
becomes a file name, so `safe_output_path` resolves the result and
requires it to be under the output root — a link to
`/../../.ssh/authorized_keys` lands in the corpus as a mangled file name
and nowhere else.

Without `-o`, output goes to `hypernix-gather` in your home directory
(`/home/<user>/hypernix-gather`, or `C:\Users\<user>\hypernix-gather`
on Windows) — a directory rather than `$HOME` itself, because `-f html`
on a hundred-page site writes a hundred files and they should not land
next to your dotfiles.

## In scripts

```bash
hnx gather -W https://example.org -Q 2 -f jsonl -o ./corpus --json -q > run.json
case $? in
  0) echo "crawled $(jq .pages run.json) pages" ;;
  1) echo "could not start" >&2 ;;
  2) echo "fetched nothing" >&2 ;;
  3) echo "wrote output, but some pages failed" >&2 ;;
esac
```

`--json` puts the machine-readable result on **stdout** and leaves
progress on stderr, so a pipeline reads one and a person watching reads
the other.

| Exit | Meaning |
| --- | --- |
| `0` | Wrote output, every page fetched. |
| `1` | Could not start — bad flags, or nothing to crawl. |
| `2` | Finished, fetched nothing. |
| `3` | Wrote output, but some pages failed. |

`3` rather than `0` is deliberate: a crawl that got eight pages of ten
is a corpus with holes in it, and a script should be able to see that
without parsing the JSON.

`hnx gather formats --json` lists the formats and compressors a build
supports, so a script can check before it commits to one.

## `-C` keeps the originals

Compressing writes the archive beside the files and leaves the files
there. A failed archive plus deleted originals is a lost crawl, and a
crawl is the expensive part.

So `-C` is not a way to save disk on its own — delete the originals once
you have checked the archive. The run says how many it kept, and
`--json` lists them under `retained`, separately from `outputs`:

```json
{
  "outputs":  ["/home/me/hypernix-gather/corp.zip"],
  "retained": ["/home/me/hypernix-gather/example.org/index.html", "..."]
}
```

`-U` uploads `outputs`, so an upload sends the archive and not the
loose files.

## Uploading

```bash
hnx gather -W https://example.org -f parquet -U hf://me/my-corpus --yes
hnx gather -W https://example.org -f jsonl  -U gh://me/my-corpus --yes
```

`-U` never uploads without `--yes`. Publishing a scrape is a decision,
not something to inherit from a flag left in your shell history.

## From Python

```python
from hypernix.data import gather

plan = gather.CrawlPlan(
    sites=["https://example.org"], depth=2, threads=4,
    delay=1.0, fmt="jsonl", output="./corpus",
)
result = gather.crawl(plan)
gather.write_output(result, plan)

print(len(result.pages), "pages,", len(result.skipped), "skipped")
```

`gather.fetch(url)` is the single-page path, and it is what
`hypernix.interfaces.websearch.fetch_web_page` calls — one fetcher with
robots, a rate limit, a content-type check and a size ceiling, rather
than two that drift.

## See also

- [Scavenger](Scavenger.md) — datasets that already exist.
- [CLI](CLI.md) — every subcommand.
