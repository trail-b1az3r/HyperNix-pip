# Toaster — `hypernix.toaster`

Where `hypernix.pans` is for a whole dataset pass, the toaster is for
formatting you want to apply **per line** while reading. Four tiers,
modelled on real toasters:

| Tier | Class | What it does |
|---|---|---|
| 1 | `TwoSliceToaster` | Pairs adjacent lines as `(prompt, response)` Q/A entries. |
| 2 | `FourSliceToaster` | Batches four lines into a 2-turn chat (user/assistant/user/assistant). |
| 3 | `ConveyorToaster` | Streaming mode — yields a formatted entry for every line as it arrives; useful when feeding from a live log. |
| 4 | `ToasterOven` | Whole-document formatting — reads the entire file, wraps each blank-line-delimited document with a header/footer, yields one string per document. |

All four accept `source: Path | str | Iterable[str]` and implement
`__iter__`, so they can be swapped by name via the factory.

## `TwoSliceToaster(source, prompt_tag="Q: ", response_tag="A: ")`

Two lines in, one entry out. `line_a` is the prompt, `line_b` the
response; output is `"{prompt_tag}{a}\n{response_tag}{b}"` per pair.
Blank lines are skipped before pairing; a trailing unpaired line is
silently dropped (no partial pair is yielded).

```python
from hypernix.toaster import TwoSliceToaster
list(TwoSliceToaster(["What is 2+2?", "4", "Capital of France?", "Paris"]))
# ["Q: What is 2+2?\nA: 4", "Q: Capital of France?\nA: Paris"]
```

## `FourSliceToaster(source, user_tag="<USER>", assistant_tag="<ASSISTANT>")`

Four lines in, one 2-turn chat entry out:
`"{user_tag} {l0}\n{assistant_tag} {l1}\n{user_tag} {l2}\n{assistant_tag} {l3}"`.
Same blank-line skip / trailing-partial-drop behavior as `TwoSliceToaster`.

## `ConveyorToaster(source, template="<TEXT>{line}</TEXT>")`

Streaming per-line formatter — wraps every non-empty line in
`template.format(line=line)`. Any `str.format`-compatible template with
a `{line}` placeholder works.

## `ToasterOven(source, header="<DOCUMENT>", footer="</DOCUMENT>")`

Whole-document formatting. Lines are grouped into "documents" separated
by blank lines in the source; each document is bracketed:
`"{header}\n{doc_lines joined by \n}\n{footer}"`. A trailing document
without a final blank line is still yielded.

## Factory

```python
from hypernix.toaster import toaster
t = toaster("two-slice-toaster", "qa_pairs.txt")
```

`toaster(tier, source, **kw)` — `tier` is case-insensitive with `_`
normalized to `-`. Valid tiers (also in `TIERS: dict[str, type]`):
`"two-slice-toaster"`, `"four-slice-toaster"`, `"conveyor-toaster"`,
`"toaster-oven"`.

Note: unlike `hypernix.pans.pick_pan`, this factory does **not** wrap
`TypeError` on bad kwargs with a friendlier message, and an unknown
`tier` raises a plain `KeyError` rather than a `ValueError` listing
valid tiers.

### Required modules

Standard library only — `dataclasses`, `pathlib`, `collections.abc`.


> **Canonical source:** `src/hypernix/data/toaster.py`. The `hypernix.toaster` import spelling used in examples is the compatibility-facing API; the source path above is the implementation to check when verifying behavior. The paired-line tiers skip blanks and silently discard an incomplete final pair/group; validate source formatting if every record must be retained.

---

## See also

- [Pans](Pans.md) — whole-dataset-pass equivalent (`Sink.pour()`-compatible)
- `hypernix.food_processor` — bulk chopping/slicing of a single large document
- `hypernix.blender` — mixing multiple sources instead of formatting a single one
