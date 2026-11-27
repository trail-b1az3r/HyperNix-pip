# Ranges — labelers for `mediocre_fridge.collect_responses_from`

A **range** is a callable ``(prompt, response) -> "GOOD" | "BAD"`` you
plug straight into
:func:`hypernix.mediocre_fridge.collect_responses_from` as
``label_rule=...``.  Three concrete implementations sit on a ladder of
sophistication:

| Module | Sophistication | Deps | When to reach for it |
|---|---|---|---|
| `new_range.NewRange` | low | none | Quick triage; you don't have a teacher model. |
| `old_range.OldRange` | mid | none | You want explainability and reference / keyword scoring. |
| `industrial_range.IndustrialRange` | high | a judge oven (local or remote) | You have a stronger model that can act as the judge. |

All three share the call signature, so you can swap them in `label_rule=`
without touching the rest of your pipeline.

## new_range — first-fail rubric

Bag of zero-dependency heuristics evaluated in order. Any rule that
returns True yields ``BAD``; otherwise ``GOOD``. The defaults catch
empty responses, refusal phrases, math prompts answered without
digits, and long single-character runs.

```python
from hypernix import new_range

r = new_range.new_range()
r("How many planets are in the solar system?", "lots of them")  # "BAD"
r.last_failed_rule                                                # "math_lacks_digit"
r("Hello?", "aaaaaaaaaaaa")                                       # "BAD" (repetition)
r("Capital of France?", "Paris")                                  # "GOOD"
```

Bring your own rules:

```python
def starts_with_no(prompt: str, response: str) -> bool:
    return response.lower().startswith("no")

r = new_range.new_range(rules=[starts_with_no])
```

`is_refusal` knows the ways current models decline ("I'm unable to", "I
must decline", "I apologize, but"…) and reads curly apostrophes as
straight ones, so "I can’t" counts (0.72.8).

Built-in rule set: `is_empty`, `is_too_short`, `is_too_long`,
`is_refusal`, `math_lacks_digit`, `is_repetition`. The default
selection is `(is_empty, is_refusal, math_lacks_digit, is_repetition)`.

## old_range — scored rubric with explainability

Each rule returns ``(score, reason)`` where ``score`` is a float in
``[0, 1]`` **or** ``None`` (= "no opinion, skip me"). The aggregate is
the weighted mean over rules that did vote. Rules whose score is
exactly 0 short-circuit to ``BAD`` by default ("definitely BAD per
this rule" shouldn't be drowned out).

```python
from hypernix import old_range

r = old_range.old_range(
    references={
        "What is the capital of France?": "Paris is the capital of France.",
    },
    keywords={
        "List a primary color.": ["red", "blue", "yellow"],
    },
    threshold=0.5,
)

label = r("List a primary color.", "Sure: red, blue, and yellow.")
# -> "GOOD"

label, agg, breakdown = r.label_with_breakdown(
    "How many planets are in the solar system?",
    "lots of them",
)
# label = "BAD"
# breakdown = [
#     ("length_score",     1.0,  "len ok (12)"),
#     ("refusal_score",    1.0,  "no refusal"),
#     ("math_digit_score", 0.0,  "math prompt without digit"),
#     ("overlap_score",    None, "no reference"),
#     ("keyword_score",    None, "no keywords"),
#     ("repetition_score", 1.0,  "no long runs"),
# ]
```

### Built-in scored rules

| Rule | What it scores | Returns None when |
|---|---|---|
| `length_score` | empty / too-short / too-long penalty | — (always opinion) |
| `refusal_score` | 0.0 if a refusal phrase appears | — |
| `math_digit_score` | math-y prompts: 1.0 with digits, 0.0 without | non-math prompt |
| `overlap_score` | content-word Jaccard with reference (stopwords filtered) | no reference |
| `keyword_score` | fraction of required keywords present | no keywords |
| `repetition_score` | 0.0 on long single-char runs, 1.0 otherwise | — |

### Tuning knobs

```python
old_range.OldRange(
    rules=[...],                 # custom rule list (default = all six)
    weights={"length_score": 2.0, "overlap_score": 3.0},   # default 1.0 each
    threshold=0.5,               # GOOD / BAD cutoff on aggregate
    fatal_rules=("*",),          # default: any rule scoring 0.0 forces BAD.
                                 # Set to specific names for selective fatality
                                 # or to () to disable entirely.
    references={prompt: ref},    # for overlap_score
    keywords={prompt: [...]},    # for keyword_score
    soft_max_chars=1024,         # length_score linear-decay start
    hard_max_chars=4096,         # length_score score=0
    min_chars=1,
)
```

## industrial_range — LLM-as-judge

Wraps anything with a `.complete(prompt, max_new_tokens, temperature, stop)`
method (every `CodeOven` does — and so does any caller-supplied wrapper
around an HTTP API). Asks the judge for a one-word verdict, parses the
reply, caches results.

```python
from hypernix import industrial_range, old_oven

teacher = old_oven.preheat(repo_id="qwen3.5-9b", device="cuda")
judge = industrial_range.industrial_range(judge=teacher)

judge("Capital of France?", "Paris")     # -> "GOOD"
judge("Capital of France?", "London")    # -> "BAD"
```

### Pairwise comparison

For preference-pair datasets:

```python
verdict = judge.compare(
    "Sort [3, 1, 2] in Python.",
    "sorted([3, 1, 2])",                 # response A
    "[3, 1, 2].sort()",                  # response B
)
# verdict in {"A", "B", "T"}  (T = tie)
```

### Customizing the judge prompt

```python
judge = industrial_range.IndustrialRange(
    judge=teacher,
    rubric=(
        "You are evaluating a Python code snippet. Answer GOOD if the "
        "code is syntactically correct and accomplishes the prompt; "
        "BAD if it has syntax errors, imports nothing it uses, or "
        "doesn't address the prompt."
    ),
    template=(
        "{rubric}\n\n"
        "TASK: {prompt}\n"
        "CODE:\n{response}\n"
        "VERDICT:"
    ),
    max_new_tokens=8,
    temperature=0.0,
    stop=("\n",),
    use_cache=True,
)
```

### Batched calls

```python
labels = judge.label_batch([("Q1", "R1"), ("Q2", "R2"), ...])
verdicts = judge.compare_batch([("Q", "A1", "B1"), ...])
```

The cache is per-instance and stores both pointwise and pairwise
queries. Repeated calls with the same `(prompt, response)` return the
cached label without billing the judge again.

### Robust parsing

The verdict parser handles:

- a clean one-word reply (`"GOOD"`, `"BAD"`, `"A"`, `"B"`, `"T"`)
- a justification on the same line (`"GOOD because the response is correct"`)
- both keywords appearing — first wins
- "not good" (BAD) and "not bad" (GOOD), which the word alone misread
  before 0.72.8
- a reasoning model's `<think>…</think>` block, which is skipped: the
  verdict is what comes after it
- a pairwise reply that merely starts with "T" ("The better one is A"):
  only a standalone `T` or a tie word ("tie", "equal") is a tie
- nothing parseable — falls back to `fallback_label`, `BAD` by default
  (conservative: don't pollute the positive set with unjudged samples).
  Setting `fallback_label` had no effect before 0.72.8.

### A judge on a server (0.72.8)

The judge does not have to be loaded in this process. `OpenAIJudge`
talks to anything that answers OpenAI's `/chat/completions` — the
HyperNix runner, LM Studio, llama-server, or a hosted provider:

```python
from hypernix.evaluation.industrial_range import OpenAIJudge, industrial_range

judge = industrial_range(OpenAIJudge("http://127.0.0.1:8081/v1", model="lfm2.5-2.6b"),
                         max_new_tokens=16)
judge("Capital of France?", "Paris")     # -> "GOOD"
```

A reasoning model thinks before it answers: give it room
(`max_new_tokens` in the hundreds). A `"\n"` stop is not sent to a chat
server, where it would end the reply at its first line.

## Putting them in the pipeline

`mediocre_fridge.collect_responses_from` accepts any callable as
`label_rule=`:

```python
from hypernix import (
    industrial_range, mediocre_fridge, new_range, old_oven, old_range,
)

prompts = ["Capital of France?", "2 + 2 = ?", "Sort [3,1,2] in Python."]
oven = old_oven.preheat(local_dir="./candidate-snapshot")

# Option A — fast triage with rubric heuristics:
ex_a = mediocre_fridge.collect_responses_from(
    oven, prompts, label_rule=new_range.new_range(),
)

# Option B — explainable scored rubric with references:
ex_b = mediocre_fridge.collect_responses_from(
    oven, prompts,
    label_rule=old_range.old_range(
        references={"Capital of France?": "Paris"},
        keywords={"Sort [3,1,2] in Python.": ["sorted", "[1,"]},
    ),
)

# Option C — LLM-as-judge with a stronger model:
teacher = old_oven.preheat(repo_id="qwen3.5-9b", device="cuda")
ex_c = mediocre_fridge.collect_responses_from(
    oven, prompts,
    label_rule=industrial_range.industrial_range(judge=teacher),
)
```

## Climbing the ladder

A reasonable progression for a new project:

1. **Start with `new_range`.** Get a few hundred examples through it,
   eyeball the BAD bucket, add custom rules for the failure modes you
   actually see.
2. **Move to `old_range` once you have references.** Add
   `references={prompt: known_answer}` for whatever known-answer
   prompts you have, dial in `threshold` so the GOOD bucket looks
   good to you.
3. **Promote to `industrial_range` when you have access to a stronger
   model.** Use `compare()` for preference data, `label()` for
   pointwise. Cache aggressively so iterating on the rubric doesn't
   re-bill the judge.

You can also chain: use `industrial_range` to label and `old_range`
to *validate* — flag examples where the heuristic and the LLM-judge
disagree for human review.
