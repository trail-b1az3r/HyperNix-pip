# Context relay — prompts longer than the model's context

`hypernix.runtime.relay` (0.72.8) lets a model with a small context read a
document that does not fit in it, by passing the work along a chain of
copies of the model:

```
part 1 ──► instance 0 ──notes──► part 2 ──► instance 1 ──notes──► …
                                                         └──► answer
```

Each hop sees the question, the notes so far and one part of the
document — never more than the context holds — and writes the notes the
next hop starts from. The last hop answers from the notes alone. Hops
rotate through the instances.

## Two modes

| Mode | How | Good for |
|---|---|---|
| `chain` (default) | Strictly in series: each hop reads the notes the previous one wrote | Questions where later parts only make sense with the earlier ones in mind |
| `map` | Every part digested on its own, in parallel — one thread per instance, each reading its own parts one at a time — then the digests folded together in series and answered | Speed; finding things spread across a long document |

When the whole prompt fits, a relay is one ordinary call.

## In Python

```python
from hypernix.runtime.relay import Relay, load_instances, plan

# Four copies of a 4k-context GGUF on consecutive ports (all start or none do).
pool, instances = load_instances("small-4k.gguf", instances=4, context_tokens=4096)
try:
    result = Relay(instances, context_tokens=4096).run(document, "What changed in v2?")
    print(result.answer)
    print(result.input_tokens, result.output_tokens, [h.to_dict() for h in result.hops])
finally:
    pool.unload()
```

An *instance* is any callable `(messages, max_tokens) -> Reply(text,
input_tokens, output_tokens)`; `openai_instance(client, model)` wraps any
OpenAI-compatible client (llama-server, LM Studio, the T1 bridge).

## What it costs

Reading in hops costs more than one big read would: the instructions,
the question and the notes are read again on every hop.
`plan(document, question, context_tokens=…)` says how much before
anything runs:

```python
p = plan(document, "What changed?", context_tokens=4096, instances=4)
p.requests, p.input_tokens, p.output_tokens   # worst case
p.direct_input_tokens                          # one big read
p.overhead_factor                              # relay tokens ÷ one big read's
```

The plan keeps 10% of the context free (the token count is an estimate),
budgets the notes at a sixth of the context unless told otherwise, and
refuses — `ContextTooSmall`, saying what does not fit — when there is no
room left for any of the document. Notes a model writes past their
budget are cut, so one long reply cannot push the next hop past the
context.

## On the T1 API

`POST /inference/relay` runs a relay under every rule of
`/inference/chat` — registry, assignment, quota, cascade — meters each
hop as feature `relay`, and prices the hops, the planned worst case and
one big read side by side. `"estimate_only": true` returns the plan and
the prices without running anything. See [T1 API](T1-API.md#prompts-longer-than-the-context-inferencerelay).
