# Architecture

This page is generated. It is rebuilt by the `arch-map` workflow every
time a public release is published, and it updates **in place** — there
is one chart on this page and it is always the current one.

That is the whole point of generating it. A hand-drawn architecture
diagram is wrong within a month and nobody notices, because the only
person who would notice is somebody reading it to learn the system —
exactly the person least able to tell that it is out of date. This one
is read out of the source tree: imports become edges, packages become
nodes, and the beta surface is whatever is actually marked beta.

## How to read it

The first chart is the shipped system: the major dependencies, the
modules, and what talks to what. The second, smaller chart below it is
the surface that is not settled yet:

| Line | Means |
|---|---|
| **solid** | Shipped. In the main chart, working today. |
| **dotted** | Added in beta. Present in the tree, not yet promised. |
| **red** | Declared and not built yet. |

A module is beta when it says so — `__beta__ = True` at module level —
and an edge is planned when a module names it in `__planned__`. Nothing
here is maintained by hand, because a legend maintained by hand decays
exactly like the chart it explains.

Only the largest modules are drawn. The tail of any real package is
single-file utilities, and a chart with ninety nodes communicates less
than no chart at all.

<!-- ARCHMAP:START -->
```mermaid
graph TD
    %% Shipped architecture — solid lines.
    dep_anthropic([Claude API])
    dep_fastapi([FastAPI])
    dep_numpy([NumPy])
    dep_rich([Rich])
    dep_torch([PyTorch])
    dep_uvicorn([uvicorn])
    t1api["t1api"]
    system["system"]
    quant["quant"]
    interfaces["interfaces"]
    hyperlink["hyperlink"]
    data["data"]
    models["models"]
    training["training"]
    monitoring["monitoring"]
    optimizers["optimizers"]
    neuron["neuron"]
    waiter["waiter"]
    elements["elements"]
    evaluation["evaluation"]
    scriptgen["scriptgen"]
    security["security"]
    audio["audio"]
    chat["chat"]
    timing["timing"]
    dilute["dilute"]
    t1sdk["t1sdk"]
    bridge["bridge"]
    hypernix["hypernix"]
    chat --> security
    chat --> timing
    data --> system
    dilute --> models
    evaluation --> models
    evaluation --> system
    hypernix --> interfaces
    hypernix --> system
    interfaces --> audio
    interfaces --> chat
    interfaces --> data
    interfaces --> dilute
    interfaces --> elements
    interfaces --> evaluation
    interfaces --> models
    interfaces --> monitoring
    interfaces --> neuron
    interfaces --> quant
    interfaces --> security
    interfaces --> system
    interfaces --> t1api
    interfaces --> t1sdk
    interfaces --> timing
    interfaces --> training
    models --> chat
    models --> system
    models --> training
    monitoring --> data
    monitoring --> evaluation
    monitoring --> system
    monitoring --> timing
    quant --> models
    security --> t1api
    system --> models
    system --> quant
    t1api --> interfaces
    t1api --> security
    timing --> system
    training --> evaluation
    training --> models
    training --> monitoring
    training --> optimizers
    training --> quant
    training --> system
    training --> timing
    waiter --> security
    waiter --> t1api
    audio --> dep_numpy
    audio --> dep_torch
    chat --> dep_rich
    chat --> dep_torch
    data --> dep_numpy
    data --> dep_rich
    evaluation --> dep_rich
    interfaces --> dep_numpy
    interfaces --> dep_rich
    interfaces --> dep_torch
    interfaces --> dep_uvicorn
    models --> dep_numpy
    models --> dep_torch
    monitoring --> dep_rich
    monitoring --> dep_torch
    neuron --> dep_numpy
    neuron --> dep_torch
    optimizers --> dep_torch
    quant --> dep_numpy
    quant --> dep_torch
    security --> dep_rich
    system --> dep_anthropic
    system --> dep_torch
    t1api --> dep_fastapi
    timing --> dep_torch
    training --> dep_numpy
    training --> dep_torch
    waiter --> dep_rich
```

### In beta, and not yet built

Solid is shipped and lives in the chart above. **Dotted** is a feature added in beta — present in the tree, not yet promised. **Red** is declared and not built yet.

```mermaid
graph LR
    nothing["nothing in beta right now"]
```
<!-- ARCHMAP:END -->

## Regenerating it yourself

```bash
python -m hypernix.system.archmap --write wiki/Architecture.md
```

It rewrites only the block between the two markers; everything else on
this page is ordinary prose and is left alone.
