# Hydrogen elements

HyperNix elements are optional extensions that can participate in a model's lifecycle or transform prompts, messages and generated output. **Hydrogen** defines the framework and registry; **Carbon** loads user-authored elements; **Natural Gas** attaches chat-compatible elements to an individual oven; and **3H** interprets editable `.hnx` files.

For the scripting language, see [3H](3H.md) and the dedicated [syntax reference](HNX-Syntax.md).

## The element family

| Component | Responsibility |
|---|---|
| **Hydrogen** (hydrogen.py, H) | Defines Element, ElementSpec, ElementContext and Registry; validates permissions, resolves dependencies and controls activation. |
| **Carbon** (carbon.py, C) | Prompt/output quality-of-life hooks, scaffolding and user-element loading. |
| **Natural Gas** (natural_gas.py) | Attaches elements around one oven's chat and complete methods. |
| **3H** (isotope_3h.py) | Tritium-named Hydrogen runtime for .hnx files; not a 119th periodic-table element. |
| **Magnesium** (magnesium.py, Mg) | System element for process-priority management; not chat middleware. |

The built-in registry contains Hydrogen, Carbon and Magnesium. Hydrogen is the framework entry; Carbon declares the oven permission; Magnesium declares processes. The periodic-table symbol table remains 118 elements.

## Create and inspect

Legacy Python elements remain supported:

~~~bash
hnx elements new Na
~~~

Create an editable 3H source file:

~~~bash
hnx elements new Na --format hnx
~~~

The scaffold is written beneath ~/.hypernix/elements/ by default. It refuses to overwrite an existing file or claim a built-in symbol.

~~~bash
hnx elements list
hnx elements list --json
hnx elements info Na
hnx elements info 3H
~~~

The list command includes recorded load failures, so a broken user file does not silently hide why it was skipped.

## Select elements for chat

~~~bash
# All chat-compatible built-ins and default-directory user elements.
hnx chat --elements

# A directory, including nested folders.
hnx chat --elements ~/.hypernix/elements

# One file is the requested root.
hnx chat --elements ~/.hypernix/elements/sodium.hnx

# Select roots that declare this group.
hnx chat --elements -G "chat"

# Filter the roots found in one directory.
hnx chat --elements ./my-elements -G "prompt"
~~~

The same options work with hypernix chat.

- **--elements** with no path selects all chat-compatible elements: built-ins plus user files in ~/.hypernix/elements/. It does not attach every system element. Entries without oven, such as Magnesium, are excluded.
- **--elements PATH** accepts a directory or one .py / .hnx file. Directory discovery is recursive.
- When one file is selected, only that file is a requested root. Sibling files are also loaded as possible dependency providers, but are not attached just because they are siblings.
- **-G "name"** filters requested roots using their groups metadata. Matching is case-insensitive. Dependencies of selected roots are still included even if they do not belong to the group.
- **-G** may be used by itself; the default user element directory is considered.

Periods 6 and 7 (atomic numbers 55–118) remain experimental. To explicitly select one, add **--experimental-elements**.

## Hydrogen element format v1.1.0

The on-disk format version is **1.1.0**.

| Field | Meaning |
|---|---|
| symbol | Case-sensitive periodic-table symbol, such as Na. |
| summary | Short description. |
| version | Element's own version string. |
| dependencies | Required symbols or registered names. Mandatory in .hnx v1.1.0; use [] when none are needed. |
| groups | Optional group names for -G. |
| permissions | Declared capabilities, such as oven or processes. |

Hydrogen resolves dependencies transitively, deduplicates shared dependencies and calculates dependency-first activation order. Missing dependencies and cycles stop activation before any element starts. Group filtering selects roots; it never suppresses a required dependency.

A chat root must declare oven. A dependency that requests non-chat permissions is refused for automatic chat activation. A dependency may transform chat if it declares oven.

## Hooks and ordering

An element subclasses Element and provides an ElementSpec. Optional hooks:

| Hook | When it runs | Contract |
|---|---|---|
| activate() | When enabled. | No arguments. |
| deactivate() | When disabled. | No arguments; undo activation work. |
| transform_prompt(prompt) | Before completion. | String to string. |
| transform_messages(messages) | Before chat. | Message-list to message-list. |
| transform_output(text) | After generation. | String to string. |

Prompt and message hooks run dependency-first. Output hooks run in reverse order. Hydrogen activates dependencies before dependants and deactivates them in reverse order.

Natural Gas patches one oven instance, not the NeoOven class. Detaching restores that instance's methods. Shared registry attachments use reference counting, so ending one chat does not stop an element still used by another chat.

## Permissions and trust

Hydrogen recognises processes, network, filesystem, oven and shell. A permission declaration is not a security review; inspect an element before enabling it.

**Python .py elements are executable Python.** Carbon compiles and executes them in an isolated namespace. This prevents a file such as json.py from shadowing the standard library, but it is not a security sandbox. Only load Python elements you trust.

**3H .hnx elements use a restricted interpreter.** The current language does not expose arbitrary Python execution, imports, shell commands, network access or filesystem access. See [3H](3H.md) and [the syntax reference](HNX-Syntax.md).
