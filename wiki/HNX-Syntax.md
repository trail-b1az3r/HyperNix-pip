# `.hnx` element syntax reference

This is the dedicated language and file-format reference for the 3H interpreter. The current on-disk element format is **v1.1.0**. Files are UTF-8 text and use the .hnx extension.

For the framework, see [Elements](Elements.md). For the interpreter design, see [3H](3H.md).

## File layout

A v1.1.0 file has three parts, in order: a format declaration; JSON Lines (JSONL) with exactly one element metadata object followed by zero or more data objects; and the hook-function body.

~~~text
#!hnx 1.1.0
{"kind":"element","symbol":"Na","summary":"Prefix user prompts","version":"0.1.0","dependencies":["C"],"groups":["chat","prompt"],"permissions":["oven"]}
{"kind":"data","name":"prefix","value":"[Na]"}

fn transform_prompt(prompt: string) -> string {
    let cleaned: string = prompt.trim();
    return data.get("prefix") + " " + cleaned;
}
~~~

Blank lines and lines beginning with # or // are ignored. Keep each JSONL object on one physical line. The element record must appear before the script body. The first non-empty line must be the `#!hnx 1.1.0` header.

## Element metadata record

Exactly one record must have `"kind":"element"`.

| Field | Required? | Meaning |
|---|---:|---|
| kind | Yes | Must be element. |
| symbol | Yes | Case-sensitive periodic-table symbol, such as Na. |
| summary | Recommended | A short description. |
| version | No | Element version string; defaults to 0.1.0. |
| dependencies | **Yes** | Array of symbols or registered element names. Use [] when there are no dependencies. |
| groups | No | Array of group names selected with hnx chat -G "name". |
| permissions | No | Array of Hydrogen permissions. Use ["oven"] for a chat transform. |

A v1.1.0 record without dependencies is rejected. Dependencies load transitively, so a file does not manually import dependency scripts. The dependency must still be available to the registry, for example as a built-in or a file in the selected directory.

Groups choose roots; they do not filter out required dependencies.

## Additional JSONL data

A data record looks like this:

~~~json
{"kind":"data","name":"prefix","value":"[Na]"}
{"kind":"data","name":"limits","value":{"max":64,"enabled":true}}
~~~

Each data record must have kind=data and a string name. Its value can be any JSON value: string, number, boolean, null, array or object. Inside a hook, records are available through the data mapping:

~~~ts
let prefix: string = data.get("prefix");
let maximum: number = data.get("limits").get("max");
~~~

The mapping is a per-call local copy. It is configuration/data, not persistent storage; changing it does not edit the .hnx file.

## Supported hooks

Only these top-level hook names are accepted:

| Hook | Parameters | Expected result |
|---|---|---|
| activate | none | Called when Hydrogen activates the element. |
| deactivate | none | Called when Hydrogen deactivates the element. |
| transform_prompt | prompt | Return the transformed prompt string. |
| transform_messages | messages | Return the transformed list of role/content dictionaries. |
| transform_output | text | Return the transformed output string. |

Hooks are optional; absent hooks use the base Hydrogen no-op. Use the exact parameter count shown above. Default arguments, keyword-only parameters, *args and **kwargs are not supported.

## Function declaration styles

The parser accepts these signature styles:

**3H / Python-inspired:**

~~~text
fn transform_prompt(prompt: string) -> string {
    return prompt.trim();
}
~~~

**TypeScript-inspired:**

~~~ts
function transform_output(text: string): string {
    return text.trim();
}
~~~

**C++-inspired:**

~~~cpp
std::string transform_prompt(std::string prompt) {
    return prompt;
}
~~~

**Python-style declaration:**

~~~python
def transform_prompt(prompt: str) -> str {
    return prompt.strip();
}
~~~

The C++-inspired signature form currently recognises string, void, int, float, double, bool and auto return/declaration keywords, with typed parameters such as std::string prompt. This is syntax convenience, not a C++ compiler or standard-library implementation.

## Local variables and types

TypeScript-like declarations:

~~~ts
let promptText: string = prompt.trim();
const enabled: boolean = true;
var attempts: integer = 3;
auto copy = promptText;
~~~

C++-inspired declarations:

~~~cpp
std::string promptText = prompt;
int attempts = 3;
bool enabled = true;
double ratio = 0.5;
~~~

Python-style assignments also work:

~~~python
prompt_text = prompt.strip()
attempts = 3
~~~

Recognised type-name aliases include string / str, boolean / bool, number / float / double, integer / int, json, any and void. Types are syntax annotations and normalisation hints; the interpreter does not perform static type checking.

## Expressions and operators

| Category | Supported forms |
|---|---|
| Literals | Strings, integers, floats, true, false, null and Python spellings True, False, None. |
| Collections | Lists, tuples and dictionaries. |
| Arithmetic | +, -, *, /, //, %. |
| Boolean logic | and, or, not, plus &&, ||, !. |
| Equality | ==, !=, ===, !==. Strict-equality spellings normalise to Python-style equality; they do not implement JavaScript type coercion. |
| Comparison | <, <=, >, >=, in, not in, is, is not. |
| Access | Local names, list/dictionary indexing and slices. |
| Conditional | if, else if, else, and expression-level conditional expressions. |
| Formatting | Formatted strings (f-strings). |

Conditional blocks use braces and one statement per line:

~~~text
if (prompt.includes("!")) {
    return prompt.replace("!", "?");
} else if (len(prompt) > 100) {
    return prompt[:100];
} else {
    return prompt;
}
~~~

## Built-in functions and methods

Allowed global functions: len, str, int, float, bool, min, max, sum, sorted, all, any, json_parse and json_stringify.

Supported string methods include trim / strip, lstrip, rstrip, lower / toLowerCase, upper / toUpperCase, casefold, title, includes / contains, startswith / startsWith, endswith / endsWith, replace, split, join, count and find.

Supported dictionary methods: get, keys, values, items, update.

Supported list methods: append, extend, index and count.

Method availability depends on the value's runtime type.

## Complete example

~~~text
#!hnx 1.1.0
{"kind":"element","symbol":"Na","summary":"Normalise and prefix prompts","version":"0.1.0","dependencies":["C"],"groups":["chat","prompt"],"permissions":["oven"]}
{"kind":"data","name":"prefix","value":"[Na]"}

fn transform_prompt(prompt: string) -> string {
    let cleaned: string = prompt.trim();
    if (cleaned.includes("!")) {
        cleaned = cleaned.replace("!", "?");
    }
    return data.get("prefix") + " " + cleaned;
}
~~~

Create a starter file:

~~~bash
hnx elements new Na --format hnx
~~~

Edit the generated file and select it:

~~~bash
hnx chat --elements ~/.hypernix/elements/sodium.hnx --message "Hello!"
~~~

A one-shot --message run loads and attaches elements like an interactive chat, then releases the attachment when the turn finishes.

## Unsupported syntax

The current interpreter intentionally does not support:

- Loops (for, while), break or continue.
- try / except, raise, with, classes, decorators or nested functions.
- Imports, arbitrary attribute access, arbitrary function calls or Python reflection.
- Assignment to object attributes or collection indexes; assignments target local names.
- Filesystem, network, process, operating-system or shell APIs.
- Multiline JSON values; JSONL records are one JSON object per line.
- Multiple statements on one source line or C/C++ preprocessor directives.

If a file uses unsupported syntax, Carbon records a load failure and skips it. Keep each block, declaration and statement on its own line.

## Related pages

- [3H runtime design and security model](3H.md)
- [Hydrogen element framework](Elements.md)
