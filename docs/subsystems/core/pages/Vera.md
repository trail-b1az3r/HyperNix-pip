# Vera

**Vera** is the smoke-testing and linting tool built into HyperNix. 

## Features
- Linting using `ast` and `ruff`
- Smoke testing (module import validation)
- Pytest fallback
- Argument testing and execution tracking (`-dr`, `-C`, `-FT`)
- Advanced AI error explanation powered by Qwen3.5

## Usage
Run `hnx vera <python_file> [options]`

**Options**:
- `-dr, --dry-run`: Run the file with the dry-run flag
- `-C`: Fully run the file
- `-FT`: Test each argument one at a time
- `-q <depth>`: Change argument testing depth (default 1, up to 10)
- `-t <time>`: Timeout limit
- `-T <unit>`: Timeout unit (`s` for seconds, `M` for minutes, `ml` for milliseconds)
