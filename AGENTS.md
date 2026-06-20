@README.md

# Working agreement

The README above is the project context. This file is the bar for changing the
code. Hold it.

- **Minimalism.** Every file, function, and line earns its keep. Prefer deleting
  to adding, and the smaller change to the larger one.
- **One source of truth.** No duplicated constant, default, or idiom.
- **No hardcoding.** Name a value once and derive the rest; no sprinkled magic
  numbers.
- **Explicit over clever.** Dict dispatch over if/elif chains; keyword arguments;
  types on signatures; assert shapes and preconditions.
- **Fail loud.** No silent fallbacks, no swallowed errors.
- **Nest by concept.** `nets/`, `data/`, `plot/`; name a module for what it is,
  never a junk name or a "stage".

Zero tolerance for duplication, dead code, hardcoded magic, and if/else soup.

Before every commit, all green:

```bash
uv run just
```
