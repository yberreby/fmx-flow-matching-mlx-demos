@README.md

- **Minimalism.** Every file, function, and line must earn its keep.
- **One source of truth.** No duplicated constant, default, or idiom.
- **No hardcoding.** No sprinkled magic numbers.
- **Explicit over clever.** Dict dispatch over if/elif chains; keyword arguments;
- **Fail loud.** No silent fallbacks, no swallowed errors. Types on signatures; assert shapes and preconditions.

Zero tolerance for duplication, dead code, hardcoded magic, and if/else soup.

Before committing, ensure all green:

```bash
uv run just
```
