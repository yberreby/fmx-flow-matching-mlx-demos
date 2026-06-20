import subprocess
from pathlib import Path

TEMPLATE = Path("README.md.in")
OUTPUT = Path("README.md")
PLACEHOLDER = "{{MODULE_TREE}}"


def main() -> None:
    tree = subprocess.run(["uv", "run", "pypatree"], capture_output=True,
                          text=True, check=True).stdout.strip()
    OUTPUT.write_text(TEMPLATE.read_text().replace(PLACEHOLDER, tree))


if __name__ == "__main__":
    main()
