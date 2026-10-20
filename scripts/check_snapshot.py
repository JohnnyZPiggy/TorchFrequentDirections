"""Verify every pinned snapshot byte and parse Python/JSON/SVG locally."""

import ast
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "reports/issue1/framework-inventory.json").read_text(encoding="utf-8"))
    expected = {r["path"] for r in manifest["files"]}
    actual = {p.relative_to(root / ".sae").as_posix() for p in (root / ".sae").rglob("*")
              if p.is_file() and "__pycache__" not in p.parts}
    assert actual == expected, (actual - expected, expected - actual)
    broken = []
    for row in manifest["files"]:
        path = root / ".sae" / row["path"]
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == row["sha256"], row["path"]
        text = data.decode("utf-8")
        if path.suffix == ".py":
            ast.parse(text)
        elif path.suffix == ".json":
            json.loads(text)
        elif path.suffix == ".svg":
            ET.fromstring(text)
        if path.suffix == ".md":
            for link in re.findall(r"\]\(([^)]+)\)", text):
                target = link.split("#")[0]
                if not target or re.match(r"^[a-z]+:", target) or "<" in target or " " in target:
                    continue
                if not path.parent.joinpath(target).exists():
                    broken.append({"path": row["path"], "target": target})
    assert broken == manifest["preexisting_relative_link_failures"], broken
    # Project-root adaptation of the relocatable example must actually work here.
    for path in (root / "AGENTS.md", root / "README.md"):
        for link in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            target = link.split("#")[0]
            if target and not re.match(r"^[a-z]+:", target):
                assert path.parent.joinpath(target).exists(), (path.name, target)
    print(json.dumps({"status": "PASS", "framework_sha": manifest["framework_sha"],
                      "files_verified": len(expected), "preexisting_template_links": len(broken),
                      "project_root_links": "PASS", "parsers": "Python AST / JSON / SVG XML"}))


if __name__ == "__main__":
    main()
