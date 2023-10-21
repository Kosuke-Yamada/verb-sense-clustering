import gzip
import json
from pathlib import Path
from typing import Any, Iterable, Union

from ruamel import yaml


def write_json(items: Iterable[Any], file: Union[str, Path]) -> None:
    file = str(file)
    if file.endswith(".jsonl") or file.endswith(".jsonl.gz"):
        output_items = []
        for item in items:
            output_item = {}
            for k, v in item.items():
                if isinstance(v, Path):
                    output_item[k] = str(v)
                else:
                    output_item[k] = v
            output_items.append(output_item)
        with gzip.open(file, "wt") if file.endswith(".gz") else open(
            file, "w"
        ) as fo:
            for item in output_items:
                print(json.dumps(item, ensure_ascii=False), file=fo)
    else:
        output_items = {
            k: str(v) if isinstance(v, Path) else v for k, v in items.items()
        }
        with gzip.open(file, "wt") if file.endswith(".gz") else open(
            file, "w"
        ) as fo:
            json.dump(output_items, fo, ensure_ascii=False, indent=2)


def read_json(file: Union[str, Path]) -> Iterable:
    file = str(file)
    if file.endswith(".jsonl") or file.endswith(".jsonl.gz"):
        with gzip.open(file, "rt") if file.endswith(".gz") else open(file) as f:
            for line in f:
                yield json.loads(line)
    else:
        with gzip.open(file, "rt") if file.endswith(".gz") else open(file) as f:
            for item in json.load(f):
                yield item


def read_yaml(file: Union[str, Path]) -> dict[str, Any]:
    yaml.SafeLoader.add_constructor(
        "tag:yaml.org,2002:python/object/apply:pathlib.PosixPath",
        lambda loader, node: Path(*loader.construct_sequence(node)),
    )
    with open(file, "r") as f:
        data = yaml.safe_load(f)
    return data
