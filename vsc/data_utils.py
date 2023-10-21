import json
from pathlib import Path

from ruamel import yaml


def write_jsonl(items, file):
    output_items = []
    for item in items:
        output_item = {}
        for k, v in item.items():
            if isinstance(v, Path):
                output_item[k] = str(v)
            else:
                output_item[k] = v
        output_items.append(output_item)
    with open(file, "w") as fo:
        for item in output_items:
            print(json.dumps(item, ensure_ascii=False), file=fo)


def write_json(items, file):
    output_items = {}
    for k, v in items.items():
        if isinstance(v, Path):
            output_items[k] = str(v)
        else:
            output_items[k] = v
    with open(file, "w") as fo:
        json.dump(output_items, fo, ensure_ascii=False, indent=2)


def read_jsonl(file):
    with open(file) as f:
        for line in f:
            yield json.loads(line)


def read_json(file):
    with open(file) as f:
        return json.load(f)


def read_yaml(file):
    yaml.SafeLoader.add_constructor(
        "tag:yaml.org,2002:python/object/apply:pathlib.PosixPath",
        lambda loader, node: Path(*loader.construct_sequence(node)),
    )
    with open(file, "r") as f:
        data = yaml.safe_load(f)
    return data
