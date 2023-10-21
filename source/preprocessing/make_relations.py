import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

from tqdm import tqdm

from vsc.data_utils import write_json


def make_f2f_dataframe(input_files, relations):
    head = "{http://framenet.icsi.berkeley.edu}"

    output_list = []
    for input_file in tqdm(input_files):
        root = ET.parse(input_file).getroot()
        target_frame = root.attrib["name"]
        for e in root:
            if e.tag == f"{head}frameRelation":
                relation = e.attrib["type"]
                for ee in e:
                    related_frame = ee.text
                    output_dict = {
                        "target_frame": target_frame,
                        "relation": relation,
                        "related_frame": related_frame,
                    }
                    if relation in relations["parent"]:
                        output_dict["hierarchy"] = "parent"
                    elif relation in relations["child"]:
                        output_dict["hierarchy"] = "child"
                    else:
                        output_dict["hierarchy"] = "similar"
                    output_list.append(output_dict)
    return output_list


def main(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)

    relations = {
        "child": [
            "Is Inherited by",
            "Is Used by",
            "Has Subframe(s)",
            "Is Perspectivized in",
            "Is Preceded by",
        ],
        "parent": [
            "Inherits from",
            "Uses",
            "Subframe of",
            "Perspective on",
            "Precedes",
            "Is Inchoative of",
            "Is Causative of",
        ],
    }

    outputs = make_f2f_dataframe(args.input_dir.glob("*.xml"), relations)
    write_json(outputs, args.output_dir / "f2f_list.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    main(args)
