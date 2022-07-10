# -*-coding:utf-8-*-

import argparse
import glob
import os
import xml.etree.ElementTree as ET

import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path", type=str, default="../data/raw/FNDATA-1.7/frame"
    )
    parser.add_argument(
        "--output_path", type=str, default="../data/preprocessing/relations"
    )
    return parser.parse_args()


def make_dir_path(input_path, output_path):
    path_dict = {"input": input_path, "output": output_path}
    for key, path in path_dict.items():
        path_dict[key] = os.path.abspath(path) + "/"
        if "output" in key:
            if not os.path.isdir(path_dict[key]):
                os.makedirs(path_dict[key])
    return path_dict


def make_f2f_dataframe(input_path):
    head = "{http://framenet.icsi.berkeley.edu}"
    relation_dict = {
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
    output_list = []
    for file_path in tqdm(list(glob.glob(input_path + "*.xml"))):
        root = ET.parse(file_path).getroot()
        target_frame = root.attrib["name"]
        for e in root:
            if e.tag == head + "frameRelation":
                relation = e.attrib["type"]
                for ee in e:
                    related_frame = ee.text
                    output_dict = {
                        "target_frame": target_frame,
                        "relation": relation,
                        "related_frame": related_frame,
                    }
                    if relation in relation_dict["parent"]:
                        output_dict["hierarchy"] = "parent"
                    elif relation in relation_dict["child"]:
                        output_dict["hierarchy"] = "child"
                    else:
                        output_dict["hierarchy"] = "similar"
                    output_list.append(output_dict)
    return pd.DataFrame(output_list)


def main():
    args = parse_args()
    path_dict = make_dir_path(args.input_path, args.output_path)

    df = make_f2f_dataframe(path_dict["input"])
    df.to_json(
        path_dict["output"] + "f2f_list.jsonl",
        orient="records",
        force_ascii=False,
        lines=True,
    )


if __name__ == "__main__":
    main()
