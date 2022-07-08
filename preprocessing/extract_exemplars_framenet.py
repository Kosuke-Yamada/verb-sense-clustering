# -*-coding:utf-8-*-

import argparse
import glob
import os
import xml.etree.ElementTree as ET

import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="../data/raw/FNDATA-1.7/lu")
    parser.add_argument(
        "--output_path",
        type=str,
        default="../data/preprocessing/framenet",
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


def make_lu_list(file_list):
    lu_list = []
    for file_path in tqdm(file_list):
        lu_list.append(ET.parse(file_path).getroot().attrib)
    return pd.DataFrame(lu_list)


def get_start2index(text):
    word_count = 0
    start2index = {0: 0}
    pre_char = text[0].strip()
    index = 1
    for char in text[1:]:
        char = char.strip()
        if char != "":
            if pre_char == "":
                word_count += 1
                start2index[index] = word_count
            else:
                start2index[index] = word_count
        pre_char = char
        index += 1
    return start2index


def make_exemplars_dataframe(df, input_path):
    head = "{http://framenet.icsi.berkeley.edu}"
    id2frame = df[["ID", "frame"]].set_index("ID").to_dict()["frame"]

    output_list, ex_idx = [], 0
    for verb in tqdm(sorted(set(df[df["POS"] == "V"]["name"]))):
        verb = verb.replace("/", "-")
        for lu_id in df[df["name"] == verb]["ID"]:
            root = ET.parse(input_path + "lu" + str(lu_id) + ".xml").getroot()
            for e in root.findall(head + "subCorpus"):
                for ee in e.findall(head + "sentence"):
                    eee = ee.findall(head + "text")[0]
                    text = eee.text
                    start2index = get_start2index(text)
                    output_dict = {}
                    for eee in ee.findall(head + "annotationSet"):
                        for eeee in eee.findall(head + "layer"):
                            if eeee.attrib["name"] == "Target":
                                for eeeee in eeee.findall(head + "label"):
                                    if int(eeeee.attrib["start"]) <= int(
                                        eeeee.attrib["end"]
                                    ) and int(eeeee.attrib["start"]) < len(text):
                                        output_dict["target_widx"] = start2index[
                                            int(eeeee.attrib["start"])
                                        ]
                                        break
                    if "target_widx" not in output_dict:
                        continue

                    output_dict.update(
                        {
                            "text_widx": text,
                            "lu_id": str(lu_id),
                            "frame_name": id2frame[lu_id],
                            "verb": verb.split(".")[0],
                            "ex_idx": ex_idx,
                            "resource": "framenet",
                        }
                    )
                    output_list.append(output_dict)
                    ex_idx += 1
    return pd.DataFrame(output_list)


def main():
    args = parse_args()
    path_dict = make_dir_path(args.input_path, args.output_path)

    df_lu = make_lu_list(glob.glob(path_dict["input"] + "lu*.xml"))
    df = make_exemplars_dataframe(df_lu, path_dict["input"])
    df.to_json(
        path_dict["output"] + "exemplars.jsonl",
        orient="records",
        force_ascii=False,
        lines=True,
    )


if __name__ == "__main__":
    main()
