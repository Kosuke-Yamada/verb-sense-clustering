# -*-coding:utf-8-*-

import argparse
import glob
import os
import re
import xml.etree.ElementTree as ET

import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_frame_path",
        type=str,
        default="../data/raw/ontonotes/metadata/frames",
    )
    parser.add_argument(
        "--input_anno_path", type=str, default="../data/raw/ontonotes/annotations"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="../data/preprocessing/propbank",
    )
    return parser.parse_args()


def make_dir_path(input_frame_path, input_anno_path, output_path):
    path_dict = {
        "input_frame": input_frame_path,
        "input_anno": input_anno_path,
        "output": output_path,
    }
    for key, path in path_dict.items():
        path_dict[key] = os.path.abspath(path) + "/"
        if "output" in key:
            if not os.path.isdir(path_dict[key]):
                os.makedirs(path_dict[key])
    return path_dict


def make_exemplars_dataframe(frame_list, anno_list):
    id2frame = {}
    for path in tqdm(frame_list):
        root = ET.parse(path).getroot()
        for e in root.findall("predicate"):
            for ee in e:
                if len(ee.attrib) != 0:
                    frame_dict = ee.attrib
                    lu_id = frame_dict["id"]
                    frame = "_".join(frame_dict["name"].split())
                    id2frame[lu_id] = lu_id + "-" + frame

    output_list, ex_idx = [], 0
    for path in tqdm(anno_list):
        with open(path, "r") as f:
            line_list = [d.strip() for d in f.readlines()]
        flag = 0
        for line in line_list:
            if flag == 1 and line == "":
                for target_widx, lu_id in zip(target_widx_list, lu_id_list):
                    output_dict = {
                        "ex_idx": ex_idx,
                        "text_widx": " ".join(word_list),
                        "lu_id": lu_id,
                        "frame_name": id2frame[lu_id],
                        "target_widx": target_widx,
                        "verb": lu_id.split(".")[0],
                        "resource": "propbank",
                    }
                    output_list.append(output_dict)
                    ex_idx += 1
                flag = 0
            if line == "Leaves:":
                word_count, flag = -1, 1
                word_list, lu_id_list, target_widx_list = [], [], []
            if flag == 1:
                line_head = line.split()[0]
                line_num = re.match("\d+", line_head)
                if (line_num is not None) and len(line.split()) >= 2:
                    word = line.split()[1]
                    if word[0] != "*":
                        word_count += 1
                        word_list.append(word)
                if line_head == "prop:":
                    lu_id = line.split()[1]
                    lu_id_list.append(lu_id)
                    target_widx_list.append(word_count)
    return pd.DataFrame(output_list)


def main():
    args = parse_args()
    path_dict = make_dir_path(
        args.input_frame_path, args.input_anno_path, args.output_path
    )

    frame_list = glob.glob(path_dict["input_frame"] + "*xml")
    anno_list = glob.glob(path_dict["input_anno"] + "*/*/*/*.onf")

    df = make_exemplars_dataframe(frame_list, anno_list)
    df.to_json(
        path_dict["output"] + "exemplars.jsonl",
        orient="records",
        force_ascii=False,
        lines=True,
    )


if __name__ == "__main__":
    main()
