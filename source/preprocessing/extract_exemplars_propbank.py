import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from tqdm import tqdm

from vsc.data_utils import write_jsonl


def make_exemplars(frames, anno_list):
    id2frame = {}
    for file in tqdm(frames):
        root = ET.parse(file).getroot()
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
    return output_list


def main(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)

    outputs = make_exemplars(
        args.input_frame_dir.glob("*.xml"),
        args.input_annotation_dir.glob("*/*/*/*.onf"),
    )
    write_jsonl(outputs, args.output_dir / "exemplars.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_frame_dir", type=Path, required=True)
    parser.add_argument("--input_annotation_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    main(args)
