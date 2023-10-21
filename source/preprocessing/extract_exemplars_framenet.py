import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from vsc.data_utils import write_jsonl


def make_lu_list(files):
    lu_list = []
    for file in tqdm(files):
        lu_list.append(ET.parse(file).getroot().attrib)
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


def make_exemplars(df, input_dir):
    head = "{http://framenet.icsi.berkeley.edu}"
    id2frame = df[["ID", "frame"]].set_index("ID").to_dict()["frame"]

    output_list, ex_idx = [], 0
    for verb in tqdm(sorted(set(df[df["POS"] == "V"]["name"]))):
        verb = verb.replace("/", "-")
        for lu_id in df[df["name"] == verb]["ID"]:
            root = ET.parse(f"{input_dir}/lu{lu_id}.xml").getroot()
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
                                    ) and int(eeeee.attrib["start"]) < len(
                                        text
                                    ):
                                        output_dict[
                                            "target_widx"
                                        ] = start2index[
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
    return output_list


def main(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df_lu = make_lu_list(args.input_dir.glob("lu*.xml"))
    outputs = make_exemplars(df_lu, args.input_dir)
    write_jsonl(outputs, args.output_dir / "exemplars.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    main(args)
