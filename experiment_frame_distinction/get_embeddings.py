# -*-coding:utf-8-*-

import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from _collate_fn import collate_fn_elmo, collate_fn_transformers
from _dataset import EmbeddingsDataset
from _model import EmbeddingsNet


def parse_args():
    RESOURCE = ["framenet", "propbank"][0]
    DEVICE = ["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cpu"][0]
    MODEL_NAME = [
        "elmo",
        "bert-base-uncased",
        "bert-large-uncased",
        "albert-base-v2",
        "roberta-base",
        "gpt2",
        "xlnet-base-cased",
    ][0]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default="../data/experiment_frame_distinction/dataset",
    )
    parser.add_argument("--elmo_path", type=str, default="../data/raw/elmo")
    parser.add_argument(
        "--output_path",
        type=str,
        default="../data/experiment_frame_distinction/embeddings",
    )

    parser.add_argument("--resource", type=str, default=RESOURCE)
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


def make_dir_path(input_path, elmo_path, output_path, resource, model_name):
    path_dict = {
        "input": "/".join([input_path, resource]),
        "elmo": elmo_path,
        "output": "/".join([output_path, resource, model_name]),
    }
    for key, path in path_dict.items():
        path_dict[key] = os.path.abspath(path) + "/"
        if "output" in key:
            if not os.path.isdir(path_dict[key]):
                os.makedirs(path_dict[key])
    return path_dict


def main():
    args = parse_args()
    path_dict = make_dir_path(
        args.input_path,
        args.elmo_path,
        args.output_path,
        args.resource,
        args.model_name,
    )

    df = pd.read_json(
        path_dict["input"] + "exemplars.jsonl", orient="records", lines=True
    )

    if args.model_name in ["elmo"]:
        model = EmbeddingsNet(args.model_name, args.device, path=path_dict["elmo"])
        collate_fn = collate_fn_elmo
        layer = 2
    else:
        model = EmbeddingsNet(args.model_name, args.device)
        collate_fn = collate_fn_transformers
        layer = 24 if args.model_name in ["bert-large-uncased"] else 12

    dl = DataLoader(
        EmbeddingsDataset(df, args.model_name),
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
    )

    vec_dict = {l: [] for l in range(layer + 1)}
    for batch in tqdm(dl):
        with torch.no_grad():
            embs = model.get_vec(batch)
            for l in range(layer + 1):
                vec_dict[l] += list(embs[l].cpu().detach().numpy())

    df_vec = df.reset_index(drop=True).reset_index().rename(columns={"index": "vec_id"})
    df_vec.to_json(
        path_dict["output"] + "exemplars.jsonl",
        orient="records",
        force_ascii=False,
        lines=True,
    )

    for l, vec in vec_dict.items():
        vec_dict = {"vec": np.array(vec)}
        np.savez_compressed(
            path_dict["output"] + "vec_" + str(l).zfill(2),
            **vec_dict,
        )


if __name__ == "__main__":
    main()
