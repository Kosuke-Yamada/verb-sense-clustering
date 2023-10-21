import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from vsc.collate_fn import collate_fn_elmo, collate_fn_transformers
from vsc.data_utils import write_jsonl
from vsc.dataset import EmbeddingsDataset
from vsc.model import EmbeddingsNet


def main(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_json(args.input_file, orient="records", lines=True)

    if args.model_name in ["all-in-one-cluster"]:
        vec_dict = {0: [[0]] * len(df)}
    else:
        if args.model_name in ["elmo"]:
            options = (
                args.elmo_dir / "elmo_2x4096_512_2048cnn_2xhighway_options.json"
            )
            weights = (
                args.elmo_dir / "elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
            )
            model = EmbeddingsNet(
                args.model_name, args.device, options, weights
            )
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

    df_vec = (
        df.reset_index(drop=True)
        .reset_index()
        .rename(columns={"index": "vec_id"})
    )
    write_jsonl(df_vec.to_dict("records"), args.output_dir / "exemplars.jsonl")

    for l, vec in vec_dict.items():
        vec_dict = {"vec": np.array(vec)}
        l = str(l).zfill(2)
        np.savez_compressed(args.output_dir / f"vec-{l}", **vec_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--elmo_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--model_name",
        type=str,
        choices=[
            "all-in-one-cluster",
            "elmo",
            "bert-base-uncased",
            "bert-large-uncased",
            "roberta-base",
            "albert-base-v2",
            "gpt2",
            "xlnet-base-cased",
        ],
    )
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    print(args)
    main(args)
