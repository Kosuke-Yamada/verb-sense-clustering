import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from vsc.clustering import VerbSenseClustering
from vsc.data_utils import read_json, read_jsonl, write_json, write_jsonl


def main(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.layer == "best":
        best_score, layer = -1, -1
        for file in args.input_dev_dir.glob("score-*.json"):
            score = read_json(file)["score"]
            if best_score < score:
                best_score = score
                layer = file.name.split("-")[1].split(".json")[0]
    else:
        layer = args.layer

    df = pd.DataFrame(read_jsonl(args.input_dir / "exemplars.jsonl"))
    vec_array = np.load(
        args.input_dir / f"vec-{layer}.npz",
        allow_pickle=True,
    )["vec"]

    df = df[df["sets"] == args.sets]
    vec_array = vec_array[df.index]
    df = df.reset_index()

    vsc = VerbSenseClustering(args.n_seeds, args.covariance_type)

    output_list = []
    for verb in tqdm(sorted(set(df["verb"]))):
        df_verb = df[df["verb"] == verb].copy()
        vec_verb = vec_array[df_verb.index]
        if args.model_name == "all-in-one-cluster":
            df_verb["cluster_id"] = vsc.run_all_in_one_cluster(vec_verb)
        else:
            df_verb["cluster_id"] = vsc.run_clustering(
                vec_verb, len(set(df_verb["frame_name"]))
            )
        score = vsc.calc_matching_scores(
            df_verb["frame_name"], df_verb["cluster_id"]
        )
        output_list.append({"verb": verb, "score": score})

    df_output = pd.DataFrame(output_list)
    file = f"verb_scores-{args.layer}.jsonl"
    write_jsonl(output_list, args.output_dir / file)

    score_dict = {"score": df_output["score"].mean(), "layer": layer}
    file = f"score-{args.layer}.json"
    write_json(score_dict, args.output_dir / file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--input_dev_dir", type=Path, required=False)

    parser.add_argument(
        "--resource", type=str, choices=["framenet", "propbank"]
    )
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
    parser.add_argument("--layer", type=str, default="00")
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--covariance_type", type=str, default="spherical")
    parser.add_argument("--sets", type=str, choices=["dev", "test"])
    args = parser.parse_args()
    main(args)
