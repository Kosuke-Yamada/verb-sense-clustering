import torch
import torch.nn as nn
from allennlp.modules.elmo import batch_to_ids


def collate_fn_elmo(batch):
    output_dict = {
        "verb": [],
        "frame_name": [],
        "ex_idx": [],
        "batch_size": len(batch),
    }
    output_dict["input_ids"] = batch_to_ids([b["input_words"] for b in batch])
    output_dict["target_tidx"] = torch.LongTensor(
        [b["target_tidx"] for b in batch]
    )

    for b in batch:
        output_dict["verb"].append(b["verb"])
        output_dict["frame_name"].append(b["frame_name"])
        output_dict["ex_idx"].append(b["ex_idx"])
    return output_dict


def collate_fn_transformers(batch):
    output_dict = {
        "verb": [],
        "frame_name": [],
        "ex_idx": [],
        "batch_size": len(batch),
    }
    if "token_type_ids" in batch[0]:
        inputs = ["input_ids", "token_type_ids", "attention_mask"]
    else:
        inputs = ["input_ids", "attention_mask"]
    for input in inputs:
        output_dict[input] = nn.utils.rnn.pad_sequence(
            [torch.LongTensor(b[input]) for b in batch], batch_first=True
        )
    output_dict["target_tidx"] = torch.LongTensor(
        [b["target_tidx"] for b in batch]
    )

    for b in batch:
        output_dict["verb"].append(b["verb"])
        output_dict["frame_name"].append(b["frame_name"])
        output_dict["ex_idx"].append(b["ex_idx"])
    return output_dict
