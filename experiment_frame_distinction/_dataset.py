from torch.utils.data import Dataset
from transformers import AutoTokenizer


class EmbeddingsDataset(Dataset):
    def __init__(self, df, model_name):

        self.df = df
        self.model_name = model_name

        if model_name in ["elmo"]:
            pass
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if model_name in ["gpt2", "roberta-base"]:
                self.tokenizer.add_prefix_space = True

        self._preprocess()

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        return self.out_inputs[idx]

    def _preprocess(self):
        self.out_inputs = []
        for df_dict in self.df.to_dict("records"):
            text_widx = df_dict["text_widx"].split()
            if self.model_name in ["elmo"]:
                inputs = {"input_words": text_widx}
                inputs["target_tidx"] = df_dict["target_widx"]
            else:
                if self.model_name in ["gpt2"]:
                    text_widx = (
                        [self.tokenizer.bos_token]
                        + text_widx
                        + [self.tokenizer.eos_token]
                    )
                inputs = self.tokenizer(text_widx, is_split_into_words=True)
                inputs["target_tidx"] = inputs.word_ids().index(df_dict["target_widx"])
            inputs.update(
                {
                    "frame_name": df_dict["frame_name"],
                    "verb": df_dict["verb"],
                    "ex_idx": df_dict["ex_idx"],
                }
            )
            self.out_inputs.append(inputs)
        self.data_num = len(self.out_inputs)
