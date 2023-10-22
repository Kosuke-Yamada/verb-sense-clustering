from spacy_alignments.tokenizations import get_alignments
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class EmbeddingsDataset(Dataset):
    def __init__(self, df, model_name):
        self.df = df
        self.model_name = model_name

        if model_name != "elmo":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self._preprocess()

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        return self.out_inputs[idx]

    def _preprocess(self):
        self.out_inputs = []
        for df_dict in self.df.to_dict("records"):
            text_widx = df_dict["text_widx"].split()
            if self.model_name == "elmo":
                inputs = {"input_words": text_widx}
                inputs["target_tidx"] = df_dict["target_widx"]
            else:
                inputs = self.tokenizer(" ".join(text_widx))
                text_tidx = self.tokenizer.convert_ids_to_tokens(
                    inputs["input_ids"]
                )
                alignments, previous_char_idx_list = [], [1]
                for char_idx_list in get_alignments(text_widx, text_tidx)[0]:
                    if len(char_idx_list) == 0:
                        alignments.append(previous_char_idx_list)
                    else:
                        alignments.append(char_idx_list)
                        previous_char_idx_list = char_idx_list
                inputs["target_tidx"] = alignments[df_dict["target_widx"]][0]
                # inputs = self.tokenizer(text_widx, is_split_into_words=True)
                # inputs["target_tidx"] = inputs.word_ids().index(
                #     df_dict["target_widx"]
                # )
            inputs.update(
                {
                    "frame_name": df_dict["frame_name"],
                    "verb": df_dict["verb"],
                    "ex_idx": df_dict["ex_idx"],
                }
            )
            self.out_inputs.append(inputs)
        self.data_num = len(self.out_inputs)
