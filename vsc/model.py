import torch
import torch.nn as nn
from allennlp.modules.elmo import Elmo, _ElmoCharacterEncoder
from transformers import AutoConfig, AutoModel


class EmbeddingsNet(nn.Module):
    def __init__(self, model_name, device, options=None, weights=None):
        super(EmbeddingsNet, self).__init__()
        self.model_name = model_name
        self.device = device
        if model_name == "elmo":
            self.model = Elmo(options, weights, 2, dropout=0).to(device)
            self.model_t = _ElmoCharacterEncoder(options, weights).to(device)
        else:
            config = AutoConfig.from_pretrained(
                model_name, output_hidden_states=True
            )
            self.model = AutoModel.from_pretrained(
                model_name, config=config
            ).to(device)

    def _get_elmo(self, batch):
        input_ids = batch["input_ids"].to(self.device)
        outputs = self.model_t(input_ids)["token_embedding"][:, 1:-1, :]
        embs = torch.concat([outputs, outputs], axis=2)
        embs2 = self.model(input_ids)["elmo_representations"]
        return torch.cat([embs.unsqueeze(0)] + [e.unsqueeze(0) for e in embs2])

    def _get_transformers(self, batch):
        if "token_type_ids" in batch:
            embs = self.model(
                input_ids=batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
                token_type_ids=batch["token_type_ids"].to(self.device),
            )["hidden_states"]
        else:
            embs = self.model(
                input_ids=batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
            )["hidden_states"]
        return torch.cat([e.unsqueeze(0) for e in embs])

    def get_vec(self, batch):
        if self.model_name == "elmo":
            embs = self._get_elmo(batch)
        else:
            embs = self._get_transformers(batch)
        return embs[
            :,
            torch.LongTensor(range(len(batch["target_tidx"]))),
            batch["target_tidx"],
        ]
