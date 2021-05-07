from torch import nn
import torch
from transformers import AutoModel


class DeepPavlovBERT(nn.Module):
    def __init__(self, classes_num):
        super(DeepPavlovBERT, self).__init__()

        self.backbone = AutoModel.from_pretrained(
            pretrained_model_name_or_path='DeepPavlov/rubert-base-cased-sentence'
        )

        self.classifier_head = nn.Sequential(
            nn.Linear(in_features=768, out_features=classes_num)
        )

    def forward(self, input_ids, labels=None):
        x = self.backbone(input_ids)
        x = x.pooler_output

        if labels is not None:
            return self.classifier_head(x)

        return x


if __name__ == '__main__':
    model = DeepPavlovBERT(3)
    model(torch.tensor([[1, 2, 35]]))
