import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from runUtils import train, test, predict, device
from txt_pro import bert, txt_config
from txt_pro.text import getTextDataset
import config

config.setup_seed()


def test_txt():
    model = torch.load('model_txt', map_location=device)
    dataset = getTextDataset('./data_pre/input/trainData.json')

    val_dataset = Subset(dataset, range(3500, 4000))
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)
    print('final validation accuracy:', test(model, val_loader))


def predict_txt():
    model = torch.load('model_txt', map_location=device)
    test_loader = DataLoader(getTextDataset('./data_pre/input/testData.json'), batch_size=config.batch_size, shuffle=False)
    predict(model, test_loader)


class TextModel(nn.Module):
    def __init__(self, fine_tune: bool):
        super().__init__()

        self.bert = bert.getBert()
        for param in self.bert.parameters():
            param.requires_grad_(fine_tune)

        self.dp = nn.Dropout(0.5)
        self.fc = nn.Linear(768, 3)

    def forward(self, data):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        token_type_ids = data['token_type_ids'].to(device)

        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        out = self.fc(self.dp(out['pooler_output']))

        return out


def run():
    model = TextModel(fine_tune=txt_config.fine_tune)
    model.to(device)

    bert_params = list(map(id, model.bert.parameters()))
    down_params = filter(lambda p: id(p) not in bert_params, model.parameters())

    optimizer = AdamW([
        {'params': model.bert.parameters(), 'lr': txt_config.bert_lr},
        {'params': down_params, 'lr': txt_config.lr}
    ])

    dataset = getTextDataset('./data_pre/input/trainData.json')
    train_dataset = Subset(dataset, range(0, 3500))

    val_dataset = Subset(dataset, range(3500, 4000))
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    train(model, optimizer, train_loader, val_loader, "txt")


if __name__ == '__main__':
    # run()
    test_txt()
    predict_txt()