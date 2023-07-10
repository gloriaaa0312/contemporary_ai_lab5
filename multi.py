import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from runUtils import train, test, predict, device
from multiData import getMultiDataset
from img_pro import img_config
from txt_pro import txt_config
import config

config.setup_seed()


def test_mul():
    model = torch.load('model_multi', map_location=device)
    dataset = getMultiDataset('./data_pre/input/trainData.json')
    val_dataset = Subset(dataset, range(3500, 4000))
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)
    print('final validation accuracy:', test(model, val_loader))


def predict_mul():
    model = torch.load('model_multi', map_location=device)
    test_loader = DataLoader(getMultiDataset('./data_pre/input/testData.json'), batch_size=config.batch_size,
                             shuffle=False)
    predict(model, test_loader)


class MultiModel(nn.Module):
    def __init__(self, fine_tune: bool):
        super().__init__()

        self.fig_model = torch.load('model_fig', map_location=device)
        self.txt_model = torch.load('model_txt', map_location=device)

        for param in self.fig_model.ViT.parameters():
            param.requires_grad_(fine_tune)
        for param in self.txt_model.bert.parameters():
            param.requires_grad_(fine_tune)

        self.dp = nn.Dropout(0.5)
        self.fc = nn.Linear(768 * 2, 3)

    def forward(self, data):
        pixel_values = data['img']['pixel_values'][:, 0].to(device)
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        token_type_ids = data['token_type_ids'].to(device)

        img_out = self.fig_model.ViT(
            pixel_values=pixel_values
        )

        bert_out = self.txt_model.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        out = torch.concat([img_out['pooler_output'], bert_out['pooler_output']], dim=1)
        out = self.fc(self.dp(out))

        return out


def run():
    print(device)
    model = MultiModel(fine_tune=config.fine_tune)
    model.to(device)

    bert_params = list(map(id, model.txt_model.bert.parameters()))
    ViT_params = list(map(id, model.fig_model.ViT.parameters()))
    down_params = filter(lambda p: id(p) not in bert_params + ViT_params, model.parameters())
    optimizer = AdamW([
        {'params': model.txt_model.bert.parameters(), 'lr': txt_config.bert_lr},
        {'params': model.fig_model.ViT.parameters(), 'lr': img_config.ViT_lr},
        {'params': down_params, 'lr': config.lr}
    ])

    dataset = getMultiDataset('./data_pre/input/trainData.json')
    # print(len(dataset))
    train_dataset = Subset(dataset, range(0, 3500))
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    val_dataset = Subset(dataset, range(3500, 4000))
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)

    train(model, optimizer, train_loader, val_loader, "multi")


if __name__ == '__main__':
    # run()
    test_mul()
    predict_mul()
