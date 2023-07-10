import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from runUtils import train, test, predict, device
from img_pro import img_config, ViT
from img_pro.image import getImgDataset
import config

config.setup_seed()


def test_img():
    model = torch.load('model', map_location=device)
    dataset = getImgDataset('./data_pre/input/trainData.json')

    val_dataset = Subset(dataset, range(3500, 4000))
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)

    print('final validation accuracy:', test(model, val_loader))


def predict_img():
    model = torch.load('model', map_location=device)
    test_loader = DataLoader(getImgDataset('./data_pre/input/testData.json'), batch_size=config.batch_size, shuffle=False)
    predict(model, test_loader)


class ImgModel(nn.Module):
    def __init__(self, fine_tune: bool):
        super().__init__()

        self.ViT = ViT.getViT()
        for param in self.ViT.parameters():
            param.requires_grad_(fine_tune)

        self.dp = nn.Dropout(0.1)
        self.fc = nn.Linear(768, 3)

    def forward(self, data):
        pixel_values = data['img']['pixel_values'][:, 0].to(device)

        out = self.ViT(
            pixel_values=pixel_values
        )
        out = self.fc(self.dp(out['pooler_output']))

        return out


def run():
    model = ImgModel(fine_tune=img_config.fine_tune)
    model.to(device)

    ViT_params = list(map(id, model.ViT.parameters()))
    down_params = filter(lambda p: id(p) not in ViT_params, model.parameters())
    optimizer = AdamW([
        {'params': model.ViT.parameters(), 'lr': img_config.ViT_lr},
        {'params': down_params, 'lr': img_config.lr}
    ])

    dataset = getImgDataset('./data_pre/input/trainData.json')
    train_dataset = Subset(dataset, range(0, 3500))

    val_dataset = Subset(dataset, range(3500, 4000))
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    train(model, optimizer, train_loader, val_loader, "img")




if __name__ == '__main__':
    # run()
    test_img()
    predict_img()