import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import ViTFeatureExtractor
from img_pro import ViT as vit
from PIL import Image
import config

config.setup_seed()

tags = {
    'positive': 0,
    'negative': 1,
    'neutral': 2,
    '': 3  # 仅占位
}


# 构建自定义数据集
class ImgDataset(Dataset):
    def __init__(self, data, extractor: ViTFeatureExtractor):
        self.data = data
        self.extractor = extractor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        guid = self.data[item]['guid']
        img = self.data[item]['img']
        tag = self.data[item]['tag']

        # 将tag转化为int并转成张量
        tag = torch.tensor(tags[tag], dtype=torch.long)

        img = self.extractor(
            images=Image.open('data_pre/data/' + img),
            return_tensors='pt'
        )

        return {
            'guid': guid,
            'img': img,
            'tag': tag
        }
        

def getImgDataset(path: str):
    with open(path, 'r', encoding='utf-8') as fs:
        data = json.load(fs)
    extracker = vit.getViTExtractor()
    return ImgDataset(data, extracker)



def img_drive():
    img_dataset = getImgDataset('../data_pre/input/trainData.json')

    # 构建dataLoader
    img_loader = DataLoader(dataset=img_dataset, batch_size=config.batch_size, shuffle=True)
    # 获取预训练模型
    pretrained = vit.getViT()
    # 关闭参数的梯度计算
    for param in pretrained.parameters():
        param.requires_grad_(False)

    # 遍历
    for i, data in enumerate(img_loader):
        print(data['img']['pixel_values'].shape)
        out = pretrained(
            pixel_values=data['img']['pixel_values'][:, 0]
        )
        print(out['last_hidden_state'].shape)
        print(out['pooler_output'].shape)
        break
