import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import txt_pro.bert as bert
import txt_pro.txt_config as txt_config
import config

config.setup_seed()

tags = {
    'positive': 0,
    'negative': 1,
    'neutral': 2,
    '': 3
}


# 构建自定义数据集
class TextDataset(Dataset):
    def __init__(self, data, max_len, tokenizer: BertTokenizer):
        self.data = data
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        guid = self.data[item]['guid']
        text = self.data[item]['text']
        tag = self.data[item]['tag']

        # 将tag转成张量
        tag = torch.tensor(tags[tag], dtype=torch.long)

        # 用BERT分词器对文本进行编码
        encoding_data = self.tokenizer.encode_plus(
            text,add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # 以字典形式返回
        return {
            'guid': guid,
            'text': text,
            'input_ids': encoding_data['input_ids'].flatten(),
            'attention_mask': encoding_data['attention_mask'].flatten(),
            'token_type_ids': encoding_data['token_type_ids'].flatten(),
            'tag': tag
        }


def text_drive():
    with open('data_pre/input/trainData.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    tokenizer = bert.getBertTokenizer()
    text_dataset = TextDataset(data, txt_config.max_len, tokenizer)

    # 构建dataLoader
    txt_loader = DataLoader(dataset=text_dataset, batch_size=config.batch_size, shuffle=True)

    # 获取预训练模型
    pretrained = bert.getBert()
    # 关闭参数的梯度计算
    for param in pretrained.parameters():
        param.requires_grad_(False)

    for i, data in enumerate(txt_loader):
        # print(data)
        out = pretrained(
            input_ids=data['input_ids'],
            attention_mask=data['attention_mask'],
            token_type_ids=data['token_type_ids']
        )
        print(out['last_hidden_state'].shape)
        break


# 读取文本数据集，方便在其他文件中调用
def getTextDataset(path: str):
    with open(path, 'r', encoding='utf-8') as fs:
        data = json.load(fs)

    tokenizer = bert.getBertTokenizer()
    return TextDataset(data, txt_config.max_len, tokenizer)
# data_loader = DataLoader(getTextDataset('../data_pre/input/trainData.json'), batch_size=config.batch_size, shuffle=True)
