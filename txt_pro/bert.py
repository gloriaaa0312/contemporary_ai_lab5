# 这个文件定义了获取预训练的Bert模型和分词器的函数

from transformers import BertModel, BertTokenizer
import txt_pro.txt_config as txt_config
import config


config.setup_seed()


def getBert():
    return BertModel.from_pretrained(
        txt_config.pretrained_model,
        config=txt_config.pretrained_model
    )


def getBertTokenizer():
    return BertTokenizer.from_pretrained(txt_config.pretrained_model)
