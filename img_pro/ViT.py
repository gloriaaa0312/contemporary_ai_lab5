# 这个文件定义了获取预训练的Bert模型和分词器的函数

from transformers import ViTFeatureExtractor, ViTModel
from img_pro import img_config
import config

config.setup_seed()


def getViT():
    return ViTModel.from_pretrained(
        img_config.pretrained_model,
        config=img_config.pretrained_model
    )


def getViTExtractor():
    return ViTFeatureExtractor.from_pretrained(img_config.pretrained_model)