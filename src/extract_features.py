# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import json
from tqdm import tqdm 
from PIL import Image
import torch
from multiprocessing import Pool
import h5py
from transformers import logging
from transformers import CLIPFeatureExtractor, CLIPVisionModel
import open_clip

logging.set_verbosity_error()

data_dir = 'data/images/'
features_dir = 'features/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 原始clip
# encoder_name = 'openai/clip-vit-base-patch32'
# feature_extractor = CLIPFeatureExtractor.from_pretrained(encoder_name) 
# clip_encoder = CLIPVisionModel.from_pretrained(encoder_name).to(device)


# 使用 RemoteCLIP 模型
clip_model_name = 'ViT-B-32'  # 选择 RemoteCLIP 模型架构：'ViT-B-32', 'ViT-L-14', 'RN50' 等
clip_model, _, preprocess = open_clip.create_model_and_transforms(clip_model_name)
tokenizer = open_clip.get_tokenizer(clip_model_name)
# 设置权重路径
path_to_your_checkpoints = '/root/work/checkpoints/models--chendelong--RemoteCLIP/snapshots/bf1d8a3ccf2ddbf7c875705e46373bfe542bce38'
ckpt = torch.load(f"{path_to_your_checkpoints}/RemoteCLIP-{clip_model_name}.pt", map_location="cpu")

# ckpt = torch.load(f"{path_to_your_checkpoints}/RemoteCLIP-{clip_model_name}.pt", map_location="cpu")
clip_model.load_state_dict(ckpt)
clip_model = clip_model.to(device).eval()  # 将模型加载到设备并设置为评估模式

annotations = json.load(open('data/dataset_coco.json'))['images']

# 以下用来获取隐藏层输出
# 用于保存每一层的隐藏层输出
hidden_states = []

def hook_fn(module, input, output):
    # 记录每一层的输出
    hidden_states.append(output)
    return output
# 注册钩子到模型的每一层
for layer in clip_model.visual.transformer.resblocks:
    layer.attn.register_forward_hook(hook_fn)



def load_data():
    data = {'train': [], 'val': [], 'test': []}

    for item in annotations:
        file_name = item['filename']
        if item['split'] == 'train' or item['split'] == 'restval':
            data['train'].append({'file_name': file_name, 'imgid': item['imgid']})
        elif item['split'] == 'val':
            data['val'].append({'file_name': file_name, 'imgid': item['imgid']})
        elif item['split'] == 'test':
            data['test'].append({'file_name': file_name, 'imgid': item['imgid']})
    return data

def encode_split(data, split):
    df = pd.DataFrame(data[split])

    bs = 256
    h5py_file = h5py.File(features_dir + '{}.hdf5'.format(split), 'w')
    for idx in tqdm(range(0, len(df), bs)):
        cocoids = df['imgid'][idx:idx + bs]
        file_names = df['file_name'][idx:idx + bs]
        images = [Image.open(data_dir + file_name).convert("RGB") for file_name in file_names]
        with torch.no_grad(): 
            # pixel_values = feature_extractor(images, return_tensors='pt').pixel_values.to(device)
            # encodings = clip_encoder(pixel_values=pixel_values).last_hidden_state.cpu().numpy()
           
            # 对每个图像进行预处理，处理成批量 Tensor
            pixel_values = torch.stack([preprocess(image).to(device) for image in images])            
            # 执行前向传播并通过钩子获取最后一层隐藏层输出
            clip_model(pixel_values)
            # 获取最后一层的隐藏层输出
            last_layer_output = hidden_states[-1][0]  # 获取最后一层的输出            
            encodings = last_layer_output.cpu().numpy()  # 这里是整个隐藏层的输出
        for cocoid, encoding in zip(cocoids, encodings):
            h5py_file.create_dataset(str(cocoid), (50, 768), data=encoding)

data = load_data()

encode_split(data, 'train')
encode_split(data, 'val')
encode_split(data, 'test')