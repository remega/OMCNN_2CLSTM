# OM-CNN+2C-LSTM for video salinecy prediction
The model of **"Predicting Video Saliency using Object-to-Motion CNN and Two-layer Convolutional LSTM"**, which can be found in https://arxiv.org/abs/1709.06316, with the Bibtex code:  
@InProceedings{Jiang2017predicting,  
author = {Lai Jiang, and Mai Xu, and Zulin Wang},  
title = {Predicting Video Saliency with Object-to-Motion CNN and Two-layer Convolutional LSTM},  
booktitle = {arXiv preprint arXiv:1709.06316},  
month = {Sep.},  
year = {2017}  
}

This model is implemented by **tensorflow-gpu** 1.0.0, and the detail of our computational environment is listed in **'requirement.txt'**.

The **pre-trained model** can be domwloaded at  
Google drive: https://drive.google.com/drive/folders/0BwrNwFvsiqVaaEVVUmpyZ0RWSzA?usp=sharing  
BaiduYun:http://pan.baidu.com/s/1dFGlIY9  
The model then should be decompressed to the directory of ./model/pretrain/  

The newly-established eye-tracking database, which is mentioned in paper as **LEDOV**, is also available at  
Dropbox:  
BaiduYun:  

**'TestDemo.py'** is a demo to show how our model predicts the salinecy of an example video. For more detail of our model, please refer to the paper.

If any question, please contact jianglai.china@buaa.edu.cn, or use public issues section of this repository.
