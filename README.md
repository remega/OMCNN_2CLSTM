OM-CNN+2C-LSTM for video salinecy prediction
==========
The model of [**" DeepVS: A Deep Learning Based Video Saliency Prediction Approach
"**](http://openaccess.thecvf.com/content_ECCV_2018/html/Lai_Jiang_DeepVS_A_Deep_ECCV_2018_paper.html) at ECCV2018.

## Abstract
Over the past few years, deep neural networks (DNNs) have exhibited great success in predicting the saliency of images. However, there are few works that apply DNNs to predict the saliency of generic videos. In this paper, we propose a novel DNN-based video saliency prediction method. Specifically, we establish a large-scale eye-tracking database of videos (LEDOV), which provides sufficient data to train the DNN models for predicting video saliency. Through the statistical analysis of our LEDOV database, we find that human attention is normally attracted by objects, particularly moving objects or the moving parts of objects. Accordingly, we propose an object-to-motion convolutional neural network (OM-CNN) to learn spatio-temporal features for predicting the intra-frame saliency via exploring the information of both objectness and object motion. We further find from our database that there exists a temporal correlation of human attention with a smooth saliency transition across video frames. Therefore, we develop a two-layer convolutional long short-term memory (2C-LSTM) network in our DNN-based method, using the extracted features of OM-CNN as the input. Consequently, the inter-frame saliency maps of videos can be generated, which consider the transition of attention across video frames. Finally, the experimental results show that our method advances the state-of-the-art in video saliency prediction.

## Publication
The extended pre-print version of our work is published in [ECCV2018](http://openaccess.thecvf.com/content_ECCV_2018/html/Lai_Jiang_DeepVS_A_Deep_ECCV_2018_paper.html), one can cite with the Bibtex code:  
```
@InProceedings{Jiang_2018_ECCV,
author = {Jiang, Lai and Xu, Mai and Liu, Tie and Qiao, Minglang and Wang, Zulin},
title = {DeepVS: A Deep Learning Based Video Saliency Prediction Approach},
booktitle = {The European Conference on Computer Vision (ECCV)},
month = {September},
year = {2018}
} 
```

## Models
The whole architecture consists of two parts: OM-CNN and 2C-LSTM, which is shown below. The pre-trained model has already been uploaded to [Google drive](https://drive.google.com/drive/folders/0BwrNwFvsiqVaaEVVUmpyZ0RWSzA?usp=sharing) and [BaiduYun](http://pan.baidu.com/s/1dFGlIY9).  
For running the demo, please the model should be decompressed to the directory of **./model/pretrain/**.

![OM-CNN](/fig/OM-CNN.png "OM-CNN")

![2C-LSTM](/fig/2C-LSTM.png "2C-LSTM")

## Database
As introduced in our paper, our model is trained by our newly-established eye-tracking database,  [LEDOV](https://github.com/remega/LEDOV-eye-tracking-database), which is also available at [Dropbox](https://www.dropbox.com/s/pc8symd9i3cky1q/LEDOV.zip?dl=0) and [BaiduYun](http://pan.baidu.com/s/1pLmfjCZ)

## Usage
This model is implemented by **tensorflow-gpu** 1.0.0, and the detail of our computational environment is listed in **'requirement.txt'**. Just run **'TestDemo.py'** to see the saliency prediction results on a test video.

## Visual Results
Some visual results of our model and ground-truth.
![visualresult](/fig/visualresult.png "visualresult")

## Contact
If any question, please contact jianglai.china@buaa.edu.cn （or jianglai.china@aliyun.com）, or use public issues section of this repository.

## License
This code is distributed under MIT LICENSE.

## Ablation 

![Ablation](/fig/abaltionfig.png "Ablation")

## Supplementary material
[Link](https://www.dropbox.com/s/p88eya7ek9xg8pf/eccv2018supplementary-2550.pdf?dl=0)
