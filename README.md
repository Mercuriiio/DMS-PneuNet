## DMS-PneuNet: A Dynamic Multi-scale Fusion Network with CBAM for CT Severity Grading of Severe Mycoplasma pneumoniae Pneumonia in Children

**Summary:** we propose DMS-PneuNet, a novel deep learning framework based on dynamic multi-scale fusion and spatiotemporal attention mechanisms. By concurrently extracting multi-granularity features, the model employs CBAM to adaptively enhance lesion features. It utilizes Bi-LSTM to establish correlations between pathological changes across anatomical axes, ultimately achieving optimal aggregation of multi-level features through dynamic attention weights.

![image](https://github.com/Mercuriiio/PMFN-SSL/blob/main/figure/model.jpg)

### Prerequisites
- NVIDIA GPU (Tested on Nvidia GeForce RTX 4080)
- CUDA + cuDNN (Tested on CUDA 13.0 and cuDNN 9.2)
- torch>=2.12.0

### Referenced Repositories
- AutoCOPD: [AutoCOPD](https://github.com/DTyun/AutoCOPD)
- DenseNetWSO: [DenseNetWSO](https://github.com/bamos/densenet.pytorch)
- CNN: [CNN](https://github.com/msyim/VGG16)
- Conv3D: [Conv3D](https://github.com/GuangmingZhu/Conv3D_CLSTM)
- ResNet: [ResNet](https://github.com/KaimingHe/deep-residual-networks)
- SwinTrans: [SwinTrans](https://github.com/microsoft/Swin-Transformer)
- ViT: [ViT](https://github.com/jeonsworld/ViT-pytorch)
- ORACLE-CT: [ORACLE-CT](https://github.com/lavsendahal/oracle-ct)
- Lung-Nodule-SSM: [Lung-Nodule-SSM](https://github.com/EMeRALDsNRPU/Lung-Nodule-SSM-Self-Supervised-Lung-Nodule-Detection-and-Classification)

## Code Base Structure
The code base structure is explained below: 
- **1.transfer_learning.py**: Features were extracted from CT images using ResNet-50 and saved as .npy files.
- **2.multi_scale_image.py**: Scaling a standard 512*512 image to 128*128 and 256*256 resolution sizes.
- **3.train.py**: Model training and testing scripts.
- **4.plot_all_roc_and_cm.py**: Visualize the evaluation indicators.

## Data Preprocess
Raw CT images can be obtained from [CNCB](http://ncov-ai.big.ac.cn/download?lang=en). Run the following command to get multi-scale CT images.

```
python 1.transfer_learning.py
python 2.multi_scale_image.py
```

## Training and Evaluation

Train and test model with the following code.

```
python 3.train.py
```

## ROC curve
The ROC curve plots are implemented by running the following code.

```
python 4.plot_all_roc_and_cm.py
```
