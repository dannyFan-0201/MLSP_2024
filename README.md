# [MLSP2024][ENHANCED FEATURE EXTRACTION AND ATTENTION MECHANISMS FOR MICRO-EXPRESSION RECOGNITION]

## Chun-Ting Fang, Tsung-Jung Liu, Kuan-Hsien Liu  

***
> Abstract : Micro-expressions refer to subtle changes in expressions that are displayed by humans in a very short period of time. As a
form of non-verbal emotional expression, micro-expressions can more accurately reflect an individual’s true inner feelings.
We propose a shallow dual-branch 3DCNN architecture for the first stage of feature extraction. Additionally, we enhance
and optimize the channel attention module (CAM) within the convolutional block attention module (CBAM). We then
employ Gated Recurrent Unit (GRU) and Multi-Scale Multi-Head (MSMH) Self-Attention for the second stage of feature
extraction.Tested on 3 different databases and obtained competitive scores and effects. Numerous experimental results
show that our method can achieve good results with relatively simple input and architecture.


## Network Architecture  

<table>
  <tr>
    <td colspan="2">
  <img src = "https://github.com/dannyFan-0201/MLSP_2024/blob/main/img/architecture.PNG" alt="CMFNet" width="800"> </td>  
  </tr>
  </table>


# Environment
- Python 3.9.0
- Tensorflow 2.10.1
- keras	2.10.0
- opencv-python	
- tensorboard	
- colorama
  
or see the requirements.txt

# How to try

## Download dataset (Most datasets require an application to download)
[SMIC] [SAMM] [CASME II]

## Set dataset path

Edit in ME_model(set path in config)

```python
output_folder ='./data/negative/training_frames' # This will be automatically generated.
negativepath = './data/negative/negative_video'
positivepath = './data/negative/positive_video'
surprisepath = './data/negative/surprise_video'
excel_file_path = "/excel_file.xlsx"

```

## Parameter settings

```python
for video in directorylisting:
  .....
  -framerange = [x + 0 for x in range(30)]# Select the frame number range to enter.
  .....

class_weights = [0.3, 0.35, 0.35] # Choose the weight value you want to give(negative/positive/surprise).
loss = weighted_categorical_crossentropy(class_weights) # Choose the loss function to use.
hist = model.fit(train_images, train_labels, validation_data=(validation_images, validation_labels), callbacks=callbacks_list, batch_size=8, epochs=200, shuffle=True)
# Batch_size and epochs can be adjusted by yourself.

```

## Run training
```python

python ME_model.py 

```
1. Set the dataset path.
2. Set path and parameter details in model.
   
## Performance Evaluation

- MEGC2019 [SMIC Part] [SAMM Part] [CASME II Part]

<img src="https://github.com/dannyFan-0201/MLSP_2024/blob/main/img/performance.PNG" width="1000" height="250">

Both LOSO and MEGC2019 are employed for performance comparison between our proposed method and the
State-of-the-Art (SOTA) method in terms of UF1 and UAR. The best and second-best scores are highlighted and underlined,
respectively.
All training and testing base on same 1080Ti.

## Ablation study

- In the ablation experiment on SMIC, the effectiveness of weighted-Categorical Cross-Entropy and New Channel
  Attention was tested and evaluated, respectively. The basic architecture, 3DCNN+GRU, utilizes Categorical Cross-Entropy (CCE) as the loss function.
  
  <img src="https://github.com/dannyFan-0201/MLSP_2024/blob/main/img/ab.PNG" width="500" height="150">
  
- The visualization of loss curves using Categorical Cross-Entropy on SMIC dataset.
  <img src="https://github.com/dannyFan-0201/MLSP_2024/blob/main/img/CCE.PNG" width="500" height="250">
  
- The visualization of loss curves using Weighted Categorical Cross-Entropy on SMIC dataset.
  <img src="https://github.com/dannyFan-0201/MLSP_2024/blob/main/img/WCCE.PNG" width="500" height="250">
---
## Contact
If you have any question, feel free to contact danny80351@gmail.com

## Citation
```

