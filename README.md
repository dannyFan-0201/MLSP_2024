# [ICIP2024][MICRO-EXPRESSION RECOGNITION BASED ON 3DCNN COMBINED WITH GRU AND NEW ATTENTION MECHANISM]

## Chun-Ting Fang, Tsung-Jung Liu, Kuan-Hsien Liu  

***
> Abstract : Micro-expression, as a form of non-verbal emotional expression,play a key role in interpersonal interaction. However,
they are also quite challenging and not easy to analyze. In this paper, we propose a dual-branch shallow 3DCNN architecture
that combines GRU (Gated Recurrent Unit) and enhances the CAM (Channel Attention Module) in CBAM (Convolutional Block Attention Module)
> to make it more suitable for recognizing micro facial expressions. Experiments show that the
proposed method can achieve good results with a relatively simple architecture.


## Network Architecture  

<table>
  <tr>
    <td colspan="2">
  <img src = "https://github.com/dannyFan-0201/ICIP_2024/blob/main/img/architecture.PNG" alt="CMFNet" width="800"> </td>  
  </tr>
  </table>
  <img src = "https://github.com/dannyFan-0201/ICIP_2024/blob/main/img/CBAM.PNG" alt="CMFNet" width="500">


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

<img src="https://github.com/dannyFan-0201/ICIP_2024/blob/main/img/performance.PNG" width="1000" height="250">

Both LOSO and MEGC2019 are employed for performance comparison between our proposed method and the
State-of-the-Art (SOTA) method in terms of UF1 and UAR. The best and second-best scores are highlighted and underlined,
respectively.
All training and testing base on same 1080Ti.

## Ablation study

- In the ablation experiment on SMIC, the effectiveness of weighted-Categorical Cross-Entropy and New Channel
  Attention was tested and evaluated, respectively. The basic architecture, 3DCNN+GRU, utilizes Categorical Cross-Entropy (CCE) as the loss function.
  
  <img src="https://github.com/dannyFan-0201/ICIP_2024/blob/main/img/ab.PNG" width="500" height="100">

---
## Contact
If you have any question, feel free to contact danny80351@gmail.com

## Citation
```

