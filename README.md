# Computer Parts Classifier (CPC)

 Using imagenet.py in Jetson Inference, the CPC classifies 7 computer hardware components: 
 
 CPU, GPU, HDD (hard disk drive), MoBo (motherboard), RAM, SSD-M2, SSD-SATA.

Example:

![cpu_input](https://i.imgur.com/m2HzXP9.png)
![cpu_output](https://i.imgur.com/mlqGTDw.png)
![screenshot_result](https://i.imgur.com/txfDgFc.png)


## The Algorithm
The **CPC** uses **imagenet.py** in the **Jetson Inference library**.

I trained the **Resnet18** model on **Google Colab** using a dataset from **Kaggle**: https://www.kaggle.com/datasets/kevinvtcosta/computer-hardware-pics?resource=download

I used **set-1** (260 images per label, 1820 total) and 35 epochs.
Link to training the model on **Colab**: https://colab.research.google.com/drive/18p0dAhCEFJ5bHo6yzkRz-6fhg-Q0y6HH?usp=sharing

## Running this project

1. Login to **Jetson Nano** on **VSCode**
2. Train the model using the **Colab** link above
3. Export the **model_best.pth.tar** and **labels.txt** to **Jetson Nano** under  models/parts
4. To run the program use this command (change the code to run other images):
```
imagenet.py --model=models/parts/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=models/parts/labels.txt data/parts/test/cpu/IMG_20220209_181248.png data/parts/result/cpu_test_02.png
```
5. To evaluate the accuracy of the model,  run this cell:
```
python3 script.py data/parts/test data/parts/result
```

6. Libraries used include: **Pytorch**, **Torchvision**, and **Jetson Inference**. 

[View a video explanation here](https://youtu.be/OG5Q48jLmcg)
