# ECG_Heartbeat_classification
ECG_Heartbeat_classification.  
##Data
Data from Kaggle, https://www.kaggle.com/datasets/shayanfazeli/heartbeat  
Data Content  
Arrhythmia Dataset  
Number of Samples: 109446  
Number of Categories: 5  
Sampling Frequency: 125Hz  
Data Source: Physionet's MIT-BIH Arrhythmia Dataset  
Classes: ['N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4]  

80% of data(87554) used to train, and 20% of data (21892) used to test.  
![train_pie](train_pie.png)  
![test_pie](test_pie.png)  
As shown in the pie chart, class N accounts for 82.8% of the data, meaning that most samples belong to class N. The other four classes make up only 17.2%, with class F in particular representing just 0.7%. This indicates that the dataset is highly imbalanced.  
##Data Preprocessing
As mentioned earlier, the dataset is highly imbalanced. Therefore, when splitting a portion of the training data for validation, I applied a stratified split to preserve the class distribution.  
In addition, all data were standardized using z-score normalization.  
##Model 
I use 4 different models to classify ECG data. All models are trained with AdamW (learning rate 1e-3, weight decay 1e-3) for 100 epochs, with early stopping if the validation loss does not improve for 15 consecutive epochs.  
1.**Simple CNN**
The first model is a simple CNN with only three layers and up to 128 channels. It does not use dropout or batch normalization. Training is relatively fast, and this model serves as the baseline.

2.**Four-layer CNN**
The second model consists of four convolutional layers with up to 256 channels. Both dropout and batch normalization are applied.

3.**Residual CNN**
The third model incorporates residual connections and consists of five residual blocks (i.e., ten layers). Dropout and batch normalization are applied. This model has a larger number of parameters and produces larger feature maps, requiring more GPU memory.

4.**Coupled CNN**
The fourth model is a coupled CNN consisting of two convolution + pooling pairs, repeated twice, followed by three fully connected layers for output. This design is based on an architecture from the literature, which I re-implemented. It has more parameters than the residual CNN, but its feature maps are smaller.
