# ECG_Heartbeat_classification
ECG_Heartbeat_classification.  
Data from Kaggle, https://www.kaggle.com/datasets/shayanfazeli/heartbeat.  
Data Content  
Arrhythmia Dataset  
Number of Samples: 109446  
Number of Categories: 5  
Sampling Frequency: 125Hz  
Data Source: Physionet's MIT-BIH Arrhythmia Dataset  
Classes: ['N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4]  
80% of data used to train, and 20% used to test.  
I use 5 different models to classify ECG data. All models use a stratified split and are trained with AdamW (learning rate 1e-3, weight decay 1e-3) for 100 epochs, with early stopping if the validation loss does not improve for 15 consecutive epochs.  
