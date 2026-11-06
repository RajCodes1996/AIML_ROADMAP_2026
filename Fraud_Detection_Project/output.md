# RESULTS DAY-21

Loading Credit Card Fraud Dataset...
Dataset Shape: (284807, 30)

First 5 Rows:
         V1        V2        V3        V4        V5        V6        V7  ...       V24       V25       V26       V27       V28  Amount Class
0 -1.359807 -0.072781  2.536347  1.378155 -0.338321  0.462388  0.239599  ...  0.066928  0.128539 -0.189115  0.133558 -0.021053  149.62     0
1  1.191857  0.266151  0.166480  0.448154  0.060018 -0.082361 -0.078803  ... -0.339846  0.167170  0.125895 -0.008983  0.014724    2.69     0
2 -1.358354 -1.340163  1.773209  0.379780 -0.503198  1.800499  0.791461  ... -0.689281 -0.327642 -0.139097 -0.055353 -0.059752  378.66     0
3 -0.966272 -0.185226  1.792993 -0.863291 -0.010309  1.247203  0.237609  ... -1.175575  0.647376 -0.221929  0.062723  0.061458  123.50     0
4 -1.158233  0.877737  1.548718  0.403034 -0.407193  0.095921  0.592941  ...  0.141267 -0.206010  0.502292  0.219422  0.215153   69.99     0

[5 rows x 30 columns]

Class Distribution:
Class
0    284315
1       492
Name: count, dtype: int64

Features Shape: (284807, 29)
Target Shape: (284807,)

Data Split:
Training: (182276, 29)
Validation: (45569, 29)
Testing: (56962, 29)

Training Model...
Model Training Completed!

VALIDATION RESULTS
Accuracy: 0.9751146612828897

Confusion Matrix:
[[44365  1125]
 [    9    70]]

Classification Report:
              precision    recall  f1-score   support

  Legitimate       1.00      0.98      0.99     45490
       Fraud       0.06      0.89      0.11        79

    accuracy                           0.98     45569
   macro avg       0.53      0.93      0.55     45569
weighted avg       1.00      0.98      0.99     45569


SELECTED THRESHOLD
Threshold: 0.9500000000000002
Validation Precision: 0.32160804020100503
Validation Recall: 0.810126582278481
Validation F1: 0.460431654676259

FINAL TEST CONFUSION MATRIX
[[56687   177]
 [   11    87]]

FINAL TEST CLASSIFICATION REPORT
              precision    recall  f1-score   support

  Legitimate       1.00      1.00      1.00     56864
       Fraud       0.33      0.89      0.48        98

    accuracy                           1.00     56962
   macro avg       0.66      0.94      0.74     56962
weighted avg       1.00      1.00      1.00     56962


Model Saved Successfully!
Location: C:\Users\netware\Desktop\Fraud_Detection_Project\models\fraud_detection_v1.joblib
Evaluation Report Saved!
Location: C:\Users\netware\Desktop\Fraud_Detection_Project\reports\evaluation.txt

SAMPLE PREDICTION
Fraud Probability: 0.006313061303374749
Prediction: Legitimate

DAY 21 PROJECT EXECUTION COMPLETED!

"ADD-UI" !