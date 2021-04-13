## Experimental Tracking
https://wandb.ai/automl/fediot

```
pip install --upgrade wandb
wandb login ee0b5f53d949c84cee7decbe7a629e63fb2f8408
```

## Apr 10's update on Centralized Training:

We train the 9 device sepreately and split each datasets into trainset, optimization set and test set. 
Trainset is made by 2/3 benign data; optimization set is made by the rest benign data, and test set is made by optimization set combined with all attack data.
Trainset is used for model training; optimization is used for settling down the threshold; test set is used for calculate the model's accuracy.

The result is shown below. If the accuracy is greater than one, it represents that there has false alarm. The length parameter in the tr represents the persentage of data we select for.

Danmini:

Accuracy: 1.005
Test length: 787086
Detected Anomaly Number: 774831
Threshold: 0.11361 / 90% length


Ecobee:

Accuracy: 0.827
Test length: 627278
Detected Anomaly Number: 515195
Threshold: 0.22261 / 90% length

Ennio:

Accuracy: 0.47
Test length: 123834
Detected Anomaly Number: 52231
Threshold: 0.10018 / 90% length

Baby Monitor:

Accuracy: 1.017
Test length: 782903
Detected Anomaly Number: 736657
Threshold: 0.0256 / 90% length

737 Camera:

Accuracy: 0.7747
Test length: 578096
Detected Anomaly Number: 431672
Threshold: 0.2475 / 90% length

838 Camera:

Accuracy: 0.75488
Test length: 576842
Detected Anomaly Number: 410534
Threshold: 0.20843 / 90% length

SNH 1011:

Accuracy: 1.04
Test length: 131882
Detected Anomaly Number: 119106
Threshold: 0.04996 / 90% length

1002 Camera:

Accuracy: 0.908
Test length: 639308
Detected Anomaly Number: 566376
Threshold: 0.20283 / 90% length

1003 Camera:

Accuracy: 1.0015
Test length: 636687
Detected Anomaly Number: 641149
Threshold: 0.02576 / 90% length

## Apr 11's update on centralized training using the unified dataset

Accuracy: 1.0045
Test length: 4883916
Detected Anomaly Number: 4718612
Threshold: 0.25706 / 90% length

## Apr 13's update on centralized training using the 9 saparated datasets

We train the 9 device sepreately and split each datasets into trainset, optimization set and test set. 
Trainset is made by 2/3 benign data; optimization set is made by the rest benign data, and test set is made by optimization set combined with all attack data.
Trainset is used for model training; optimization set is used for settling down the threshold; test set is used for calculate the model's accuracy, fpr etc.

(1) Danmini_Doorbell   

Threshold: 0.01836
TN：12323
FP: 4276
TP: 770487
FN: 0
Accuracy: 0.9946
Precision: 0.9945
FPR: 0.2576

(2) Ecobee_Thermostat

Threshold: 0.05297
TN：3195
FP: 1199
TP: 622884
FN: 0
Accuracy: 0.9981
Precision: 0.9981
FPR: 0.2729

(3) Ennio_Doorbell   

Threshold: 0.02221
TN：10070
FP: 3029
TP: 110735
FN: 0
Accuracy: 0.9756
Precision: 0.9733
FPR: 0.2312

(4) Philips_Baby_Monitor

Threshold: 0.02026
TN：14560
FP: 17146
TP: 724197
FN: 0
Accuracy: 0.9781
Precision: 0.9769
FPR: 0.2920

(5) Provision_737E_Sevcurity_Camera

Threshold: 0.00756
TN：14691
FP: 6131
TP: 557274
FN: 0
Accuracy: 0.9894
Precision: 0.9891
FPR: 0.2944

(6) Provision_838_Sevcurity_Camera

Threshold: 0.0133
TN：20281
FP: 12722
TP: 543839
FN: 0
Accuracy: 0.9780
Precision: 0.9771
FPR: 0.3855

(7) Samsung_SNH_Webcam

Threshold: 0.02102
TN：13590
FP: 3881
TP: 114411
FN: 0
Accuracy: 0.9706
Precision: 0.9672
FPR: 0.2221

(8) SimpleHome_1002_Security_Camera

Threshold: 0.02577
TN：11901
FP: 3706
TP: 623701
FN: 0
Accuracy: 0.9942
Precision: 0.9941
FPR: 0.2375

(9) SimpleHome_1003_Security_Camera

Threshold: 0.01779
TN：5277
FP: 1266
TP: 630144
FN: 0
Accuracy: 0.9980
Precision: 0.9980
FPR: 0.1935
