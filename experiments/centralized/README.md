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
FPR: 0.2576 / 90% length

(2) Ecobee_Thermostat

Threshold: 0.05297
TN：3195
FP: 1199
TP: 622884
FN: 0
Accuracy: 0.9981
Precision: 0.9981
FPR: 0.2729 / 90% length

(3) Ennio_Doorbell   

Threshold: 0.02221
TN：10070
FP: 3029
TP: 110735
FN: 0
Accuracy: 0.9756
Precision: 0.9733
FPR: 0.2312 / 90% length

(4) Philips_Baby_Monitor

Threshold: 0.02026
TN：14560
FP: 17146
TP: 724197
FN: 0
Accuracy: 0.9781
Precision: 0.9769
FPR: 0.2920 / 90% length

(5) Provision_737E_Sevcurity_Camera

Threshold: 0.00756
TN：14691
FP: 6131
TP: 557274
FN: 0
Accuracy: 0.9894
Precision: 0.9891
FPR: 0.2944 / 90% length

(6) Provision_838_Sevcurity_Camera

Threshold: 0.0133
TN：20281
FP: 12722
TP: 543839
FN: 0
Accuracy: 0.9780
Precision: 0.9771
FPR: 0.3855 / 90% length

(7) Samsung_SNH_Webcam

Threshold: 0.02102
TN：13590
FP: 3881
TP: 114411
FN: 0
Accuracy: 0.9706
Precision: 0.9672
FPR: 0.2221 / 90% length

(8) SimpleHome_1002_Security_Camera

Threshold: 0.02577
TN：11901
FP: 3706
TP: 623701
FN: 0
Accuracy: 0.9942
Precision: 0.9941
FPR: 0.2375 / 90% length

(9) SimpleHome_1003_Security_Camera

Threshold: 0.01779
TN：5277
FP: 1266
TP: 630144
FN: 0
Accuracy: 0.9980
Precision: 0.9980
FPR: 0.1935 / 90% length

## Apr 13's update on centralized training using the 9 saparated datasets 
We remain the whole original data as the input instead of eliminating the largest 10% MSE

(1) Danmini Doorbell
Threshold 0.35605
Accuracy 0.999215
Precision  0.999198
False Positive Rate 0.03723 / 100% length (v.s. 0.257606 in 90% length)

(2) Ecobee_Thermostat
Threshold 0.105233
Accuracy 0.99894
Precision 0.99893
False Positive Rate 0.15134 / 100% length (v.s. 0.27287 in 90% length)

(3) Ennio Doorbell
Threshold 0.12339
Accuracy 0.99409
Precision 0.99343
False Positive Rate 0.05588 / 100% length (v.s. 0.23124 in 90% length)

(4) Philips_Baby_Monitor
Threshold 0.103674
Accuracy 0.99272
Precision 0.99219
False Positive Rate 0.097077 / 100% length (v.s. 0.29206 in 90% length)

(5) Provision_737E_Sevcurity_Camera
Threshold 0.04451
Accuracy 0.997538
Precision 0.99745
False Positive Rate 0.06834 / 100% length (v.s. 0.2944 in 90% length)

(6) Provision_838_Sevcurity_Camera
Threshold 0.04382
Accuracy 0.9957
Precision 0.9954
False Positive Rate: 0.07593 / 100% length (v.s. 0.38548 in 90% length)

(7) Samsung_SNH_Webcam

Threshold: 0.18166
Accuracy: 0.99033
Precision: 0.98898
False Positive Rate: 0.072978 / 100% length (v.s. 0.2221 / 90% length)

(8) SimpleHome_1002_Security_Camera

Threshold: 0.066662
Accuracy: 0.99697
Precision: 0.99691
FPR: 0.12398 / 100% length (v.s. 0.2375 / 90% length)

(9) SimpleHome_1002_Security_Camera

Threshold: 0.147692
Accuracy: 0.99917
Precision: 0.99920
FPR: 0.077182 / 100% length (v.s. 0.1935 / 90% length)


## Apr 24's update on Centralized Learning
Note: Each client shares the same super testset.


a. Danmini Doorbell

TN number: 15991 FP number: 608 TP number: 469796 FN number: 0

Accuracy: 0.99874 Precision: 0.9987 False Positive Rate: 0.0366



b. Ecobee_Thermostat

TN number: 3705 FP number: 689 TP number: 469977 FN number: 0

Accuracy: 0.99854 Precision: 0.99853 False Positive Rate: 0.156804


c. Ennio Doorbell

TN number: 12349 FP number: 750 TP number: 469848 FN number: 0

Accuracy: 0.99844 Precision: 0.99840 False Positive Rate: 0.057



d. Philips_Baby_Monitor

TN number: 52515 FP number: 6191 TP number: 469167 FN number: 0

Accuracy: 0.98827 Precision: 0.98697 False Positive Rate: 0.10545


e. Provision_737E_Security_Camera

TN number: 19319 FP number: 1503 TP number: 469733 FN number: 0

Accuracy: 0.99693 Precision: 0.9968 False Positive Rate: 0.0721


f. Provision_838_Security_Camera

TN number: 30610  FP number: 2393  TP number: 469551  FN number: 0

Accuracy: 0.995238 Precision: 0.994929 False Positive Rate: 0.0725


g. Samsung_SNH_Webcam

TN number: 16223 FP number: 1248 TP number: 442103 FN number: 27680

Accuracy: 0.94063 Precision: 0.99718 False Positive Rate: 0.0714


h. SimpleHome_1002_Security_Camera

TN number: 13747 FP number: 1860  TP number: 469810 FN number: 0

Accuracy: 0.99616  Precision: 0.99605 False Positive Rate: 0.1192



i. SimpleHome_1002_Security_Camera

TN number: 6069  FP number: 474 TP number: 469945 FN number: 0

Accuracy: 0.9990 Precision: 0.9989 False Positive Rate: 0.0724
