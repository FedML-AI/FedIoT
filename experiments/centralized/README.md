## Experimental Tracking
https://wandb.ai/automl/fediot

```
pip install --upgrade wandb
wandb login ee0b5f53d949c84cee7decbe7a629e63fb2f8408
```

Apr 10's update on Centralized Training:

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

