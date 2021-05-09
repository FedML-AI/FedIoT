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

(1) Mean results:
Accuracy: 0.9902 Precision: 0.9963 False Positive Rate: 0.0848

(2) Details

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

## May 6th's update on Centralized Learning
Note: Each client shares the same super testset (10%).

Mean result:

Accuracy: 0.96044
Precision: 0.99689
FPR: 0.074927

(1) Danmini_Doorbell

Threshold: 0.54931

The True negative number is  15955  
The False positive number is  644
The True positive number is  468767
The False negative number is  1029

The accuracy is  0.99656
The precision is  0.99863
The false positive rate is  0.038798


(2) Ecobee_Thermostat

Threshold = 0.148883

The True negative number is  3730
The False positive number is  664
The True positive number is  462046
The False negative number is  7931

The accuracy is  0.98188
The precision is  0.99856
The false positive rate is  0.15112

(3) Ennio_Doorbell

Threshold = 0.21051

The True negative number is  12426
The False positive number is  673
The True positive number is  469848
The False negative number is  0

The accuracy is  0.99861
The precision is  0.99857
The false positive rate is  0.051378

(4) Philips_Baby_Monitor 

Threshold = 0.179716

The True negative number is  53861
The False positive number is  4845
The True positive number is  469161
The False negative number is  6

The accuracy is  0.99081
The precision is  0.98978
The false positive rate is  0.082530

(5) Provision_737E_Security_Camera

Threshold = 0.10321

The True negative number is  19739
The False positive number is  1083
The True positive number is  469733
The False negative number is  0

The accuracy is  0.99779
The precision is  0.99770
The false positive rate is  0.05201

(6) Provision_838_Security_Camera

Threshold = 0.09312

The True negative number is  31724
The False positive number is  1279
The True positive number is  469551
The False negative number is  0

The accuracy is  0.99745
The precision is  0.99728
The false positive rate is  0.038754

(7) Samsumg_SNH_Webcam

Threshold = 0.33131

The True negative number is  16346
The False positive number is  1125
The True positive number is  338116
The False negative number is  131667

The accuracy is  0.72747
The precision is  0.99668
The false positive rate is  0.064392

(8) SimpleHome_1002

Threshold: 0.13678

The True negative number is  13715
The False positive number is  1892
The True positive number is  450713
The False negative number is  19097

The accuracy is  0.956761
The precision is  0.99582
The false positive rate is  0.12123

(9) SimpleHome_1003

Threshold: 0.20081

The True negative number is  6058
The False positive number is  485
The True positive number is  468843
The False negative number is  1102

The accuracy is  0.99667
The precision is  0.99897
The false positive rate is  0.074125

## May 8th's update on Centralized Learning
Note: Each client shares the same super testset (1%).

Mean results:

Accuracy: 0.95517
Precision: 0.97001
FPR: 0.07619

Detrails: 

(1) Danmini_Doorbell

Threshold: 0.51858

The True negative number is  15964
The False positive number is  635
The True positive number is  46758
The False negative number is  0

The accuracy is  0.989977429486876
The precision is  0.9866013968307557
The false positive rate is  0.038255316585336464

(2) Ecobee_Thermostat

Threshold: 0.15074

The True negative number is  3676
The False positive number is  718
The True positive number is  46939
The False negative number is  0

The accuracy is  0.9860128961876375
The precision is  0.9849340075959461
The false positive rate is  0.16340464269458352

(3) Ennio_Doorbell

Threshold: 0.20323

The True negative number is  12442
The False positive number is  657
The True positive number is  46810
The False negative number is  0

The accuracy is  0.9890333672736984
The precision is  0.9861588050645712
The false positive rate is  0.050156500496221085

(4) Philips_Baby_Monitor

Threshold: 0.188204

The True negative number is  53965
The False positive number is  4741
The True positive number is  46129
The False negative number is  0

The accuracy is  0.9547765536318977
The precision is  0.9068016512679379
The false positive rate is  0.08075835519367697

(5) Provision_737E_Security_Camera

Threshold: 0.107350

The True negative number is  19753
The False positive number is  1069
The True positive number is  46695
The False negative number is  0

The accuracy is  0.9841669505457885
The precision is  0.9776191273762667
The false positive rate is  0.051339928921333204

(6) Provision_838_Security_Camera

Threshold: 0.093529

The True negative number is  31719
The False positive number is  1284
The True positive number is  46513
The False negative number is  0

The accuracy is  0.983852306454047
The precision is  0.9731363893131368
The false positive rate is  0.038905554040541766

(7) Samsung_SNH_Webcam

Threshold: 0.30623

The True negative number is  16269
The False positive number is  1202
The True positive number is  32701
The False negative number is  14044

The accuracy is  0.7625825339479257
The precision is  0.9645459103914108
The false positive rate is  0.06879972525900063

(8) SimpleHome_1002

Threshold: 0.12975

The True negative number is  13666
The False positive number is  1941
The True positive number is  46692
The False negative number is  80

The accuracy is  0.9676012760704724
The precision is  0.960088828573191
The false positive rate is  0.12436727109630294

(9) SimpleHome_1003

Threshold: 0.20414

The True negative number is  6087
The False positive number is  456
The True positive number is  46215
The False negative number is  692

The accuracy is  0.9785219831618335
The precision is  0.9902294786912644
The false positive rate is  0.06969280146721687

## May 9's update on centralized training

using training samples from b, g and h to train the model then test the performance on 9 devices

threshold = 0.867741

mean accuracy = 0.6456428384941383
mean precision = 0.45845565953575573
mean false positive rate = 0.245764731536723

(a) Danmini_Doorbell

The True negative number is  13165
The False positive number is  3187
The True positive number is  2095
The False negative number is  5612

The accuracy is  0.6342740762292697
The precision is  0.396630064369557
The false positive rate is  0.19489970645792565

(b) Ecobee_Thermostat

The True negative number is  2635
The False positive number is  1693
The True positive number is  1766
The False negative number is  4463

The accuracy is  0.41687979539641945
The precision is  0.5105521827117664
The false positive rate is  0.391173752310536

(c) Ennio_Doorbell

The True negative number is  10261
The False positive number is  2643
The True positive number is  1108
The False negative number is  0

The accuracy is  0.8113759634598915
The precision is  0.2953878965609171
The false positive rate is  0.20482021078735277

(d) Philips_Baby_Monitor

The True negative number is  38280
The False positive number is  19550
The True positive number is  2449
The False negative number is  4800

The accuracy is  0.6258393644647274
The precision is  0.11132324196554388
The false positive rate is  0.33805983053778316

(e) Provision_737E

The True negative number is  9481
The False positive number is  11031
The True positive number is  1727
The False negative number is  3848

The accuracy is  0.42963928393452677
The precision is  0.13536604483461356
The false positive rate is  0.5377827613104524

(f) Provision_838

The True negative number is  16775
The False positive number is  15736
The True positive number is  2157
The False negative number is  3286

The accuracy is  0.4988143542182642
The precision is  0.12054993572905605
The false positive rate is  0.484020792962382

(g) Samsung_Webcam

The True negative number is  16803
The False positive number is  408
The True positive number is  1146
The False negative number is  0

The accuracy is  0.9777741461023043
The precision is  0.7374517374517374
The false positive rate is  0.023705769565975247

(h) SimpleHome_1002

The True negative number is  15209
The False positive number is  165
The True positive number is  1500
The False negative number is  4738

The accuracy is  0.7731352952063668
The precision is  0.9009009009009009
The false positive rate is  0.010732405359698192

(i) SimpleHome_1003

The True negative number is  6273
The False positive number is  172
The True positive number is  1924
The False negative number is  4378

The accuracy is  0.643053267435475
The precision is  0.9179389312977099
The false positive rate is  0.02668735453840186


## May 9's update on centralized training

using device a, c and f 

threshold = 0.64176

mean accuracy = 0.8034269080114653
mean precision = 0.6404332131218307
mean false positive rate = 0.09346267287646917


(a) Danmini_Doorbell

The True negative number is  16084
The False positive number is  268
The True positive number is  3732
The False negative number is  3975

The accuracy is  0.8236418803774056
The precision is  0.933
The false positive rate is  0.016389432485322895

(b) Ecobee_Thermostat

The True negative number is  3254
The False positive number is  1074
The True positive number is  3017
The False negative number is  3212

The accuracy is  0.5940134507909444
The precision is  0.7374725006110975
The false positive rate is  0.24815157116451017

(c) Ennio_Doorbell

The True negative number is  11221
The False positive number is  1683
The True positive number is  330
The False negative number is  778

The accuracy is  0.8243648301455895
The precision is  0.16393442622950818
The false positive rate is  0.13042467451952883


(d) Philips_Baby_Monitor

The True negative number is  54157
The False positive number is  3673
The True positive number is  2927
The False negative number is  4322

The accuracy is  0.8771493108375974
The precision is  0.4434848484848485
The false positive rate is  0.06351374719003977


(e) Provision_737E

The True negative number is  20224
The False positive number is  288
The True positive number is  1759
The False negative number is  3816

The accuracy is  0.842680262199563
The precision is  0.8593063019052272
The false positive rate is  0.014040561622464899


(f) Provision_838

The True negative number is  32430
The False positive number is  81
The True positive number is  2073
The False negative number is  3370

The accuracy is  0.9090741423828845
The precision is  0.9623955431754875
The false positive rate is  0.002491464427424564


(g) Samsung_Webcam

The True negative number is  13560
The False positive number is  3651
The True positive number is  285
The False negative number is  861

The accuracy is  0.7542082039548946
The precision is  0.07240853658536585
The false positive rate is  0.21213177618964615


(h) SimpleHome_1002

The True negative number is  13674
The False positive number is  1700
The True positive number is  3189
The False negative number is  3049

The accuracy is  0.7802609661299278
The precision is  0.6522806299856821
The false positive rate is  0.11057629764537531


(i) SimpleHome_1003

The True negative number is  6165
The False positive number is  280
The True positive number is  4357
The False negative number is  1945

The accuracy is  0.8254491252843806
The precision is  0.9396161311192581
The false positive rate is  0.04344453064391001
