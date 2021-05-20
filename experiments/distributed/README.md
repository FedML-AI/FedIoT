## Installation
http://doc.fedml.ai/#/installation-distributed-computing

## Experimental Tracking
```
pip install --upgrade wandb
wandb login ee0b5f53d949c84cee7decbe7a629e63fb2f8408
```

## Run Experiments
```
sh run_fedavg_distributed.sh 20 4 vae 50 1 32 0.0001
```
# Experiment Results
## Apr 19's update on Federated Learning: training using the 9 separated datasets with 10% data:

Mean Result:
Accuracy: 0.967479
Precision: 0.999
False Positive Rate: 0.427746

## Apr 22's update on Federated Learning: training using the 9 separated datasets with 50% data:
Note: Each client shares the same threshold value computed by central server.

(1) Mean Result:

Global threshold: 0.092956

Accuracy: 0.963542
Precision: 0.999716
False Positive Rate: 0.307094


(2) Detail Result (client-wise):

a. Danmini Doorbell

Accuracy:0.979501
Precision:1.00
False Positive Rate:0.000

b. Ecobee_Thermostat

Accuracy: 0.998068
Precision: 0.999923
False Positive Rate: 0.685714 

c. Ennio Doorbell

Accuracy: 0.907717
Precision: 0.999840
False Positive Rate: 0.087379 

e. Philips_Baby_Monitor

Accuracy: 0.931966
Precision: 0.999663
False Positive Rate: 0. 267974

d. Provision_737E_Security_Camera

Accuracy: 0.973180
Precision: 0.999673
False Positive Rate: 0.564417 

f. Provision_838_Security_Camera

Accuracy: 0.963966
Precision: 0.999986
False Positive Rate: 0.015504 

g. Samsung_SNH_Webcam

Accuracy: 0.944524
Precision: 0.998542
False Positive Rate: 0.664234 

h. SimpleHome_1002_Security_Camera

Accuracy: 0.982213
Precision: 0.999822
False Positive Rate: 0.459016

i. SimpleHome_1002_Security_Camera

Accuracy: 0.990744
Precision: 0.999997
False Positive Rate: 0.019608

## Apr 23's update on Federated Learning: training using the 9 separated datasets with 10% data:
Note: Each client calculates its own local threshold

(1) Mean Result:

Accuracy: 0.979725
Precision: 0.999436
FPR: 0.742984


(2) Detail Result (client-wise):

a. Danmini Doorbell

Local Threshold: 0.001383

Accuracy:0.979501
Precision:0.999670
False Positive Rate:1.0

b. Ecobee_Thermostat

Local Threshold: 0.031324

Accuracy: 0.998055
Precision: 0.999904
False Positive Rate: 0.857143 

c. Ennio Doorbell

Local Threshold: 0.013218

Accuracy: 0.958051
Precision: 0.998822
False Positive Rate: 0.0.666667

d. Philips_Baby_Monitor

Local Threshold: 0.131865

Accuracy: 0.924555
Precision: 0.999793
False Positive Rate: 0.163043

e. Provision_737E_Security_Camera

Local Threshold: 0.035064

Accuracy: 0.0.975601
Precision: 0.999415
False Positive Rate: 1.0

f. Provision_838_Security_Camera

Local Threshold: 0.043778

Accuracy: 0.968900
Precision: 1.0
False Positive Rate: 0.0

g. Samsung_SNH_Webcam

Local Threshold: 0.010397

Accuracy: 0.993260
Precision: 0.997870
False Positive Rate: 1.0 

h. SimpleHome_1002_Security_Camera

Local Threshold: 0.014768

Accuracy: 0.999593
Precision: 0.999609
False Positive Rate: 1.0

i. SimpleHome_1002_Security_Camera

Local Threshold: 0.009147

Accuracy: 0.999843
Precision: 0.999843
False Positive Rate: 1.0

## Apr 23's update on Federated Learning: using all training samples and 10% of test data:
Note: Each client shares the same threshold value computed by central server.

(1) Mean Result:

Global Threshold: 0.24026
Accuracy: 0.703906
Precision: 0.999843
FPR: 0.039062


(2) Detail Result (client-wise):

a. Danmini Doorbell

TN number: 260
FP number: 0
TP number: 76382
FN number: 16947

Accuracy: 0.818921
Precision: 1.0
False Positive Rate: 0.0

b. Ecobee_Thermostat

TN number: 56
FP number: 13
TP number: 63072
FN number: 3524

Accuracy: 0.946944
Precision: 0.999794
False Positive Rate: 0.188406

c. Ennio Doorbell

TN number: 205
FP number: 0
TP number: 11258
FN number: 12663

Accuracy: 0.475131
Precision: 1.0
False Positive Rate: 0.0

d. Philips_Baby_Monitor

TN number: 918
FP number: 0
TP number: 74885
FN number: 55117

Accuracy: 0.579002
Precision: 1.0
False Positive Rate: 0.0

e. Provision_737E_Security_Camera

TN number: 320
FP number: 6
TP number: 53693
FN number: 22458

Accuracy: 0.706265
Precision: 0.999888
False Positive Rate: 0.018405

f. Provision_838_Security_Camera

TN number: 462
FP number: 54
TP number: 54943
FN number: 31813

Accuracy: 0.634854
Precision: 0.999018
False Positive Rate: 0.104651

g. Samsung_SNH_Webcam

TN number: 273
FP number: 0
TP number: 12562
FN number: 16015

Accuracy: 0.444887
Precision: 1.0
False Positive Rate: 0.0

h. SimpleHome_1002_Security_Camera

TN number: 239
FP number: 5
TP number: 63839
FN number: 13838

Accuracy: 0.822346
Precision: 0.999922
False Positive Rate: 0.020492

i. SimpleHome_1002_Security_Camera

TN number: 100
FP number: 2
TP number: 62953
FN number: 6478

Accuracy: 0.906807
Precision: 0.999968
False Positive Rate: 0.019608

## Apr 30's update on Federated Learning: using all training samples and 50% of test data:
Note: Each client calculated its own local threshold

30 Epochs

(1) Mean Result:

Accuracy: 0.794362
Precision: 0.999900
FPR: 0.096573


(2) Detail Result (client-wise):

a. Danmini Doorbell

Local threshold: 0.104592
TN number: 260
FP number: 0
TP number: 385608
FN number: 16014

Accuracy: 0.960152
Precision: 1.0
False Positive Rate: 0.0

b. Ecobee Thermostat

Local threshold: 0.306545
TN number: 43
FP number: 23
TP number: 295972
FN number: 19805

Accuracy: 0.937213
Precision: 0.999912
False Positive Rate: 0.376812

c. Ennio Doorbell

Local threshold: 0.114706
TN number: 193
FP number: 12
TP number: 56543
FN number: 11750

Accuracy: 0.828287
Precision: 0.999788
False Positive Rate: 0.058537

d. Philips_Baby_Monitor

Local threshold: 0.284898
TN number: 917
FP number: 1
TP number: 352138
FN number: 67894

Accuracy: 0.838710
Precision: 0.999997
False Positive Rate: 0.001089

e. Provision_737E_Security_Camera

Local threshold: 0.133146
TN number: 259
FP number: 67
TP number: 282725
FN number: 16458

Accuracy: 0.944826
Precision: 0.999763
False Positive Rate: 0.205521

f. Provision_838_Security_Camera

Local threshold: 0.147193
TN number: 419
FP number: 97
TP number: 279733
FN number: 24754

Accuracy: 0.918522
Precision: 0.999653
False Positive Rate: 0.187984

g. Samsung_SNH_Webcam

Local threshold: 0.301008
TN number: 273
FP number: 0
TP number: 57711
FN number: 16734

Accuracy: 0.776038
Precision: 1.0
False Positive Rate: 0.0

h. SimpleHome_1002_Security_Camera

Local threshold = 2.181594
TN number: 244
FP number: 0
TP number: 4697
FN number: 322556

Accuracy: 0.015087
Precision: 1.0
False Positive Rate: 0

i. SimpleHome_1002_Security_Camera

Local threshold: 0.314181
TN number: 98
FP number: 4
TP number: 299155
FN number: 22374

Accuracy: 0.930423
Precision: 0.999987
False Positive Rate: 0.039216

## May 1st's update on Federated Learning: using all training samples and 10% of test data:
Note: Each client shares the same threshold value computed by central server.

(1) Mean Result:

Global Threshold: 0.234620
Accuracy: 0.929921
Precision: 0.971453
FPR: 0.078795
lr: 0.0001

(2) Detail Result (client-wise):

a. Danmini Doorbell

TN number: 16505
FP number: 94
TP number: 68761
FN number: 8229

Accuracy: 0.911
Precision: 0.99
False Positive Rate: 0.005

b. Ecobee_Thermostat

TN number: 2988
FP number: 1405
TP number: 55318
FN number: 6954

Accuracy: 0.874
Precision: 0.975
False Positive Rate: 0.31

c. Ennio Doorbell

TN number: 13022
FP number: 77
TP number: 11026
FN number: 1

Accuracy: 0.996
Precision: 0.993
False Positive Rate: 0.005

d. Philips_Baby_Monitor

TN number: 56819
FP number: 1887
TP number: 66604
FN number: 5610

Accuracy: 0.943
Precision: 0.972
False Positive Rate: 0.032

e. Provision_737E_Security_Camera

TN number: 19461
FP number: 1361
TP number: 52843
FN number: 2812

Accuracy: 0.945
Precision: 0.9748
False Positive Rate: 0.065

f. Provision_838_Security_Camera

TN number: 29519
FP number: 3483
TP number: 51675
FN number: 2595

Accuracy: 0.930
Precision: 0.936
False Positive Rate: 0.1055

g. Samsung_SNH_Webcam

TN number: 16489
FP number: 981
TP number: 11380
FN number: 0

Accuracy: 0.9659
Precision: 0.9206
False Positive Rate: 0.056

h. SimpleHome_1002_Security_Camera

TN number: 14199
FP number: 1407
TP number: 53748
FN number: 8567

Accuracy: 0.871
Precision: 0.974
False Positive Rate: 0.09

i. SimpleHome_1002_Security_Camera

TN number: 6356
FP number: 186
TP number: 58332
FN number: 4659

Accuracy: 0.9303
Precision: 0.9968
False Positive Rate: 0.028


## May 3rd's update on Federated Learning: using all training samples and 10% of test data:
Note: Each client shares the same threshold value computed by central server.

batch size = 64, lr = 1e-3, round = 50, epoch = 15

(1) Mean Result:

Global Threshold: 0.263881
Accuracy: 0.929010
Precision: 0.969896
FPR: 0.071560

(2) Detail Result (client-wise):

a. Danmini Doorbell

TN number: 16517
FP number: 82
TP number: 68868
FN number: 8122

Accuracy: 0.912
Precision: 0.999
False Positive Rate: 0.0049

b. Ecobee_Thermostat

TN number: 3565
FP number: 828
TP number: 56937
FN number: 5335

Accuracy: 0.908
Precision: 0.986
False Positive Rate: 0.188

c. Ennio Doorbell

TN number: 13028
FP number: 71
TP number: 10840
FN number: 187

Accuracy: 0.989
Precision: 0.993
False Positive Rate: 0.005

d. Philips_Baby_Monitor

TN number: 56993
FP number: 1713
TP number: 64826
FN number: 7388

Accuracy: 0.930
Precision: 0.974
False Positive Rate: 0.029

e. Provision_737E_Security_Camera

TN number: 18543
FP number: 2279
TP number: 51658
FN number: 3887

Accuracy: 0.918
Precision: 0.958
False Positive Rate: 0.109

f. Provision_838_Security_Camera

TN number: 28558
FP number: 4444
TP number: 50384
FN number: 3886

Accuracy: 0.905
Precision: 0.919
False Positive Rate: 0.1346

g. Samsung_SNH_Webcam

TN number: 16598
FP number: 872
TP number: 11158
FN number: 222

Accuracy: 0.962
Precision: 0.928
False Positive Rate: 0.0499

h. SimpleHome_1002_Security_Camera

TN number: 14227
FP number: 1379
TP number: 57120
FN number: 5195

Accuracy: 0.916
Precision: 0.976
False Positive Rate: 0.088

i. SimpleHome_1002_Security_Camera

TN number: 6322
FP number: 220
TP number: 57732
FN number: 5259

Accuracy: 0.921
Precision: 0.996
False Positive Rate: 0.034

## May 5th's update on Federated Learning: using all training samples and 10% of test data:
Note: Each client calculate its own local threshold
batch size = 64, lr = 1e-3, round = 50, epoch = 15

(1) Mean Result:

Accuracy: 0.868
Precision: 0.914
FPR: 0.147

(2) Detail Result (client-wise):

a. Danmini Doorbell

Local threshold: 0.006024
TN number: 15926
FP number:673
TP number: 76990
FN number: 0

Accuracy: 0.993
Precision: 0.991
False Positive Rate: 0.041

b. Ecobee_Thermostat

Local threshold: 0.227607
TN number: 3156
FP number: 1237
TP number: 58507
FN number: 3765

Accuracy: 0.925
Precision: 0.979
False Positive Rate: 0.282

c. Ennio Doorbell

Local threshold: 0.011878
TN number: 10169
FP number: 2930
TP number: 11027
FN number: 0

Accuracy: 0.879
Precision: 0.790
False Positive Rate: 0.224

d. Philips_Baby_Monitor

Local threshold: 0.064790
TN number: 54558
FP number: 4148
TP number: 72214
FN number: 0

Accuracy: 0.968
Precision: 0.946
False Positive Rate: 0.071

e. Provision_737E_Security_Camera

Local threshold: 0.090970
TN number: 15909
FP number: 4913
TP number: 55655
FN number: 0

Accuracy: 0.936
Precision: 0.919
False Positive Rate: 0.236

f. Provision_838_Security_Camera

Local threshold: 0.112439
TN number: 23413
FP number: 9589
TP number: 54270
FN number: 0

Accuracy: 0.890
Precision: 0.850
False Positive Rate: 0.290558

g. Samsung_SNH_Webcam

Local threshold: 0.156782
TN number: 16311
FP number: 1159
TP number: 11380
FN number: 0

Accuracy: 0.960
Precision: 0.908
False Positive Rate: 0.066

h. SimpleHome_1002_Security_Camera

Local threshold: 1.381402
TN number: 14526
FP number: 1080
TP number: 6297
FN number: 56018

Accuracy: 0.267
Precision: 0.854
False Positive Rate: 0.069

i. SimpleHome_1003_Security_Camera

Local threshold: 0.084559
TN number: 6226
FP number: 316
TP number: 62991
FN number: 0

Accuracy: 0.995
Precision: 0.995
False Positive Rate: 0.048


## May 8th's update on Federated Learning: using all training samples and 1% of test data:
Note: Each client calculate its own local threshold
batch size = 64, lr = 1e-3, round = 50, epoch = 15 

(1) Mean results:

Accuracy: 0.86532
Precision: 0.608786
FPR: 0.14705

a. Danmini Doorbell

Local threshold: 0.005843

TN number: 15923
FP number:676
TP number: 7625
FN number: 0

Accuracy: 0.97209
Precision: 0.91856
False Positive Rate: 0.040725

b. Ecobee_Thermostat

Local threshold: 0.206668

TN number is 3166
FP number is 1227
TP number is 6169
FN number is 39

Accuracy: 0.880577
Precision: 0.834100
False Positive Rate: 0.279308


c. Ennio Doorbell

Local threshold: 0.011196
TN number: 10231
FP number: 2868
TP number: 1043
FN number: 0

Accuracy: 0.7972
Precision: 0.266684
False Positive Rate: 0.218948

d. Philips_Baby_Monitor

Local threshold: 0.061079
TN number: 54403
FP number: 4303
TP number: 6957
FN number: 0

Accuracy: 0.93447
Precision: 0.61785
False Positive Rate: 0.073297

e. Provision_737E_Security_Camera

Local threshold: 0.089253
TN number: 15845
FP number: 4977
TP number: 5472
FN number: 0

Accuracy: 0.810717
Precision: 0.523686
False Positive Rate: 0.239026

f. Provision_838_Security_Camera

Local threshold: 0.126871
TN number: 23716
FP number: 9286
TP number: 5280
FN number: 0

Accuracy: 0.75743
Precision: 0.36248
False Positive Rate: 0.28138

g. Samsung_SNH_Webcam

Local threshold: 0.14421
TN number: 16175
FP number: 1295
TP number: 1060
FN number: 0

Accuracy: 0.93011
Precision: 0.45010
False Positive Rate: 0.07412

h. SimpleHome_1002_Security_Camera

Local threshold: 1.080745
TN number: 14496
FP number: 1110
TP number: 1362
FN number: 4799

Accuracy: 0.728534
Precision: 0.55097
False Positive Rate: 0.07113

i. SimpleHome_1003_Security_Camera

Local threshold: 0.08757
TN number: 6244
FP number: 298
TP number: 6270
FN number: 0

Accuracy: 0.97674
Precision: 0.95463
False Positive Rate: 0.04555


## May 9th's update on Federated Learning: using all training samples and 1% of test data:
Note: Each client shares the same global threshold
batch size = 64, lr = 1e-3, round = 50, epoch = 15 

(1) Mean results:

Global threshold: 0.26388
Accuracy: 0.92613
Precision: 0.96572
FPR: 0.071501

a. Danmini Doorbell


TN number: 16514
FP number:85
TP number: 6627
FN number: 998

Accuracy: 0.95529
Precision: 0.98734
False Positive Rate: 0.00512

b. Ecobee_Thermostat

TN number is 3564
FP number is 829
TP number is 5185
FN number is 1023

Accuracy: 0.82530
Precision: 0.86216
False Positive Rate: 0.18871


c. Ennio Doorbell

TN number: 13025
FP number: 74
TP number: 1043
FN number: 0

Accuracy: 0.99477
Precision: 0.93375
False Positive Rate: 0.00565

d. Philips_Baby_Monitor

TN number: 56986
FP number: 1720
TP number: 6770
FN number: 187

Accuracy: 0.97096
Precision: 0.79741
False Positive Rate: 0.02930

e. Provision_737E_Security_Camera

TN number: 18585
FP number: 2237
TP number: 4848
FN number: 624

Accuracy: 0.89119
Precision: 0.68426
False Positive Rate: 0.10743

f. Provision_838_Security_Camera

TN number: 28553
FP number: 4449
TP number: 4996
FN number: 284

Accuracy: 0.87637
Precision: 0.52896
False Positive Rate: 0.13481

g. Samsung_SNH_Webcam

TN number: 16595
FP number: 875
TP number: 957
FN number: 103

Accuracy: 0.94722
Precision: 0.52238
False Positive Rate: 0.05009

h. SimpleHome_1002_Security_Camera

TN number: 14223
FP number: 1383
TP number: 5255
FN number: 906

Accuracy: 0.89484
Precision: 0.79165
False Positive Rate: 0.08862

i. SimpleHome_1003_Security_Camera

TN number: 6321
FP number: 221
TP number: 6225
FN number: 45

Accuracy: 0.97924
Precision: 0.96572
False Positive Rate: 0.03378


## May 10's update on FL

Each client shares the same global threshold 

Note: The propotion of test data in each devices is 

A: 5% B: 1% C: 1% D: 10% E: 10% F: 1% G: 10% H: 5% I: 1%

(1) Mean results

Global threshold: 0.225507
Accuracy: 0.922332
Precision: 0.885823
False Positive Rate: 0.090127

(2) Details

a. Danmini_Doorbell

The True negative number is 16512.000000
The False positive number is 87.000000
The True positive number is 34653.000000
The False negative number is 3800.000000

The accuracy is 0.929394
The precision is 0.997496
The false positive rate is 0.005241

b. Ecobee_Thermostat

The True negative number is 3126.000000
The False positive number is 1267.000000
The True positive number is 5499.000000
The False negative number is 709.000000

The accuracy is 0.813602
The precision is 0.812740
The false positive rate is 0.288413

c. Ennio_Doorbell

The True negative number is 13025.000000
The False positive number is 74.000000
The True positive number is 1043.000000
The False negative number is 0.000000

The accuracy is 0.994767
The precision is 0.933751
The false positive rate is 0.005649

d. Philips_Baby_Monitor

The True negative number is 56907.000000
The False positive number is 1799.000000
The True positive number is 70904.000000
The False negative number is 1310.000000

The accuracy is 0.976253
The precision is 0.975255
The false positive rate is 0.030644

e. Provision_737E

The True negative number is 18270.000000
The False positive number is 2552.000000
The True positive number is 52938.000000
The False negative number is 2717.000000

The accuracy is 0.931103
The precision is 0.954010
The false positive rate is 0.122563

f. Provision_838

True negative number is 27174.000000
The False positive number is 5828.000000
The True positive number is 5060.000000
The False negative number is 220.000000

The accuracy is 0.842015
The precision is 0.464732
The false positive rate is 0.176595

g. Samsung_Webcam

The True negative number is 16496.000000
The False positive number is 974.000000
The True positive number is 11380.000000
The False negative number is 0.000000

The accuracy is 0.966239
The precision is 0.921159
The false positive rate is 0.055753

h. SimpleHome_1002

The True negative number is 14191.000000
The False positive number is 1415.000000
The True positive number is 27521.000000
The False negative number is 3597.000000

The accuracy is 0.892732
The precision is 0.951099
The false positive rate is 0.090670

i. SimpleHome_1003

The True negative number is 6309.000000
The False positive number is 233.000000
The True positive number is 5925.000000
The False negative number is 345.000000

The accuracy is 0.954886
The precision is 0.962163
The false positive rate is 0.035616


## May 12's update on FL
With the global testset, global threshold

The accuracy is 0.80
The precision is 0.75
The false positive rate is 0.172
Threshold = 0.355
TN = 7448
FP = 1552
TP = 4649
FN = 1550

With the global testset, local threshold

The accuracy is 0.76
The precision is 0.652
The false positive rate is 0.325
Avg Threshold is 0.2916

Threshold = [0.0149, 0.203, 0.0305, 0.135, 0.05, 0.075, 0.271, 1.69, 0.155]
TN = 6078
FP = 2922
TP = 5478
FN = 721

## May 20's on Sampling( Dropout) method

Using 10% training data and global testset. 

comm_round = 100, epoch_per_round = 1, batch_size = 64, lr = 1e-3

I. Centralized training

The True negative number is  6672
The False positive number is  2328
The True positive number is  981
The False negative number is  5219

The accuracy is  0.503
The precision is  0.296
The false positive rate is  0.259

II. FedIoT && Global threshold:

 (1) Random sample 1 device
 
 a. 
 
The threshold is 0.220851

The True negative number is 1000
The False positive number is 0
The True positive number is 800
The False negative number is 0

The accuracy is 1.000
The precision is 1.000
The false positive rate is 0.000

b.

The threshold is 0.220851

The True negative number is 865
The False positive number is 135
The True positive number is 800
The False negative number is 0

The accuracy is 0.925
The precision is 0.856
The false positive rate is 0.135

c. 

The threshold is 0.220851

The True negative number is 995
The False positive number is 5
The True positive number is 300
The False negative number is 0

The accuracy is 0.996
The precision is 0.984
The false positive rate is 0.005

d. 

The threshold is 0.220851

The True negative number is 927
The False positive number is 73
The True positive number is 800
The False negative number is 0

The accuracy is 0.959
The precision is 0.916
The false positive rate is 0.073

e.

The threshold is 0.220851

The True negative number is 792
The False positive number is 208
The True positive number is 800
The False negative number is 0

The accuracy is 0.884
The precision is 0.794
The false positive rate is 0.208

f.

The threshold is 0.220851

The True negative number is 772
The False positive number is 228
The True positive number is 800
The False negative number is 0

The accuracy is 0.873
The precision is 0.778
The false positive rate is 0.228

g.

The threshold is 0.220851

The True negative number is 999
The False positive number is 1
The True positive number is 300
The False negative number is 0

The accuracy is 0.999
The precision is 0.997
The false positive rate is 0.001

h.

The threshold is 0.220851

The True negative number is 992
The False positive number is 8
The True positive number is 800
The False negative number is 0

The accuracy is 0.996
The precision is 0.990
The false positive rate is 0.008

i. 

The threshold is 0.220851

The True negative number is 998
The False positive number is 2
The True positive number is 762
The False negative number is 38

The accuracy is 0.978
The precision is 0.997
The false positive rate is 0.002

Mean results

accuracy_mean_global = 0.957
precision_mean_global = 0.924
fpr_mean_global = 0.073

 (2) Random samples 3 devices

a.

The threshold is 0.178757

The True negative number is 1000
The False positive number is 0
The True positive number is 800
The False negative number is 0

The accuracy is 1
The precision is 1
The false positive rate is 0

b.

The threshold is 0.178757

The True negative number is 701
The False positive number is 299
The True positive number is 800
The False negative number is 0

The accuracy is 0.834
The precision is 0.728
The false positive rate is 0.299

c.

The threshold is 0.178757

The True negative number is 990
The False positive number is 10
The True positive number is 300
The False negative number is 0

The accuracy is 0.992
The precision is 0.968
The false positive rate is 0.010

d.

The threshold is 0.178757

The True negative number is 929
The False positive number is 71
The True positive number is 800
The False negative number is 0

The accuracy is 0.961
The precision is 0.918
The false positive rate is 0.071

e.

The threshold is 0.178757

The True negative number is 844
The False positive number is 156
The True positive number is 800
The False negative number is 0

The accuracy is 0.913
The precision is 0.837
The false positive rate is 0.156

f.

The threshold is 0.178757

The True negative number is 800
The False positive number is 200
The True positive number is 800
The False negative number is 0

The accuracy is 0.889
The precision is 0.800
The false positive rate is 0.200

g.

The threshold is 0.178757

The True negative number is 998
The False positive number is 2
The True positive number is 300
The False negative number is 0

The accuracy is 0.998
The precision is 0.993
The false positive rate is 0.002

h.

The threshold is 0.178757

The True negative number is 984
The False positive number is 16
The True positive number is 800
The False negative number is 0

The accuracy is 0.991
The precision is 0.980
The false positive rate is 0.016

i.

The threshold is 0.178757

The True negative number is 997
The False positive number is 3
The True positive number is 800
The False negative number is 0

The accuracy is 0.998
The precision is 0.996
The false positive rate is 0.003

Mean results

accuracy_mean_global = 0.952987
precision_mean_global = 0.913446
fpr_mean_global = 0.084111

(3) Random sample 6 devices

a.

The threshold is 0.171713

The True negative number is 1000
The False positive number is 0
The True positive number is 800
The False negative number is 0

The accuracy is 1
The precision is 1
The false positive rate is 0

b.

The threshold is 0.171713

The True negative number is 535
The False positive number is 465
The True positive number is 800
The False negative number is 0

The accuracy is 0.742
The precision is 0.632
The false positive rate is 0.465

c.

The threshold is 0.171713

The True negative number is 901
The False positive number is 99
The True positive number is 300
The False negative number is 0

The accuracy is 0.924
The precision is 0.751
The false positive rate is 0.099

d.

The threshold is 0.171713

The True negative number is 926
The False positive number is 74
The True positive number is 800
The False negative number is 0

The accuracy is 0.958
The precision is 0.915
The false positive rate is 0.074

e.

The threshold is 0.171713

The True negative number is 876
The False positive number is 124
The True positive number is 800
The False negative number is 0

The accuracy is 0.931
The precision is 0.865
The false positive rate is 0.124

f.

The threshold is 0.171713

The True negative number is 836
The False positive number is 164
The True positive number is 800
The False negative number is 0

The accuracy is 0.909
The precision is 0.830
The false positive rate is 0.164

g.

The threshold is 0.171713

The True negative number is 996
The False positive number is 4
The True positive number is 300
The False negative number is 0

The accuracy is 0.997
The precision is 0.987
The false positive rate is 0.004

h.

The threshold is 0.171713

The True negative number is 942
The False positive number is 58
The True positive number is 800
The False negative number is 0

The accuracy is 0.968
The precision is 0.932
The false positive rate is 0.058

i.

The threshold is 0.171713

The True negative number is 996
The False positive number is 4
The True positive number is 800
The False negative number is 0

The accuracy is 0.998
The precision is 0.995
The false positive rate is 0.004

Mean results

accuracy_mean_global = 0.936
precision_mean_global = 0.879
fpr_mean_global = 0.110

(4). No sampling method, using all devices for training

a.

The threshold is 0.159562

The True negative number is 1000
The False positive number is 0
The True positive number is 800
The False negative number is 0

The accuracy is 1
The precision is 1
The false positive rate is 0

b.

The threshold is 0.159562

The True negative number is 305
The False positive number is 695
The True positive number is 800
The False negative number is 0

The accuracy is 0.614
The precision is 0.535
The false positive rate is 0.695

c.

The threshold is 0.159562

The True negative number is 88.000000
The False positive number is 114
The True positive number is 300
The False negative number is 0

The accuracy is 0.912
The precision is 0.725
The false positive rate is 0.114

d.

The threshold is 0.159562

The True negative number is 930
The False positive number is 70
The True positive number is 800
The False negative number is 0

The accuracy is 0.961
The precision is 0.920
The false positive rate is 0.070

e.

The threshold is 0.159562

The True negative number is 841
The False positive number is 159
The True positive number is 800
The False negative number is 0

The accuracy is 0.912
The precision is 0.834
The false positive rate is 0.159

f.

The threshold is 0.159562

The True negative number is 802
The False positive number is 198
The True positive number is 800
The False negative number is 0

The accuracy is 0.890
The precision is 0.802
The false positive rate is 0.198

g.

The threshold is 0.159562
The True negative number is 995
The False positive number is 5
The True positive number is 300
The False negative number is 0

The accuracy is 0.996
The precision is 0.984
The false positive rate is 0

h.

The threshold is 0.159562

The True negative number is 893
The False positive number is 107
The True positive number is 800
The False negative number is 0

The accuracy is 0.940
The precision is 0.882
The false positive rate is 0.107

i.

The threshold is 0.159562

The True negative number is 996
The False positive number is 4
The True positive number is 800
The False negative number is 0

The accuracy is 0.998
The precision is 0.995
The false positive rate is 0.004

Mean result

accuracy_mean_global = 0.914
precision_mean_global = 0.853
fpr_mean_global = 0.150




