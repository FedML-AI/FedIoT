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

