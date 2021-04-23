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