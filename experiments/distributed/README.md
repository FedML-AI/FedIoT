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
##Apr 19's update on Federated Learning: training using the 9 separated datasets with 10% data:

Mean Result:
Accuracy: 0.967479 
Precision: 0.999 
False Positive Rate: 0.427746

##Apr 22's update on Federated Learning: training using the 9 separated datasets with 50% data:

(1) Mean Result:

Global threshold: 0.092956
Accuracy: 0.963542
Precision: 0.999716
FPR: 0.307094


Detail Result (client-wise):

(1) Danmini Doorbell
Accuracy:0.979501
Precision:1.00
False Positive Rate:0.000

(2) Ecobee_Thermostat

Accuracy: 0.998068
Precision: 0.999923
False Positive Rate: 0.685714 

(3) Ennio Doorbell

Accuracy: 0.907717
Precision: 0.999840
False Positive Rate: 0.087379 

(4) Philips_Baby_Monitor

Accuracy: 0.931966
Precision: 0.999663
False Positive Rate: 0. 267974

(5) Provision_737E_Security_Camera

Accuracy: 0.973180
Precision: 0.999673
False Positive Rate: 0.564417 

(6) Provision_838_Security_Camera

Accuracy: 0.963966
Precision: 0.999986
False Positive Rate: 0.015504 

(7) Samsung_SNH_Webcam

Accuracy: 0.944524
Precision: 0.998542
False Positive Rate: 0.664234 

(8) SimpleHome_1002_Security_Camera

Accuracy: 0.982213
Precision: 0.999822
FPR: 0.459016

(9) SimpleHome_1002_Security_Camera

Accuracy: 0.990744
Precision: 0.999997
FPR: 0.019608

