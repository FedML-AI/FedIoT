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
## Experiment Results
Apr 19's update on Federated Learning training using the 9 saparated datasets with 10% data:

Mean Result:
Accuracy: 0.967479 Precision: 0.999 FPR: 0.427746

Apr 22's update on Federated Learning training using the 9 saparated datasets with 50% data:

Mean Result:



Detail Result:

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

