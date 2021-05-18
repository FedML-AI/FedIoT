## Raspberry Pi Results on May 18th.

We use one raspberry Pi 4 with 8 GB ram to simulate the experiment. We take 10% of the benign data as the trainset and set the number of worker equals to 1.
As a result, in each epoch, the server will randomly select a model from all 9 devices model and send it as the global model to the raspberry Pi. The results are
below. We test it with the global testset.

Accuracy: 0.783
Precision: 0.663
FPR: 0.260
Total end to end time: 2547s
Process Memory in use: 46.5MB

Device A:
Avg epoch running time: 28.11s
Avg uplink time: 0.187s
Avg comm time: 9.87s
Avg comp time: 18.24s
ratio between comm and comp: 0.54

Device B:
Avg epoch running time: 24.83s
Avg uplink time: 0.157s
Avg comm time: 10.37s
Avg comp time: 10.465s
ratio between comm and comp: 0.72

Device C:
Avg epoch running time: 27.73s
Avg uplink time: 0.156s
Avg comm time: 10.226s
Avg comp time: 17.51s
ratio between comm and comp: 0.584

Device D:
Avg epoch running time: 23.72s
Avg uplink time: 0.15s
Avg comm time: 10.62s
Avg comp time: 13.10s
ratio between comm and comp: 0.81

Device E:
Avg epoch running time: 22.78s
Avg uplink time: 0.17s
Avg comm time: 10.79s
Avg comp time: 11.99s
ratio between comm and comp: 0.90

Device F:
Avg epoch running time: 27.2s
Avg uplink time: 0.178s
Avg comm time: 9.64s
Avg comp time: 17.56s
ratio between comm and comp: 0.548

Device G:
Avg epoch running time: 24.23s
Avg uplink time: 0.163s
Avg comm time: 11.67s
Avg comp time: 12.56s
ratio between comm and comp: 0.923

Device H:
Avg epoch running time: 22.85s
Avg uplink time: 0.167s
Avg comm time: 9.80s
Avg comp time: 13.04s
ratio between comm and comp: 0.75

Device I:
Avg epoch running time: 20.2s
Avg uplink time: 0.176s
Avg comm time: 10.559s
Avg comp time: 9.641s
ratio between comm and comp: 1.095

## Raspberry Pi Results on May 19th. 

# Personalized threshold 

At each communication round, the system would randomly select 1 device and its data to train the model


(1) Mean results:

accuracy_mean_global 0.79482
precision_mean_global 0.67386
fpr_mean_global 0.25611

(2) 

TN number: 6695
FP number: 2305
TP number: 5342
FN number: 857

(3)Personalized threshold details

threshold_a: 0.15509

threshold_b: 0.38903

threshold_c: 0.18088

threshold_d: 0.6344

threshold_e: 0.38689

threshold_f: 0.53697

threshold_g: 0.48035

threshold_h: 2.18243

threshold_i: 0.37587


## Global threshold

(1) Mean results

threshold_global 0.71018
accuracy_mean_global 0.73806
precision_mean_global 0.70541
fpr_mean_global 0.20656

(2)

TN number: 7141
FP number: 1859
TP number: 4027
FN number: 2172


