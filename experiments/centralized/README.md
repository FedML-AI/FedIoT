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


## May 10's update on centralized training 

a, c, f with number-adjusted test set

mean accuracy = 0.8536064879730138
mean precision = 0.7372995968904188
mean false positive rate = 0.1449454553978402

(A) Danmini_Doorbell

The Threshold is  tensor(0.2860)

The True negative number is  15885
The False positive number is  467
The True positive number is  38535
The False negative number is  0

The accuracy is  0.9914916100351632
The precision is  0.9880262550638429
The false positive rate is  0.028559197651663405

(B) Ecobee_Thermostat

The Threshold is  tensor(0.6940)

The True negative number is  2922
The False positive number is  1406
The True positive number is  4996
The False negative number is  1233

The accuracy is  0.7500236809699725
The precision is  0.7803811308965948
The false positive rate is  0.32486136783733827

(C) Ennio_Doorbell

The Threshold is  tensor(0.5094)

The True negative number is  10869
The False positive number is  2035
The True positive number is  836
The False negative number is  272

The accuracy is  0.8353554096488724
The precision is  0.29118773946360155
The false positive rate is  0.15770303781773093

(D) Philips_Baby_Monitor

The Threshold is  tensor(0.5525)

The True negative number is  53432
The False positive number is  4398
The True positive number is  69708
The False negative number is  2798

The accuracy is  0.9447888534249939
The precision is  0.9406525787385637
The false positive rate is  0.07605049282379388

(E) Provision_737E

The Threshold is  tensor(0.1201)

The True negative number is  15852
The False positive number is  4660
The True positive number is  55758
The False negative number is  0

The accuracy is  0.9389012717975613
The precision is  0.9228706676818167
The false positive rate is  0.22718408736349455

(F) Provision_838

The Threshold is  tensor(0.0547)

The True negative number is  28540
The False positive number is  3971
The True positive number is  5443
The False negative number is  0

The accuracy is  0.8953733466828265
The precision is  0.5781814319099214
The false positive rate is  0.12214327458398695

(G) Samsung_Webcam

The Threshold is  tensor(0.9247)

The True negative number is  13600
The False positive number is  3611
The True positive number is  1163
The False negative number is  10303

The accuracy is  0.5148028036405482
The precision is  0.24361122748219521
The false positive rate is  0.20980768113415837

(H) SimpleHome_1002

The Threshold is  tensor(0.6931)

The True negative number is  13667
The False positive number is  1707
The True positive number is  25610
The False negative number is  5585

The accuracy is  0.8434151474156628
The precision is  0.9375114397627851
The false positive rate is  0.11103161181215038

(I) SimpleHome_1003

The Threshold is  tensor(0.5781)

The True negative number is  6141
The False positive number is  304
The True positive number is  6202
The False negative number is  100

The accuracy is  0.9683062681415235
The precision is  0.9532739010144482
The false positive rate is  0.04716834755624515


## May 10's update on centralized training 

b, g, h with number-adjusted test set

mean accuracy = 0.6106045410553165
mean precision = 0.46476935125981833
mean false positive rate = 0.19898762645265686


(A) Danmini_Doorbell 

The Threshold is  tensor(1.3826)
The True negative number is  13165
The False positive number is  3187
The True positive number is  1733
The False negative number is  36802
The accuracy is  0.2714303933536174
The precision is  0.3522357723577236
The false positive rate is  0.19489970645792565

(B) Ecobee_Thermostat

The Threshold is  tensor(1.7047)
The True negative number is  3196
The False positive number is  1132
The True positive number is  163
The False negative number is  6066
The accuracy is  0.3181775125509141
The precision is  0.12586872586872586
The false positive rate is  0.26155268022181144

(C) Ennio_Doorbell

The Threshold is  tensor(1.3766)
The True negative number is  10517
The False positive number is  2387
The True positive number is  1108
The False negative number is  0
The accuracy is  0.8296460176991151
The precision is  0.3170243204577968
The false positive rate is  0.18498140111593303

(D) Philips_Baby_Monitor

The Threshold is  tensor(1.8861)
The True negative number is  45391
The False positive number is  12439
The True positive number is  414
The False negative number is  72092
The accuracy is  0.3514378222440462
The precision is  0.032210378899867734
The false positive rate is  0.21509597094933425

(E) Provision_737E

The Threshold is  tensor(2.3248)
The True negative number is  13832
The False positive number is  6680
The True positive number is  4348
The False negative number is  51410
The accuracy is  0.23836370787990036
The precision is  0.3942691331157055
The false positive rate is  0.3256630265210608

(F) Provision_838

The Threshold is  tensor(1.3024)
The True negative number is  20118
The False positive number is  12393
The True positive number is  2142
The False negative number is  3301
The accuracy is  0.5864994466986352
The precision is  0.14736842105263157
The false positive rate is  0.3811940573959583

(G) Samsung_Webcam

The Threshold is  tensor(0.2115)
The True negative number is  16188
The False positive number is  1023
The True positive number is  11466
The False negative number is  0
The accuracy is  0.9643268124280783
The precision is  0.9180879173672832
The false positive rate is  0.059438731044099705

(H) SimpleHome_1003

The Threshold is  tensor(0.0862)
The True negative number is  13575
The False positive number is  1799
The True positive number is  31195
The False negative number is  0
The accuracy is  0.9613691511520539
The precision is  0.9454749348366369
The false positive rate is  0.11701574086119422

(I) SimpleHome_1003

The Threshold is  tensor(0.1661)
The True negative number is  6116
The False positive number is  329
The True positive number is  6302
The False negative number is  0
The accuracy is  0.9741900054914882
The precision is  0.9503845573819937
The false positive rate is  0.05104732350659426

## May 11's update on centralized training 

I. Baseline 1: Local data model.

Mean accuracy: 0.520

Mean precision: 0.460

Mean false positive rate: 0.811

(a) Danmini_Doorbell

threshold = 0.358700

The True negative number is  1988
The False positive number is  7012
The True positive number is  6200
The False negative number is  0

The accuracy is 0.539
The precision is 0.469
The false positive rate is 0.779

(b) Ecobee

threshold = 0.102526

The True negative number is  1674
The False positive number is  7326
The True positive number is  6200
The False negative number is  0

The accuracy is 0.518
The precision is 0.458
The false positive rate is 0.814

(c) Ennio

threshold = 0.117762

The True negative number is  1763
The False positive number is  7237
The True positive number is  6200
The False negative number is  0

The accuracy is 0.524
The precision is 0.461
The false positive rate is 0.804

(d) Philips

threshold = 0.113661

The True negative number is  2136
The False positive number is  6864
The True positive number is  6200
The False negative number is  0

The accuracy is 0.548
The precision is 0.475
The false positive rate is 0.763

(e) Provision_7

threshold = 0.040038

The True negative number is  1319
The False positive number is  7681
The True positive number is  6200
The False negative number is  0

The accuracy is 0.495
The precision is 0.447
The false positive rate is 0.853

(f) Provision_8

threshold = 0.045994

The True negative number is  1507
The False positive number is  7493
The True positive number is  6200
The False negative number is  0

The accuracy is 0.507
The precision is 0.453
The false positive rate is 0.833

(g) Samsung

threshold = 0.192586

The True negative number is  2700
The False positive number is  6300
The True positive number is  6200
The False negative number is  0

The accuracy is 0.586
The precision is 0.496
The false positive rate is 0.700

(h) SimpleHome_1002

hreshold = 0.068146

The True negative number is  595
The False positive number is  8405
The True positive number is  6200
The False negative number is  0

The accuracy is 0.447
The precision is 0.425
The false positive rate is 0.934

(i) SimpleHome_1003

threshold = 0.138309

The True negative number is  1669
The False positive number is  7331
The True positive number is  6200
The False negative number is  0

The accuracy is 0.518
The precision is 0.458
The false positive rate is 0.815


II. Baseline 2: BGI data model.

threshold = 0.126311

The True negative number is  3933
The False positive number is  5067
The True positive number is  6200
The False negative number is  0

The accuracy is 0.667
The precision is 0.550
The false positive rate is 0.563


III. Baseline 3: All union data model

Threshold = 0.146357

The True negative number is  6592
The False positive number is  2408
The True positive number is  6101
The False negative number is  99

The accuracy is 0.835
The precision is 0.717
The false positive rate is 0.268
