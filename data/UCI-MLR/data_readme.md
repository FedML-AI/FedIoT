## Introduction
We select a real world IoT devices' traffic dataset called N-BaIoT to test our platform and algorithm.
[N-BaIoT](https://archive.ics.uci.edu/ml/datasets/detection_of_IoT_botnet_attacks_N_BaIoT)

The data is composed by three components, benign_traffic, bashlite attacks and mirai attacks

For each attack, it has five different executed methods as shown below.

bashlite attacks:
1. Scan: Scanning the network for vulnerable devices
2. Junk: Sending spam data
3. UDP: UDP flooding
4. TCP: TCP flooding
5. COMBO: Sending spam data and opening a connection to a specified IP address and port

mirai attacks:
1. Scan: Automatic scanning for vulnerable devices
2. Ack: Ack flooding
3. Syn: Syn flooding
4. UDP: UDP flooding
5. UDPplain: UDP flooding with fewer options, optimized for higher PPS

Note: We found the TCP and UDP samples under bashlite attacks dataset are inconsistant, therefore we exclude them from our test dataset

## Data Selection

(1) The Training and Opt dataset:

Due to the limitation of computing resources in IoT devices, we intercept the first 10 percent of benign traffic samples as our dataset for model training and threshold selection.

|                           | Device A | Device B |Device C |Device D  |Device E |Device F |Device G |Device H |Device I |
| ---------- | :-----------:  | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| Benign Sample Number      | 4955     | 1311     |3910     |17524     |6215     |9851     |5215     |4658     |1953     |
| Training Sample Number    | 3319     | 879      |2620     |11742     |4164     |6601     |3495     |3121     |1308     |
| Opt Sample Number         | 1636     | 432      |1290     |5782      |2051     |3250     |1720     |1537     |645      |

***

(2) The Test dataset:

We integrate the benign and attack samples from 9 devices into one test set in order to fairly compare the performance between proposed methods. To be specific, for each device, we choose the last 1000 samples of benign set and the first 100 samples of each attack modes. 

|                         | Device A | Device B |Device C |Device D  |Device E  |Device F  |Device G  |Device H  |Device I |
| ---------- | :-----------:  | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| Benign Sample Number    | 1000     | 1000     | 1000    | 1000     | 1000     | 1000     | 1000     | 1000     | 1000     |
| Attack Sample Number    | 800      | 800      | 300     | 800      | 800      | 800      |300       | 800      | 800      |

We evaluate every model on this global testset and the total number is 15,200.
