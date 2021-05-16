import matplotlib.pyplot as plt

# atk_name = ['combo_g', 'junk_g', 'scan_g', 'ack_m', 'scan_m', 'syn_m', 'udp_m', 'udpplain_m']
# device_list = ['A','B','C','D','E','F','G','H','I']
# atk_type = {}
# atk_type[0] = [2986, 1453, 1492, 5110, 5384, 6129, 11883, 4099]
# atk_type[1] = [530, 303, 275, 1133, 432, 1168, 1515, 874]
# atk_type[2] = [530, 298, 281, 0, 0, 0, 0, 0]
# atk_type[3] = [5815, 2835, 2786, 9112, 10362, 11813, 21703, 8081]
# atk_type[4] = [6138, 3090, 2930, 6055, 9678, 6575, 15625, 5668]
# atk_type[5] = [575, 291, 284, 580, 971, 619, 1586, 538]
# atk_type[6] = [5867, 2830, 2770, 0, 0, 0, 0, 0]
# atk_type[7] = [2714, 1429, 1391, 5574, 2296, 6286, 7594, 3912]
# atk_type[8] = [594, 274, 286, 1072, 437, 1225, 1571, 844]
#
# for i in range(len(device_list)):
#     x = list(range(len(atk_name)))
#     total_width, n = 0.5, 2
#     width = total_width / n
#     plt.bar(x, atk_type[i], width=width, label='Atk type distribution',tick_label = atk_name, fc='y')
#     for j in range(len(x)):
#         x[j] = x[j] + width
#     plt.legend()
#     plt.xlabel('Type of Attack')
#     plt.ylabel('Attack Sample Number')
#     plt.title('Attack Sample Distribution of device ' + device_list[i])
#     plt.show()
#     print(i)
exper_name = ['base_a', 'base_b', 'base_c', 'fl_a1', 'fl_a2']
eva_type = ['Accuracy','FPR','Precision','Threshold','True Negative', 'False Positive',
            'True Positive','False Negative']
result = {}
result[0] = [0.52, 0.667, 0.835, 0.80, 0.76] #acc
result[1] = [0.815, 0.563, 0.268, 0.172, 0.325] #fpr
result[2] = [0.458, 0.550, 0.717, 0.75, 0.652] #pre
result[3] = [0.1308, 0.1263, 0.146357, 0.355, 0.2916] #tr
result[4] = [1706, 3933, 6592, 7448, 6078] #tn
result[5] = [7294, 5067, 2408, 1552, 2922] #fp
result[6] = [6200, 6200, 6101, 4649, 5478] #tp
result[7] = [0, 0, 99, 1550, 721] #fn

for i in range(8):
    x = list(range(len(exper_name)))
    total_width, n = 0.5, 2
    width = total_width / n
    plt.bar(x, result[i], width=width,tick_label = exper_name, fc='y')
    for j in range(len(x)):
        x[j] = x[j] + width
    plt.legend()
    plt.xlabel('Type of Experiment')
    plt.ylabel('Sample Number Account')
    plt.title(eva_type[i] + ' Performance')
    plt.show()
    print(i)