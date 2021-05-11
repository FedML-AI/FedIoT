import matplotlib.pyplot as plt

atk_name = ['combo_g', 'junk_g', 'scan_g', 'ack_m', 'scan_m', 'syn_m', 'udp_m', 'udpplain_m']
device_list = ['A','B','C','D','E','F','G','H','I']
atk_type = {}
atk_type[0] = [2986, 1453, 1492, 5110, 5384, 6129, 11883, 4099]
atk_type[1] = [530, 303, 275, 1133, 432, 1168, 1515, 874]
atk_type[2] = [530, 298, 281, 0, 0, 0, 0, 0]
atk_type[3] = [5815, 2835, 2786, 9112, 10362, 11813, 21703, 8081]
atk_type[4] = [6138, 3090, 2930, 6055, 9678, 6575, 15625, 5668]
atk_type[5] = [575, 291, 284, 580, 971, 619, 1586, 538]
atk_type[6] = [5867, 2830, 2770, 0, 0, 0, 0, 0]
atk_type[7] = [2714, 1429, 1391, 5574, 2296, 6286, 7594, 3912]
atk_type[8] = [594, 274, 286, 1072, 437, 1225, 1571, 844]

for i in range(len(device_list)):
    x = list(range(len(atk_name)))
    total_width, n = 0.5, 2
    width = total_width / n
    plt.bar(x, atk_type[i], width=width, label='Atk type distribution',tick_label = atk_name, fc='y')
    for j in range(len(x)):
        x[j] = x[j] + width
    plt.legend()
    plt.xlabel('Type of Attack')
    plt.ylabel('Attack Sample Number')
    plt.title('Attack Sample Distribution of device ' + device_list[i])
    plt.show()
    print(i)