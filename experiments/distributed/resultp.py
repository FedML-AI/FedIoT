import matplotlib.pyplot as plt

name_list = ['A','B','C','D','E','F','G','H','I']
acc_num_list_cen = [0.8236, 0.5940, 0.8243, 0.8771, 0.8426, 0.9090, 0.7542, 0.7802, 0.8254]
acc_num_list_dis = [0.9552, 0.8253, 0.9947, 0.9709, 0.8911, 0.8763, 0.9472, 0.8948, 0.9792]
x =list(range(len(name_list)))
total_width, n = 0.5, 2
width = total_width / n
plt.bar(x, acc_num_list_cen, width=width, label='Centralized training with a,c &f data',fc = 'y')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, acc_num_list_dis, width=width, label='Federated training',tick_label = name_list,fc = 'r')
plt.legend()
plt.ylim(0.7, 1.0)
plt.xlabel('Device index')
plt.ylabel('Accuracy Magnitude')
plt.title('Accuracy Performance')
plt.show()

pre_num_list_cen = [0.933, 0.737, 0.1639, 0.4435, 0.859, 0.9624, 0.0724, 0.652, 0.9396]
pre_num_list_dis = [0.987, 0.862, 0.9337, 0.7974, 0.6842, 0.5289, 0.5223, 0.7916, 0.9657]
plt.bar(x, pre_num_list_cen, width=width, label='Centralized training with a,c &f data',fc = 'y')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, pre_num_list_dis, width=width, label='Federated training',tick_label = name_list,fc = 'r')
plt.legend()
plt.ylim(0.5, 1.0)
plt.xlabel('Device index')
plt.ylabel('Precision Magnitude')
plt.title('Precision Performance')
plt.show()

fpr_num_list_cen = [0.016, 0.248, 0.130, 0.063, 0.014, 0.002, 0.212, 0.1105, 0.0434]
fpr_num_list_dis = [0.005, 0.188, 0.005, 0.0293, 0.107, 0.1348, 0.05, 0.088, 0.033]
plt.bar(x, fpr_num_list_cen, width=width, label='Centralized training with a,c &f data',fc = 'y')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, fpr_num_list_dis, width=width, label='Federated training',tick_label = name_list,fc = 'r')
plt.legend()
plt.xlabel('Device index')
plt.ylabel('False Postive Rate')
plt.title('FPR Performance')
plt.show()

tr_num_list_cen = [0.64176, 0.64176, 0.64176, 0.64176, 0.64176, 0.64176, 0.64176, 0.64176, 0.64176]
tr_num_list_dis = [0.26388, 0.26388, 0.26388, 0.26388, 0.26388, 0.26388, 0.26388, 0.26388, 0.26388]
plt.bar(x, tr_num_list_cen, width=width, label='Centralized training with a,c &f data',fc = 'y')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, tr_num_list_dis, width=width, label='Federated training',tick_label = name_list,fc = 'r')
plt.legend()
plt.xlabel('Device index')
plt.ylabel('Threshold Magnitude')
plt.title('Threshold Performance')
plt.show()

be_num_list = [49548, 13113, 39100, 175240, 62154, 98514, 52150, 46585, 19528]
att_num_list = [7707, 6229, 1109, 7250, 5575, 5443, 1146, 6239, 6302]
plt.bar(x, be_num_list, width=width, label='Benign Sample Number',fc = 'y')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, att_num_list, width=width, label='Attack Sample Number',tick_label = name_list,fc = 'r')
plt.legend()
plt.xlabel('Device index')
plt.ylabel('Sample Count')
plt.title('Benign and Attack Data Distribution')
plt.show()

train_num_list = [round(num*2/3) for num in be_num_list]
opt_num_list = [round(num*1/3) for num in be_num_list]
test_num_list = [opt_num_list[i] + att_num_list[i] for i in range(len(att_num_list))]
plt.bar(x, train_num_list, width=width, label='Train Sample',fc = 'y')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, opt_num_list, width=width, label='Optimization Sample',tick_label = name_list,fc = 'r')
plt.bar(x, test_num_list, width=width, label='Attack Sample',tick_label = name_list,fc = 'b', alpha = 0.2)
plt.legend()
plt.xlabel('Device index')
plt.ylabel('Sample Count')
plt.title('Train, Optimization and Test Sets Data Distribution')
plt.show()