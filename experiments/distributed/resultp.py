import matplotlib.pyplot as plt

name_list = ['A','B','C','D','E','F','G','H','I']
acc_num_list_cen = [0.999215, 0.99894, 0.99409, 0.99272, 0.997538, 0.9957, 0.99033, 0.99697, 0.99917]
acc_num_list_dis = [0.979501, 0.998068, 0.907717, 0.931966, 0.973180, 0.963966, 0.944524, 0.982213, 0.990744]
x =list(range(len(name_list)))
total_width, n = 0.5, 2
width = total_width / n
plt.bar(x, acc_num_list_cen, width=width, label='Centralized training',fc = 'y')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, acc_num_list_dis, width=width, label='Federated training',tick_label = name_list,fc = 'r')
plt.legend()
plt.ylim(0.9, 1.0)
plt.xlabel('Device index')
plt.ylabel('Accuracy Magnitude')
plt.title('Accuracy Performance')
plt.show()

pre_num_list_cen = [0.999198, 0.99893, 0.99343, 0.99219, 0.99745, 0.9954, 0.98898, 0.99691, 0.99920]
pre_num_list_dis = [1.00, 0.999923, 0.999840, 0.999663, 0.999673, 0.999986, 0.998542, 0.999822, 0.999997]
plt.bar(x, pre_num_list_cen, width=width, label='Centralized training',fc = 'y')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, pre_num_list_dis, width=width, label='Federated training',tick_label = name_list,fc = 'r')
plt.legend()
plt.ylim(0.98, 1.0)
plt.xlabel('Device index')
plt.ylabel('Precision Magnitude')
plt.title('Precision Performance')
plt.show()

fpr_num_list_cen = [0.03723, 0.15134, 0.05588, 0.097077, 0.06834, 0.07593, 0.072978, 0.12398, 0.077182]
fpr_num_list_dis = [0.000, 0.685714, 0.087379, 0.267974, 0.564417, 0.015504, 0.664234, 0.459016, 0.019608]
plt.bar(x, fpr_num_list_cen, width=width, label='Centralized training',fc = 'y')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, fpr_num_list_dis, width=width, label='Federated training',tick_label = name_list,fc = 'r')
plt.legend()
plt.xlabel('Device index')
plt.ylabel('False Postive Rate')
plt.title('FPR Performance')
plt.show()

tr_num_list_cen = [0.35605, 0.105233, 0.12339, 0.103674, 0.04451, 0.04382,0.18166, 0.066662, 0.147692]
tr_num_list_dis = [0.092956, 0.092956, 0.092956, 0.092956, 0.092956, 0.092956, 0.092956, 0.092956, 0.092956]
plt.bar(x, tr_num_list_cen, width=width, label='Centralized training',fc = 'y')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, tr_num_list_dis, width=width, label='Federated training',tick_label = name_list,fc = 'r')
plt.legend()
plt.xlabel('Device index')
plt.ylabel('Threshold Magnitude')
plt.title('Threshold Performance')
plt.show()