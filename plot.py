import matplotlib.pyplot as plt
import glob
import pickle5 as pickle

exp_glob = 'exp/*lr0.0001*feat20*/*/test/metrics.p'
#exp_glob = 'exp/HeySnipsSequence_batch_size64_epochs250_lr*_tcn_feat20_len3_stacks3_filters16_dil5_bn_skip/*/test/metrics.p'
experiments = glob.glob(exp_glob)

metrics_list = []
names = []
for ex in experiments:
  with open(ex, "rb",) as input_file:
    e = pickle.load(input_file)
    metrics_list.append(e)
    names.append(ex.split('/')[-4].split('250_')[-1])

plt.figure()
x_metric = 'parameters'
y_metric = 'accuracy'
print(metrics_list)
for i, m in enumerate(metrics_list):
 print(m[x_metric], m[y_metric])
 plt.plot(m[x_metric], m[y_metric], "x", label=names[i])

plt.legend()
plt.show()