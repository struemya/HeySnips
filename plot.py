import matplotlib.pyplot as plt
import glob
import pickle5 as pickle

#Set experiment glob here
exp_glob = 'exp/*lr0.001*feat*/*/test/metrics.p'
experiments = glob.glob(exp_glob)

metrics_list = []
names = []
for ex in experiments:
  with open(ex, "rb",) as input_file:
    e = pickle.load(input_file)
    metrics_list.append(e)
    names.append(ex)

plt.figure()
x_metric = 'parameters'
y_metric = 'fp'
print(metrics_list)
for i, m in enumerate(metrics_list):
 print(m[x_metric], m[y_metric])
 plt.plot(m[x_metric], m[y_metric], "o", label=names[i])

plt.legend()
plt.show()