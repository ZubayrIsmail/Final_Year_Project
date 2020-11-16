import test_metrics
import matplotlib.pyplot as plt
import json
import statistics

filename = 'result1604968042_dlib_.json'
#filename = 'result1604912468_resnet50_.json'

with open(filename, 'r') as f:
    data = json.load(f)

y_score = data['y_score']
y_target = data['y_target']
times = data['runtimes']

prc = test_metrics.precision(y_score, y_target)
rcll = test_metrics.recall(y_score, y_target)
acc = test_metrics.accuracy(y_score, y_target)

print('some info : ')
print('//-------------')
print('model name : ' + data['model'])
print('number of candidates : ' + str(len(data['candidates'])))
print('number of image pairs used : ' + str(len(data['test_data'])))
print('//-------------')
print('')


print('model results metrics :')
print('//-------------')
print('precision : ' + str(prc))
print('recall : ' + str(rcll))
print('accuracy : ' + str(acc))
print('//-------------')
print('')

print('execution time summary')
print('//-------------')
print("average execution time : " + str(statistics.mean(times)))
print('execution time standard deviation : ' + str(statistics.stdev(times)))
print('longest execution time : ' + str(max(times)))
print('shortest execution time : ' + str(min(times)))
print('//-------------')

fig, axs = plt.subplots(1, 1,
                            figsize=(10, 7),
                            tight_layout=True)
axs.hist(times, bins=50)
plt.show()

ideal_thresh = test_metrics.precision_recall_curve(y_score, y_target)

print('threshold for maximum precision : ' + str(ideal_thresh))

