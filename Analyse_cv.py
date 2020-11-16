import test_metrics
import matplotlib.pyplot as plt
import json
import statistics

filename = 'result1604872695_CV_.json'
with open(filename, 'r') as f:
    data = json.load(f)

y_score = data['y_score']
y_target = data['y_target']
times = data['runtimes']

angle_errors =y_score.count(-1)
processing_errors = y_score.count(-3)
prc = test_metrics.precision(y_score, y_target, threshold=24)
rcll = test_metrics.recall(y_score, y_target, threshold=24)
acc = test_metrics.accuracy(y_score, y_target, threshold=24)

print('some info : ')
print('//-------------')
print('model name : ' + data['model'])
print('number of candidates : ' + str(len(data['candidates'])))
print('number of image pairs used : ' + str(len(data['test_data'])))
print('//-------------')
print('')


print('model results metrics :')

print('errors occuring due it angle of face : ' +  str(angle_errors))
print('errors occuring due to failure to predict : ' + str(processing_errors))
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

ideal_thresh = test_metrics.precision_recall_curve_cv(y_score, y_target, max(y_score))

print('threshold for maximum precision : ' + str(ideal_thresh))