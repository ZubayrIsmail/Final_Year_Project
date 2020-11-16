import json
import time

filename = 'result1604866261_senet50_.json'
with open(filename, 'r') as f:
    data = json.load(f)

score = data['y_score']

new_scores = []

for i in score:
    new_scores.append(1 - i)

print(new_scores)
data['y_score'] = new_scores

filename = 'result' + str(int(time.time())) + '_' + data['model'] + '_' + '.json'

with open(filename, 'w') as f:
    json.dump(data, f)

