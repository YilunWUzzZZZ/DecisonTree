import csv
import pickle

with open('/Users/wuyilun/Downloads/reassigned_test.csv', 'r') as traindatafile:
    data = csv.reader(traindatafile)
    data = list(data)
    labels = []
    for record in data:
        del record[2]
        labels.append(record[-1])
        del record[-2:]

print(data[0])
print(labels[:10])

with open('test_data.pickle', 'wb') as target:
    pickle.dump(data, target)

with open('test_labels.pickle', 'wb') as target:
    pickle.dump(labels, target)

