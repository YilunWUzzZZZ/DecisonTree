from DecisionTree import C4_5DT
import pickle
import numpy

with open('train_data.pickle', 'rb') as fp:
    train_data = pickle.load(fp)

with open('train_labels.pickle', 'rb') as fp:
    train_labels = pickle.load(fp)

with open('test_data.pickle', 'rb') as fp:
    test_data = pickle.load(fp)

with open('test_labels.pickle', 'rb') as fp:
    test_labels = pickle.load(fp)

myAttriDiscription = [None, 8, None, 7, 14, 6, 5, 2, None, None, None]

myTree = C4_5DT(myAttriDiscription)

myTree.fit(train_data, train_labels)

labelVec_train = numpy.array([int(l) for l in train_labels])
labelVec_test = numpy.array([int(l) for l in test_labels])

predictions_tr = numpy.array(myTree.eval(train_data))
predictions_te = numpy.array(myTree.eval(test_data))

acc_tr = (predictions_tr == labelVec_train).sum() / labelVec_train.size
acc_te = (predictions_te == labelVec_test).sum() / labelVec_test.size

print("Acc on training Set: %.2f\n Acc on test Set: %.2f"%(acc_tr, acc_te))

