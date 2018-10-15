import math
from enum import Enum


class stopCondition():
    def __init__(self, MinSamples=100, shrehold = 1e-5):
        self.MinSamples = MinSamples
        self.shrehold = shrehold

class NodeType(Enum):
    LEAF = 0
    ROOT = 1

class AttriType(Enum):
    Continuous = 1
    Discrete = 0

class Node():

    def __init__(self, nodeType, attribute = None, branch_num = 0, prediction = None, discretizer=lambda x: int(x)):

        self.nodeType = nodeType
        self.entropy = 0
        if self.nodeType == NodeType.ROOT:
            self.branch_num = branch_num
            self.attribute = attribute
            self.discretizer = discretizer
            self.childrens = []

        else:
            self.prediction = prediction


    def predict(self, sample):
        if self.nodeType == NodeType.ROOT:
            return self.childrens[self.discretizer(sample[self.attribute.index])].predict(sample)

        else:
            return self.prediction

class Attribute():

    def __init__(self, index, valueNum=None):
        self.attriType = AttriType.Continuous if valueNum == None else AttriType.Discrete
        self.index = index
        self.valueNum = valueNum

class C4_5DT():

    def __init__(self, attriDiscriptions, MinSamples=100, shrehold = 0.01, classNum=2):
        self.classNum = classNum
        self.attributes = self.__constructAtrris(attriDiscriptions)
        self.stopCondition = stopCondition(MinSamples=MinSamples, shrehold=shrehold)


    def __constructAtrris(self,attriDiscrips):
        attris = []
        for i, valueNum in enumerate(attriDiscrips):
            a = Attribute(i, valueNum)
            attris.append(a)

        return attris

    def fit(self,tr, labels):
        self.root = self.__buildTree(tr, labels, self.attributes)
        return self.root


    def eval(self, dataset):
        predictions = []
        for sample in dataset:
            result = self.root.predict(sample)
            predictions.append(result)
        return predictions


    def __buildTree(self, tr, labels, attributes, parentEntropy=1, preMostLikelyClass=0):


        if len(tr) < self.stopCondition.MinSamples:
            prediction = preMostLikelyClass
            return Node(NodeType.LEAF, prediction=prediction)

        mostLikelyClass = self.__mostLikelyClass(labels)

        if len(attributes) == 0:
            prediction = mostLikelyClass
            return Node(NodeType.LEAF, prediction=prediction)

        if self.__allSameClass(labels):
            prediction = int(labels[0])
            return Node(NodeType.LEAF, prediction=prediction)

        discretizer = []
        subTrainSets = []
        subLabels = []
        splitEntropys = []
        Gains = []
        GainRatios = []

        for attribute in attributes:
            Gain, GainRatio, discretizerTemp, subTrainSetsTemp, subLabelsTemp, splitEntropyTemp = self.__calculateGainRatio(tr, labels, attribute, parentEntropy)
            Gains.append(Gain)
            GainRatios.append(GainRatio)
            discretizer.append(discretizerTemp)
            subTrainSets.append(subTrainSetsTemp)
            subLabels.append(subLabelsTemp)
            splitEntropys.append(splitEntropyTemp)

        averagedGain = sum(Gains)/len(attributes)

        candidates = []

        for i, (gain,gainratio) in enumerate(zip(Gains, GainRatios)):
            if gain >= averagedGain:
                candidates.append((i, gain, gainratio))

        winner = max(candidates, key=lambda x:x[2])

        if winner[1] < self.stopCondition.shrehold:
            prediction = mostLikelyClass
            return Node(NodeType.LEAF, prediction=prediction)

        selectedAttriIndex = winner[0]
        selectedAttri = attributes[selectedAttriIndex]
        discretizer = discretizer[selectedAttriIndex]
        subTrainSets = subTrainSets[selectedAttriIndex]
        subLabels = subLabels[selectedAttriIndex]
        splitEntropys = splitEntropys[selectedAttriIndex]

        branch_num = selectedAttri.valueNum if selectedAttri.valueNum is not None else 2
        newTree = Node(NodeType.ROOT, attribute=selectedAttri, branch_num=branch_num, discretizer=discretizer)

        #delete attribute used
        subAttributes = attributes.copy()
        subAttributes.pop(selectedAttriIndex)

        for i in range(newTree.branch_num):
            newTree.childrens.append(self.__buildTree(subTrainSets[i], subLabels[i], subAttributes, parentEntropy=splitEntropys[i], preMostLikelyClass=mostLikelyClass))

        return newTree

    def __calculateGainRatio(self, tr, labels, attribute, parentEntropy, epsilon = 1e-6):
        if attribute.attriType == AttriType.Discrete:
            index = attribute.index
            table = [[0 for _ in range(self.classNum)] for _ in range(attribute.valueNum)]
            subTrainSets = [[] for _ in range(attribute.valueNum)]
            subLabels = [[] for _ in range(attribute.valueNum)]

            #iterate over the training set
            for i,sample in enumerate(tr):
                catogory = int(sample[index])
                label = int(labels[i])
                table[catogory][label] +=1
                subTrainSets[catogory].append(sample)
                subLabels[catogory].append(label)

            #calculate entropy
            entropy = 0
            IV = 0
            totalSampleInOrigin = len(tr)
            splitEntropys = []

            for j in range(attribute.valueNum):
                totalSample = sum(table[j])
                splitEntropy = self.__entropy(table[j], totalSample) if totalSample != 0 else 0
                splitEntropys.append(splitEntropy)
                entropy += totalSample / totalSampleInOrigin * splitEntropy
                p = totalSample / totalSampleInOrigin
                IV += p * math.log2(p + epsilon)

            Gain = parentEntropy-entropy
            GainRatio = Gain/IV
            return Gain, GainRatio, lambda x: int(x), subTrainSets, subLabels, splitEntropys


        else:

            subTrainSets = [[] for _ in range(2)]
            subLabels = [[] for _ in range(2)]
            splitEntropys = []

            #count sample number for each classs
            counter = [0 for _ in range(self.classNum)]
            total = len(labels)
            for l in labels:
                counter[int(l)] += 1

            index = attribute.index
            tr = [(tr[i], int(labels[i])) for i in range(total)]
            tr = sorted(tr, key=lambda x: float(x[0][index]))
            prevalue = float(tr[0][0][index])
            firstSplit = [0 for _ in range(self.classNum)]
            bestSplitVal = None
            bestRatio = None
            minEntropy = 100

            for cnum, sample in enumerate(tr):
                cvalue = float(sample[0][index])
                label = sample[1]

                if  prevalue != cvalue:
                    splitVal = (prevalue + cvalue)/2
                    classCounts = [counter[k] - firstSplit[k] for k in range(self.classNum)]
                    firstSplitNum = cnum
                    secondSplitNum = total - cnum
                    entropys = [self.__entropy(firstSplit, firstSplitNum), self.__entropy(classCounts, secondSplitNum)]
                    entropy = firstSplitNum / total * entropys[0] + secondSplitNum / total * entropys[1]

                    if entropy < minEntropy:
                        splitEntropys = entropys
                        minEntropy = entropy
                        bestSplitVal = splitVal
                        bestRatio = self.__entropy([firstSplitNum, secondSplitNum], total)
                        subTrainSets[0] = list(map(lambda x:x[0], tr[:firstSplitNum]))
                        subTrainSets[1] = list(map(lambda x:x[0], tr[firstSplitNum:]))
                        subLabels[0] = list(map(lambda x:x[1], tr[:firstSplitNum]))
                        subLabels[1] = list(map(lambda x: x[1], tr[firstSplitNum:]))

                    prevalue = cvalue

                firstSplit[label] += 1

            if bestSplitVal is not None:
                bestGain = parentEntropy - minEntropy
                bestGainRatio = bestGain / (bestRatio + epsilon)
                discretizer = lambda x: 1 if float(x) >= bestSplitVal else 0

            else:
                bestGain = 0
                bestGainRatio = 0
                discretizer = None

            return bestGain, bestGainRatio, discretizer, subTrainSets, subLabels, splitEntropys


    def __entropy(self, counts, total, classNum=2, epsilon=1e-6):

        possibs = map(lambda x: -x/total * math.log(x/total+epsilon, classNum), counts)
        return sum(possibs)


    def __mostLikelyClass(self, labels):

        counter = [0 for _ in range(self.classNum)]
        for l in labels:
            counter[int(l)] += 1

        return counter.index(max(counter))

    def __allSameClass(self, labels):
        c = int(labels[0])
        for l in labels:
            if int(l) != c:
                return False

        return True
