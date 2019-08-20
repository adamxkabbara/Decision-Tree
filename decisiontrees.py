import numpy as np
from math import log
import operator

# Constants
FEATURE_COUNT = 22

# node class
# Used to create node for the decision tree
# pure: set true if all data points have the same label
# children: a list of all the data points for the current node
class node:

    def __init__(self, dataSet=None, yes=None, no=None, label=None, split=None, prune=None, isLeaf=False):
        self.dataSet = dataSet
        self.yes = yes
        self.no = no
        self.label = label
        self.split = split
        self.prune = prune
        self.isLeaf = isLeaf
        self.isPure()

    def isPure(self):
        value = self.dataSet[0][-1]
        for index in range(len(self.dataSet) - 1):
            if (self.dataSet[index + 1][-1] != value):
                self.purity = False
                return
        self.purity = True

def entropy(probabilities):
    prob_count = len(probabilities)
    total = 0

    for index in range(prob_count):
        if (probabilities[index] == 0):
            continue
        else:
            total += probabilities[index] * log(probabilities[index])
    return ((-1) * total)

def conditional_entropy(Y_one, Y_zero, Z_yes, Z_no, split, feature):
    prob_Z_yes = len(Z_yes) / float((len(Z_yes) + len(Z_no)))
    prob_Y_yes_Z_yes = 0.0
    prob_Y_no_Z_yes = 0.0
    prob_Y_yes_Z_no = 0.0

    count = 0
    for row in Y_one:
        if (row[feature] <= split):
            count+=1
    prob_Y_yes_Z_yes =  count / float(len(Z_yes))

    prob_Y_no_Z_yes = 1 - prob_Y_yes_Z_yes
    if (prob_Y_yes_Z_yes == 0):
        prob_Y_yes_Z_yes = 1
    if (prob_Y_no_Z_yes == 0):
        prob_Y_no_Z_yes = 1

    H_Y_Z_yes = -1.0 * (prob_Y_yes_Z_yes * log(prob_Y_yes_Z_yes) + prob_Y_no_Z_yes * log(prob_Y_no_Z_yes))

    count = 0
    for row in Y_one:
        if (row[feature] > split):
            count+=1
    if (len(Z_no) == 0):
        prob_Y_yes_Z_no = 1
    else:
        prob_Y_yes_Z_no =  count / float(len(Z_no))

    prob_Y_no_Z_no = 1 - prob_Y_yes_Z_no

    if (prob_Y_yes_Z_no == 0):
        prob_Y_yes_Z_no = 1
    if (prob_Y_no_Z_no == 0):
        prob_Y_no_Z_no = 1
    H_Y_Z_no = -1 * (prob_Y_yes_Z_no * log(prob_Y_yes_Z_no) + prob_Y_no_Z_no * log(prob_Y_no_Z_no))

    return ((1 - prob_Z_yes) * H_Y_Z_no + prob_Z_yes * H_Y_Z_yes)

def build_tree(training_data):
    queue = [] # stores all the impure nodes
    root = node(dataSet=training_data)
    queue.append(root)

    while(len(queue) != 0):
        currNode = queue.pop()
        infoGain = []
        training_length = len(currNode.dataSet)

        # Check if it is a pure node and if so then give it a label
        if currNode.purity is True:
            currNode.label = currNode.dataSet[0][-1]
            currNode.isLeaf = True
            continue

        for feature in range(FEATURE_COUNT):
            # sort the data to find the splits
            sortedCurr = currNode.dataSet[np.argsort(currNode.dataSet[:, feature])]
            # value* is used to find the splits

            tempSplit = 0
            Y_one = []
            Y_zero = []
            array_yes = []
            array_no = []
            (value1, value2) = (0, 0)

            for row in range(training_length):
                if (sortedCurr[row][FEATURE_COUNT] == 1):
                    Y_one.append(sortedCurr[row][:])
                else:
                    Y_zero.append(sortedCurr[row][:])
            #print(Y_one)
            for row in range(training_length - 1):
                (value1, value2) = (sortedCurr[row][feature], sortedCurr[row + 1][feature])
                
                if (value1 == value2):
                    continue

                tempSplit = (value2 + value1) / 2.0

                for row2 in sortedCurr:
                    if (row2[feature] <= tempSplit):
                        array_yes.append(row2[:])
                    else:
                        array_no.append(row2[:])
            
                infoGain.append((feature, conditional_entropy(Y_one, Y_zero, array_yes, array_no, tempSplit, feature), tempSplit))
                #print(infoGain)
                array_no.clear()
                array_yes.clear()

        infoGain.sort(key=operator.itemgetter(1))
        newNode = splitRule(currNode, infoGain[0][0], infoGain[0][2])
        infoGain.clear()


        queue.append(newNode[0])
        queue.append(newNode[1])
        #print(newNode[0].dataSet)

        # calculate the entorpy

    return root

def splitRule(currNode, feature, split):
    yes = []
    no = []
    
    for row in currNode.dataSet:
        if (row[feature] <= split):
            yes.append(row)
        if (row[feature] > split):
            no.append(row)

    np_yes = np.array(yes)
    np_no = np.array(no)

    #print(feature)
    yesNode = node(dataSet=np_yes)
    noNode = node(dataSet=np_no)

    currNode.yes = yesNode
    currNode.no = noNode
    currNode.split = (feature, split)

    return (yesNode, noNode)


def readFileToNumArray(file_name):

    # Read lines from file
    file = open(file_name, 'r')
    lines = file.readlines()
    file.close()

    array = np.empty([len(lines), FEATURE_COUNT + 1]) # add one for label at the end
    index = 0
    # sanitize the input
    for line in lines:
        line = line.split()
        array[index] = line
        index+=1

    return array

def treverse_tree(tree, point):
    currNode = tree
    while (currNode.isLeaf == False):
        if (point[currNode.split[0]] <= currNode.split[1]):
            currNode = currNode.yes
        else:
            currNode = currNode.no
    return currNode.label

def getAccuracy(training_data, labels):
    error_count = 0
    for index in range(len(labels)):
        #print(training_data[index][-1], labels[index])
        if (training_data[index][-1] != labels[index]):
            error_count+=1

    return (error_count / float(len(labels)))

def error_rate(tree, training_data):
    labels = []
    for row in training_data:
        labels.append(treverse_tree(tree, row))

    return getAccuracy(training_data, labels)

def main():
    training_set = readFileToNumArray('train.txt') # get the training data for building tree
    test_set = readFileToNumArray('test.txt') # get the training data for building tree

    tree = build_tree(training_set)

    #root
    print(tree.split, len(tree.dataSet))

    #level 2
    print(tree.yes.split, tree.yes.dataSet.shape[0])
    print(tree.no.split, tree.no.dataSet.shape[0])

    #level 3
    print(tree.yes.yes.split, tree.yes.yes.dataSet.shape[0])
    print(tree.yes.no.split, tree.yes.no.dataSet.shape[0])

    print(tree.no.yes.split, tree.no.yes.dataSet.shape[0])
    print(tree.no.no.split, tree.no.no.dataSet.shape[0])

    # Training Error Rate
    print(error_rate(tree, training_set))

    # Test Error Rate
    print(error_rate(tree, test_set))

main()
