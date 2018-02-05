__author__ = 'talhaahsan'
import pybrain
from pybrain.supervised.trainers import backprop
from pybrain.supervised.trainers import evolino
from pybrain.tools.datasets import mnist
from pybrain.tools.shortcuts import buildNetwork
from pybrain.tools.validation import ModuleValidator
import time

#Gets the training and test sets for MNIST stuff
(train, test) = mnist.makeMnistDataSets("/Users/talhaAhsan/PycharmProjects/PythonCrashCourse/MNIST_data")
print("Training and Testing sets have been formed")

StartTime = time.time()
numberOfNetworks = 1
numberOfLayers = 10
networks = []
trainers = []
scoresIndividual = []
for i in range(0, numberOfNetworks):
    print("Appending and training network " + i.__str__())
    networks.append(buildNetwork(28*28, numberOfLayers, 10))
    trainers.append(backprop.BackpropTrainer(networks[i], dataset=train))
    scoresIndividual.append(trainers[i].trainOnDataset(train))

print("Individual network scores:")
for score in scoresIndividual:
    print(score)

items = 0
errors = 0
print(trainers[0].testOnData(test[0]))

print("--- %s seconds ---" % (time.time() - StartTime).__str__())