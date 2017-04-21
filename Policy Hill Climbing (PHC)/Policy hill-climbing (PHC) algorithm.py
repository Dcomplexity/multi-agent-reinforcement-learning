import numpy as np
import matplotlib.pyplot as plt

def generateRandomFromDistribution (distribution):
    randomIndex = 0
    randomSum = distribution[randomIndex]
    randomFlag = np.random.random_sample()
    while randomFlag > randomSum:
        randomIndex += 1
        randomSum += distribution[randomIndex]
    return randomIndex

class agent:
    def __init__(self, initialStrategy = (0.5, 0.5), alpha = 0.1, gammma = 0.1):
        self.alpha = alpha
        self.gamma = gammma
        self.actions = [0, 1]
        self.reward = 0.0
        self.strategy = list(initialStrategy)
        self.actionValues = np.zeros((2))
        # self.actionRewards = np.zeros((2))
        # self.currentActionIndex = 0
        # self.nextActionIndex = 0
        self.currentAction = np.random.choice(self.actions)
        self.currentReward = 0
        self.maxAction = np.random.choice(self.actions)
        self.timeStep = 0
        self.EPSILON = 0.5 / (1 + 0.0001 * self.timeStep)
        self.lengthOfAction = len(self.actions)

    def initialSelfStrategy (self):
        for i in range(self.lengthOfAction):
            self.strategy[i] = 1 / self.lengthOfAction

    def initialActionValues (self):
        for i in range(self.lengthOfAction):
            self.actionValues[i] = 0

    def chooseAction (self):
        if np.random.binomial(1, self.EPSILON) == 1:
            self.currentAction = np.random.choice(self.actions)
        else:
            self.currentAction = self.actions[generateRandomFromDistribution(self.strategy)]

    def getCurrentAction (self):
        return self.currentAction

    def setReward (self, agentReward):
        self.currentReward = agentReward

    def updateActionValues (self, reward):
        self.actionValues[self.currentAction] += (1 - self.alpha) * self.actionValues[self.currentAction] \
                                                 + self.alpha * (self.currentReward + self.gamma * np.amax(self.actionValues[:]))

    def updateStrategy (self):
        self.maxAction = np.argmax(self.actionValues)



def calReward (action_1, action_2):
    reward_1 = 0
    reward_2 = 0
    if (action_1, action_2) == (0, 0):
        reward_1 = 1
        reward_2 = -1
    elif (action_1, action_2) == (1, 0):
        reward_1 = -1
        reward_2 = 1
    elif (action_1, action_2) == (0, 1):
        reward_1 = -1
        reward_2 = 1
    elif (action_1, action_2) == (1, 1):
        reward_1 = 1
        reward_2 = -1
    return (reward_1, reward_2)

a = agent(initialStrategy=(0.5, 0.5))
# for i in range(10):
#     a.chooseAction()
#     print (a.currentAction)
reward1, reward2= calReward(1,1)
a.setReward(reward2)
print (a.reward)
