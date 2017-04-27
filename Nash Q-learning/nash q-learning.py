import numpy as np
from itertools import permutations

# world height
WORLD_HEIGHT =3

# world width
WORLD_WIDTH = 3

# possible actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

gridIndex = []
for i in range(0, WORLD_HEIGHT):
    for j in range(0, WORLD_WIDTH):
        gridIndex.append(WORLD_WIDTH * i + j)

actions = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]
statesAllOne = []
statesValidActions = {}

for i in permutations(gridIndex, 2):
    statesAllOne.append(i)

for i in gridIndex:
    statesValidActions[i] = []

for i in range(0, WORLD_HEIGHT):
    for j in range(0, WORLD_WIDTH):
        gridIndexNumber = WORLD_WIDTH * i + j
        if i != WORLD_HEIGHT - 1:
            statesValidActions[gridIndexNumber].append(ACTION_UP)
        if i != 0:
            statesValidActions[gridIndexNumber].append(ACTION_DOWN)
        if j != 0:
            statesValidActions[gridIndexNumber].append(ACTION_LEFT)
        if j != WORLD_WIDTH - 1:
            statesValidActions[gridIndexNumber].append(ACTION_RIGHT)

class agent:
    def __init__(self, agentIndex = 0):
        self.goalState = ()
        self.qTable = {}
        self.singleQValue = {}
        self.currentState = ()
        self.nextState = ()
        self.strategy = {}
        self.agentIndex = agentIndex

    def initialSelfStrategy (self):
        for i in statesAllOne:
            self.strategy[i] = {}
            for j in statesValidActions[i[self.agentIndex]]:
                self.strategy[i][j] = 0

    def initialSelfQTable (self):
        for i in statesAllOne:
            self.qTable[i] = {}
            for j_1 in statesValidActions[i[0]]:
                for j_2 in statesValidActions[i[1]]:
                    self.qTable[i][(j_1, j_2)] = 0

    def initialSingleQValue (self):
        for i in statesAllOne:
            self.singleQValue[i] = {}
            for j in statesValidActions[i[self.agentIndex]]:
                self.singleQValue[i][j] = 0

    def singleQLearning (self, alpha, gamma):
        self.alpha = alpha
        self.gamma = gamma
        



def nextGridIndex (action, gridIndex):
    action = action
    index_i = int(gridIndex / 3)
    index_j = gridIndex - index_i * 3
    if (action == 0):
        index_i += 1
    elif (action == 1):
        index_i -= 1
    elif (action == 2):
        index_j -= 1
    elif (action == 3):
        index_j += 1
    nextIndex = index_i * 3 + index_j
    return nextIndex

def gridGameOne(action_0, action_1, currentState):
    action_0 = action_0
    action_1 = action_1
    currentState = currentState
    reward_0 = 0
    reward_1 = 0
    endGameFlag = 0

    currentIndex_0 = currentState[0]
    currentIndex_1 = currentState[1]
    nextIndex_0 = nextGridIndex(action_0, currentState[0])
    nextIndex_1 = nextGridIndex(action_1, currentState[1])

    if (nextIndex_0 == nextIndex_1):
        reward_0 = -1
        reward_1 = -1
        nextState = ()
    if (nextIndex_0 == 8 and nextIndex_1 == 6):
        reward_0 = 100
        reward_1 = 100
        nextState = (nextIndex_0, nextIndex_1)
        endGameFlag = 1
    elif (nextIndex_0 == 8):
        reward_0 = 100
        nextState = (nextIndex_0, nextIndex_1)
        endGameFlag = 1
        if (nextIndex_1 == 8):
            reward_1 = -1
        else:
            reward_1 = 0
    elif (nextIndex_1 == 6):
        reward_1 = 100
        nextState = (nextIndex_0, nextIndex_1)
        endGameFlag = 1
        if (nextIndex_0 == 6):
            reward_0 = -1
        else:
            reward_0 = 0
    elif (nextIndex_0 == nextIndex_1):
        reward_0 = -1
        reward_1 = -1
        nextState = (currentIndex_0, currentIndex_1)
    else:
        reward_0 = 0
        reward_1 = 0
        nextState = (nextIndex_0, nextIndex_1)
    return reward_0, reward_1, nextState



a1 = agent()
a1.initialSelfStrategy()
a1.initialSelfQTable()
reward_0, reward_1, nextState = gridGameOne(0, 3, (1, 3))
