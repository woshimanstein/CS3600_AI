# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        # value-iteration
        currIter = 0
        while currIter < self.iterations:
            updatedValues = self.values.copy()
            for state in self.mdp.getStates():
                potentialValue = set()
                if self.mdp.isTerminal(state):
                    updatedValues[state] = self.mdp.getReward(state)
                else:
                    for action in self.mdp.getPossibleActions(state):
                        stateAndProbPairs = self.mdp.getTransitionStatesAndProbs(state, action)
                        expected = 0
                        for pair in stateAndProbPairs:
                            expected += pair[1]*self.values[pair[0]]
                        potentialValue.add(expected)
                    updatedValues[state] = self.mdp.getReward(state) + max(potentialValue)*self.discount
            self.values = updatedValues.copy()
            currIter += 1


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        stateProbPairs = self.mdp.getTransitionStatesAndProbs(state, action)
        expectedValue = 0
        for pair in stateProbPairs:
            expectedValue += pair[1]*self.values[pair[0]]
        return self.mdp.getReward(state) + self.discount*expectedValue



    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        if len(self.mdp.getPossibleActions(state)) == 0:
            return None
        action = self.mdp.getPossibleActions(state)[0]
        value = 0
        stateProbPair = self.mdp.getTransitionStatesAndProbs(state, self.mdp.getPossibleActions(state)[0])
        for pair in stateProbPair:
            value += pair[1] * self.values[pair[0]]
        for i in range(1, len(self.mdp.getPossibleActions(state))):
            stateProbPair = self.mdp.getTransitionStatesAndProbs(state, self.mdp.getPossibleActions(state)[i])
            expectedValue = 0
            for pair in stateProbPair:
                expectedValue += pair[1]*self.values[pair[0]]
            if expectedValue > value:
                value = expectedValue
                action = self.mdp.getPossibleActions(state)[i]
        return action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
