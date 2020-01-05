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
import collections

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
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        while self.iterations > 0:
            # Make copy to store k + 1 values
            valuesCopy = self.values.copy()
            for state in self.mdp.getStates():
                vals = []
                for action in self.mdp.getPossibleActions(state):
                    value = 0
                    # Create list of values for actions
                    for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                        reward = self.mdp.getReward(state, action, nextState)
                        value += prob * (reward + self.discount * self.getValue(nextState))
                    vals.append(value)
                if vals == []:
                    valuesCopy[state] = 0
                else:
                    valuesCopy[state] = max(vals)
            self.values = valuesCopy
            self.iterations -= 1
        return self.values


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
        value = 0
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, nextState)
            value += prob * (reward + self.discount * self.values[nextState])
        return value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        actionToDo = "north"
        maxVal = -float('inf')
        for action in self.mdp.getPossibleActions(state):
            qval = self.getQValue(state, action)
            if qval > maxVal:
                maxVal = qval
                actionToDo = action
        return actionToDo

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        while self.iterations > 0:
            # Make copy to store k + 1 values
            for state in self.mdp.getStates():

                vals = []
                for action in self.mdp.getPossibleActions(state):
                    value = 0
                    # Create list of values for actions
                    for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                        reward = self.mdp.getReward(state, action, nextState)
                        value += prob * (reward + self.discount * self.getValue(nextState))
                    vals.append(value)
                if vals == []:
                    self.values[state] = 0
                else:
                    self.values[state] = max(vals)
                self.iterations -= 1
                if self.iterations == 0:
                    return self.values
        return self.values

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        queue = util.PriorityQueue()
        predecessors = {state: set() for state in self.mdp.getStates()}
        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
                continue
            for action in self.mdp.getPossibleActions(state):
                for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    predecessors[nextState].add(state)

        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
                continue
            maxVal = -float('inf')
            for action in self.mdp.getPossibleActions(state):
                qval = self.getQValue(state, action)
                if qval > maxVal:
                    maxVal = qval
            diff = abs(maxVal - self.values[state])
            queue.update(state, -diff)

        while self.iterations > 0:
            if queue.isEmpty():
                break
            next = queue.pop()
            maxVal = -float('inf')
            if not self.mdp.isTerminal(next):
                for action in self.mdp.getPossibleActions(next):
                    qval = self.getQValue(next, action)
                    if qval > maxVal:
                        maxVal = qval
                self.values[next] = maxVal
            for predecessor in predecessors[next]:
                maxVal = -float('inf')

                for action in self.mdp.getPossibleActions(predecessor):
                    qval = self.getQValue(predecessor, action)
                    if qval > maxVal:
                        maxVal = qval
                diff = abs(self.values[predecessor] - maxVal)
                if diff > self.theta:
                    queue.update(predecessor, -diff)
            self.iterations -= 1
