# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    fringe = util.Stack()
    visited = [problem.getStartState()]
    path = dict()
    fringe.push(problem.getStartState())
    goalState = problem.getStartState()
    while (not fringe.isEmpty()):
        currentState = fringe.pop()
        if (problem.isGoalState(currentState)):
            goalState = currentState
            break
        successors = problem.getSuccessors(currentState)
        for state in successors:
            if (state[0] not in visited):
                path[state[0]] = [currentState, state[1]]
                fringe.push(state[0])
                visited.append(currentState)
    actions = list()
    currentState = goalState
    while (currentState != problem.getStartState()):
        actions.append(path[currentState][1])
        currentState = path[currentState][0]
    actions.reverse()
    return actions

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    fringe = util.Queue()
    visited = []
    path = {}
    fringe.push(problem.getStartState())
    goalState = problem.getStartState()
    while (not fringe.isEmpty()):
        currentState = fringe.pop()
        if (currentState not in visited):
            visited.append(currentState)
            if (problem.isGoalState(currentState)):
                goalState = currentState
                break
            successors = problem.getSuccessors(currentState)
            for state in successors:
                if (state[0] not in visited):
                    if (state[0] not in path.keys()):
                        path.update({state[0]:(currentState, state[1])})
                    fringe.push(state[0])
    actions = list()
    currentState = goalState
    while (currentState != problem.getStartState()):
        actions.append(path[currentState][1])
        currentState = path[currentState][0]
    actions.reverse()
    return actions
    

def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    fringe = util.PriorityQueue()
    visited = [problem.getStartState()]
    path = dict()
    distanceTo = {problem.getStartState():0}
    fringe.push(problem.getStartState(), 0)
    goalState = problem.getStartState()
    while (not fringe.isEmpty()):
        currentState = fringe.pop()
        if (problem.isGoalState(currentState)):
            goalState = currentState
            break
        successors = problem.getSuccessors(currentState)
        for state in successors:
            if (state[0] not in visited):
                visited.append(currentState)
                if (not (state[0] in distanceTo) or distanceTo[state[0]] > distanceTo[currentState] + state[2]):
                    distanceTo[state[0]] = distanceTo[currentState] + state[2]
                    path[state[0]] = [currentState, state[1]]
                    fringe.push(state[0], distanceTo[state[0]])
    actions = list()
    currentState = goalState
    while (currentState != problem.getStartState()):
        actions.append(path[currentState][1])
        currentState = path[currentState][0]
    actions.reverse()
    return actions

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    fringe = util.PriorityQueue()
    visited = [problem.getStartState()]
    path = dict()
    distanceTo = {problem.getStartState():0}
    fringe.push(problem.getStartState(), heuristic(problem.getStartState(), problem))
    goalState = problem.getStartState()
    while (not fringe.isEmpty()):
        currentState = fringe.pop()
        if (problem.isGoalState(currentState)):
            goalState = currentState
            break
        successors = problem.getSuccessors(currentState)
        for state in successors:
            visited.append(currentState)
            if (not (state[0] in distanceTo) or distanceTo[state[0]] > distanceTo[currentState] + state[2] + heuristic(state[0], problem)):
                distanceTo[state[0]] = distanceTo[currentState] + state[2]
                path[state[0]] = [currentState, state[1]]
                fringe.push(state[0], distanceTo[state[0]] + heuristic(state[0], problem))
    actions = list()
    currentState = goalState
    while (currentState != problem.getStartState()):
        actions.append(path[currentState][1])
        currentState = path[currentState][0]
    actions.reverse()
    return actions


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
