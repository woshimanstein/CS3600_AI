from math import log, sqrt
import sys
from scipy.stats.distributions import chi2


class Node:
  """
  A simple node class to build our tree with. It has the following:
  
  children (dictionary<str,Node>): A mapping from attribute value to a child node
  attr (str): The name of the attribute this node classifies by. 
  islead (boolean): whether this is a leaf. False.
  """
  
  def __init__(self,attr):
    self.children = {}
    self.attr = attr
    self.isleaf = False

class LeafNode(Node):
    """
    A basic extension of the Node class with just a value.
    
    value (str): Since this is a leaf node, a final value for the label.
    islead (boolean): whether this is a leaf. True.
    """
    def __init__(self,value):
        self.value = value
        self.isleaf = True
    
class Tree:
  """
  A generic tree implementation with which to implement decision tree learning.
  Stores the root Node and nothing more. A nice printing method is provided, and
  the function to classify values is left to fill in.
  """
  def __init__(self, root=None):
    self.root = root

  def prettyPrint(self):
    print str(self)
    
  def preorder(self,depth,node):
    if node is None:
      return '|---'*depth+str(None)+'\n'
    if node.isleaf:
      return '|---'*depth+str(node.value)+'\n'
    string = ''
    for val in node.children.keys():
      childStr = '|---'*depth
      childStr += '%s = %s'%(str(node.attr),str(val))
      string+=str(childStr)+"\n"+self.preorder(depth+1, node.children[val])
    return string    

  def count(self,node=None):
    if node is None:
      node = self.root
    if node.isleaf:
      return 1
    count = 1
    for child in node.children.values():
      if child is not None:
        count+= self.count(child)
    return count  

  def __str__(self):
    return self.preorder(0, self.root)
  
  def classify(self, classificationData):
    """
    Uses the classification tree with the passed in classificationData.`
    
    Args:
        classificationData (dictionary<string,string>): dictionary of attribute values
    Returns:
        str
        The classification made with this tree.
    """
    currNode = self.root
    while not currNode.isleaf:
        value = classificationData[currNode.attr]
        currNode = currNode.children[value]
    return currNode.value
  
def getPertinentExamples(examples,attrName,attrValue):
    """
    Helper function to get a subset of a set of examples for a particular assignment 
    of a single attribute. That is, this gets the list of examples that have the value 
    attrValue for the attribute with the name attrName.
    
    Args:
        examples (list<dictionary<str,str>>): list of examples
        attrName (str): the name of the attribute to get counts for
        attrValue (str): a value of the attribute
    Returns:
        list<dictionary<str,str>>
        The new list of examples.
    """
    newExamples = []
    for example in examples:
        if example[attrName] == attrValue:
            newExamples.append(example)
    return newExamples
  
def getClassCounts(examples,className):
    """
    Helper function to get a dictionary of counts of different class values
    in a set of examples. That is, this returns a dictionary where each key 
    in the list corresponds to a possible value of the class and the value
    at that key corresponds to how many times that value of the class 
    occurs.
    
    Args:
        examples (list<dictionary<str,str>>): list of examples
        className (str): the name of the class
    Returns:
        dictionary<string,int>
        This is a dictionary that for each value of the class has the count
        of that class value in the examples. That is, it maps the class value
        to its count.
    """
    classCounts = {}
    for example in examples:
        value = example[className]
        if value in classCounts:
            classCounts[value] += 1
        else:
            classCounts.update({value:1})
    return classCounts

def getMostCommonClass(examples,className):
    """
    A freebie function useful later in makeSubtrees. Gets the most common class
    in the examples. See parameters in getClassCounts.
    """
    counts = getClassCounts(examples,className)
    return max(counts, key=counts.get) if len(examples)>0 else None

def getAttributeCounts(examples,attrName,attrValues,className):
    """
    Helper function to get a dictionary of counts of different class values
    corresponding to every possible assignment of the passed in attribute. 
	  That is, this returns a dictionary of dictionaries, where each key  
	  corresponds to a possible value of the attribute named attrName and holds
 	  the counts of different class values for the subset of the examples
 	  that have that assignment of that attribute.
    
    Args:
        examples (list<dictionary<str,str>>): list of examples
        attrName (str): the name of the attribute to get counts for
        attrValues (list<str>): list of possible values for the attribute
        className (str): the name of the class
    Returns:
        dictionary<str,dictionary<str,int>>
        This is a dictionary that for each value of the attribute has a
        dictionary from class values to class counts, as in getClassCounts
    """
    attributeCounts={}
    for value in  attrValues:
        attributeCounts[value] = {}
    for example in examples:
        if example[className] in attributeCounts[example[attrName]]:
            attributeCounts[example[attrName]][example[className]] += 1
        else:
            attributeCounts[example[attrName]].update({example[className]:1})
    return attributeCounts
        

def setEntropy(classCounts):
    """
    Calculates the set entropy value for the given list of class counts.
    This is called H in the book. Note that our labels are not binary,
    so the equations in the book need to be modified accordingly. Note
    that H is written in terms of B, and B is written with the assumption 
    of a binary value. B can easily be modified for a non binary class
    by writing it as a summation over a list of ratios, which is what
    you need to implement.
    
    Args:
        classCounts (list<int>): list of counts of each class value
    Returns:
        float
        The set entropy score of this list of class value counts.
    """
    #YOUR CODE HERE
    entropy = 0
    for count in classCounts:
        entropy -= float(count) / sum(classCounts) * log(float(count) / sum(classCounts), 2)
    return entropy
   

def remainder(examples,attrName,attrValues,className):
    """
    Calculates the remainder value for given attribute and set of examples.
    See the book for the meaning of the remainder in the context of info 
    gain.
    
    Args:
        examples (list<dictionary<str,str>>): list of examples
        attrName (str): the name of the attribute to get remainder for
        attrValues (list<string>): list of possible values for attribute
        className (str): the name of the class
    Returns:
        float
        The remainder score of this value assignment of the attribute.
    """
    remainderValue = 0
    for attrValue in attrValues:
        pertinentExamples = getPertinentExamples(examples, attrName, attrValue)
        remainderValue += float(len(pertinentExamples)) / len(examples) * setEntropy(getClassCounts(pertinentExamples, className).values())
    return remainderValue
          
def infoGain(examples,attrName,attrValues,className):
    """
    Calculates the info gain value for given attribute and set of examples.
    See the book for the equation - it's a combination of setEntropy and
    remainder (setEntropy replaces B as it is used in the book).
    
    Args:
        examples (list<dictionary<str,str>>): list of examples
        attrName (str): the name of the attribute to get remainder for
        attrValues (list<string>): list of possible values for attribute
        className (str): the name of the class
    Returns:
        float
        The gain score of this value assignment of the attribute.
    """
    return setEntropy(getClassCounts(examples, className).values()) - remainder(examples, attrName, attrValues, className)
  
def giniIndex(classCounts):
    """
    Calculates the gini value for the given list of class counts.
    See equation in instructions.
    
    Args:
        classCounts (list<int>): list of counts of each class value
    Returns:
        float
        The gini score of this list of class value counts.
    """
    result = 1
    for count in classCounts:
        result -= (float(count) / sum(classCounts)) ** 2
    return result
  
def giniGain(examples,attrName,attrValues,className):
    """
    Return the inverse of the giniD function described in the instructions.
    The inverse is returned so as to have the highest value correspond 
    to the highest information gain as in entropyGain. If the sum is 0,
    return sys.maxint.
    
    Args:
        examples (list<dictionary<str,str>>): list of examples
        attrName (str): the name of the attribute to get counts for
        attrValues (list<string>): list of possible values for attribute
        className (str): the name of the class
    Returns:
        float
        The summed gini index score of this list of class value counts.
    """
    result = 0
    for attrValue in attrValues:
        pertinentExamples = getPertinentExamples(examples, attrName, attrValue)
        result += float(len(pertinentExamples)) / len(examples) * giniIndex(getClassCounts(pertinentExamples, className).values())
    if result == 0:
        return sys.maxint
    else:
        return 1 / result
    
def makeTree(examples, attrValues,className,setScoreFunc,gainFunc):
    """
    Creates the classification tree for the given examples. Note that this is implemented - you
    just need to imeplement makeSubtrees.
    
    Args:
        examples (list<dictionary<str,str>>): list of examples
        attrValues (dictionary<string,list<string>>): list of possible values for attribute
        className (str): the name of the class
        classScoreFunc (func): the function to score classes (ie setEntropy or giniIndex)
        gainFunc (func): the function to score gain of attributes (ie infoGain or giniGain)
    Returns:
        Tree
        The classification tree for this set of examples
    """
    remainingAttributes=attrValues.keys()
    return Tree(makeSubtrees(remainingAttributes,examples,attrValues,className,getMostCommonClass(examples,className),setScoreFunc,gainFunc))
    
def makeSubtrees(remainingAttributes,examples,attributeValues,className,defaultLabel,setScoreFunc,gainFunc):
    """
    Creates a classification tree Node and all its children. This returns a Node, which is the root
    Node of the tree constructed from the passed in parameters. This should be implemented recursively,
    and handle base cases for zero examples or remainingAttributes as covered in the book.    

    Args:
        remainingAttributes (list<string>): the names of attributes still not used
        examples (list<dictionary<str,str>>): list of examples
        attrValues (dictionary<string,list<string>>): list of possible values for attribute
        className (str): the name of the class
        defaultLabel (string): the default label
        setScoreFunc (func): the function to score classes (ie setEntropy or giniIndex)
        gainFunc (func): the function to score gain of attributes (ie infoGain or giniGain)
    Returns:
        Node or LeafNode
        The classification tree node optimal for the remaining set of attributes.
    """
    if len(examples) == 0: # empty examples
        return LeafNode(defaultLabel)
    elif len(getClassCounts(examples, className)) == 1: # only one class
        return LeafNode(examples[0][className])
    elif len(remainingAttributes) == 0: # no more attributes available
        return LeafNode(getMostCommonClass(examples, className))
    else:
        bestAttribute = remainingAttributes[0]
        maxGain = gainFunc(examples, bestAttribute, attributeValues[bestAttribute], className)
        for i in range(1, len(remainingAttributes)):
            newGain = gainFunc(examples, remainingAttributes[i], attributeValues[remainingAttributes[i]], className)
            if newGain > maxGain:
                maxGain = newGain
                bestAttribute = remainingAttributes[i]
        updatedAttributes = remainingAttributes[:]
        updatedAttributes.remove(bestAttribute)
        node = Node(bestAttribute)
        for value in attributeValues[bestAttribute]:
            subset = getPertinentExamples(examples, bestAttribute, value)
            subTree = makeSubtrees(updatedAttributes, subset, attributeValues, className, getMostCommonClass(examples, className), setScoreFunc, gainFunc)
            node.children.update({value:subTree})
        return node


def makePrunedTree(examples, attrValues,className,setScoreFunc,gainFunc,q):
    """
    Creates the classification tree for the given examples. Note that this is implemented - you
    just need to imeplement makeSubtrees.
    
    Args:
        examples (list<dictionary<str,str>>): list of examples
        attrValues (dictionary<string,list<string>>): list of possible values for attribute
        className (str): the name of the class
        classScoreFunc (func): the function to score classes (ie setEntropy or giniIndex)
        gainFunc (func): the function to score gain of attributes (ie infoGain or giniGain)
        q (float): the Chi-Squared pruning parameter
    Returns:
        Tree
        The classification tree for this set of examples
    """
    remainingAttributes=attrValues.keys()
    return Tree(makePrunedSubtrees(remainingAttributes,examples,attrValues,className,getMostCommonClass(examples,className),setScoreFunc,gainFunc,q))
    
def makePrunedSubtrees(remainingAttributes,examples,attributeValues,className,defaultLabel,setScoreFunc,gainFunc,q):
    """
    Creates a classification tree Node and all its children. This returns a Node, which is the root
    Node of the tree constructed from the passed in parameters. This should be implemented recursively,
    and handle base cases for zero examples or remainingAttributes as covered in the book.    

    Args:
        remainingAttributes (list<string>): the names of attributes still not used
        examples (list<dictionary<str,str>>): list of examples
        attrValues (dictionary<string,list<string>>): list of possible values for attribute
        className (str): the name of the class
        defaultLabel (string): the default label
        setScoreFunc (func): the function to score classes (ie classEntropy or gini)
        gainFunc (func): the function to score gain of attributes (ie entropyGain or giniGain)
        q (float): the Chi-Squared pruning parameter
    Returns:
        Node or LeafNode
        The classification tree node optimal for the remaining set of attributes.
    """
    if len(examples) == 0: # empty examples
        return LeafNode(defaultLabel)
    elif len(getClassCounts(examples, className)) == 1: # only one class
        return LeafNode(examples[0][className])
    elif len(remainingAttributes) == 0: # no more attributes available
        return LeafNode(getMostCommonClass(examples, className))
    else:       
        bestAttribute = remainingAttributes[0]
        maxGain = gainFunc(examples, bestAttribute, attributeValues[bestAttribute], className)
        for i in range(1, len(remainingAttributes)):
            newGain = gainFunc(examples, remainingAttributes[i], attributeValues[remainingAttributes[i]], className)
            if newGain > maxGain:
                maxGain = newGain
                bestAttribute = remainingAttributes[i]
        updatedAttributes = remainingAttributes[:]
        updatedAttributes.remove(bestAttribute)

        #prune
        # dict1 = getAttributeCounts(examples, bestAttribute, attributeValues[bestAttribute], className)
        # dict2 = {}
        # for key in dict1.keys():
        #     classCount = 0
        #     for item in dict1[key].keys():
        #         classCount += dict1[key][item]
        #     dict2[key] = classCount
        # classCounts = getClassCounts(examples, className)
        # deviation = 0
        # for key in dict1.keys():
        #     chi = 0
        #     for item in dict1[key].keys():
        #         pi = 1.0*dict1[key][item]
        #         piAverage = (1.0*classCounts[item] / len(examples)) * dict2[key]
        #         chi += (pi - piAverage) ** 2 / piAverage
        #     deviation += chi
        # degree = len(attributeValues[bestAttribute]) - 1
        # print "deviation: ", deviation
        # print "q: " ,q
        # print "degree of freedom: ", degree
        # if chi2.sf(deviation, degree) > q:
        #     return LeafNode(getMostCommonClass(examples, className))

        attributeCounts = getAttributeCounts(examples, bestAttribute, attributeValues[bestAttribute], className)
        print "attributes counts: ", attributeCounts
        sizeDict = {}
        for value in attributeCounts.keys():
            size = 0
            for classLabel in attributeCounts[value].keys():
                size += attributeCounts[value][classLabel]
            sizeDict[value] = size
        print "size of examples: ", len(examples)
        print "size dict: ", sizeDict


        classCount = getClassCounts(examples, className)
        print "class count: ", classCount
        deviation = 0
        for value in attributeCounts.keys():
            subClassCount = attributeCounts[value]
            for classLabel in subClassCount.keys():
                expected = 1.0 * classCount[classLabel] * sizeDict[value] / len(examples)
                actual = subClassCount[classLabel]
                deviation += (actual - expected) * (actual - expected) / expected
                print "actual = ", actual
                print "expected = ", expected
                print "---------------------------------------"

        degree = len(attributeCounts) - 1
        print "deviation: ", deviation
        print "q: " ,q
        print "degree of freedom: ", degree
        if chi2.sf(deviation, degree) > q:
            return LeafNode(getMostCommonClass(examples, className))

        node = Node(bestAttribute)
        for value in attributeValues[bestAttribute]:
            subset = getPertinentExamples(examples, bestAttribute, value)
            subTree = makeSubtrees(updatedAttributes, subset, attributeValues, className, getMostCommonClass(examples, className), setScoreFunc, gainFunc)
            node.children.update({value:subTree})
        return node
