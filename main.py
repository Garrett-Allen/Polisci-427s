import math 
import numpy
import igraph

personWeights = [] #weight list for each agent
n = 10 # number of agents
pZero = .1 # probability of zero
closeness = 100 #how close agents are in model. high values = closer agents
threshold = .5
for i in range (0,n):
  personWeights.append([0] * n) #initialize 0 lists

for i in range (0,n):
  for j in range (0,n): 
    m = numpy.random.uniform(low = 0, high = 1) 
    if(m  < pZero): #used for setting 0 values in open
      r = 0
    else: 
      r = round(numpy.random.exponential(scale = closeness) + 1,2) #random closeness value with less links of high closeness
    if(j != i):
      personWeights[i][j] = r #making weights symmetric
      personWeights[j][i] = r
      
    else:
      personWeights[i][i] = 0 #1 connection w/ self.
#for i in range (0,n):
 # print("Initial Person Weights:" + str(i) +  str(personWeights[i]))
#plotting initial graph from adjacency matrix 
#G = igraph.Graph.Weighted_Adjacency(personWeights)
#layout = G.layout("lgl")
#igraph.plot(G,layout=layout)

#initalize sList
def sigmoid(x): 
  return 1/(1+ math.exp(-x))

seed = numpy.random.randint(0, n)
sList = [0]*n 
sList[seed] = 1 #setting initial believer of misinformation
p = .1 #believability of hoax 
print("seed = " + str(seed))
print("sList:" + str(-1) + str(sList))

def updateS(weightList, sList, p):
  updateList = sList.copy() #used for pointer stuff
  for i in range(0,len(weightList)):
    influenceFromContacts = 0
    for j in range (0, len(weightList)):

      influenceFromContacts+=sList[j]*p*weightList[i][j] #what is update? how to bound it (maybe (0,1)) maybe sigmoid


    update = influenceFromContacts
    updateList[i] = round(update,2)
  return updateList


def cutTies(weightList,sList):
  numberofCutTies = 0
  for i in range(0,len(weightList)):
    for j in range(0,len(weightList)):
      difference = abs(sList[i] - sList[j])


      if(sigmoid(difference) > threshold and weightList[i][j] != 0): #what is threshold?


        weightList[i][j] = 0
        weightList[j][i] = 0
        numberofCutTies+=1

  print(numberofCutTies)

for i in range (0,4): 
  updateS(personWeights,sList,p)
  cutTies(personWeights,sList)
  sList = updateS(personWeights,sList,p)
  print("sList:" + str(i) + str(sList))
for i in range (0,n): 
  print("End PersonWeights:" + str(i) + str(personWeights[i]))

    
##to-do: output measures we want 


#problems: 
#should we have constant threshold?
#is sigmoid good? how can we bound so that these are comparable
#what function should we use for the update each time that is reasonable
#what function should we use to "cut ties"
#how do we initialize the weights in a way that make sense?
#how do we appropriately bound weights/s scores so that they are easily comparable? 