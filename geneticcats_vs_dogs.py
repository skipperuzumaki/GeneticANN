import cv2
import numpy as np

def extract(path):
  img = cv2.imread(path)
  img = cv2.resize(img, (20,20) , interpolation = cv2.INTER_AREA) 
  img1 = np.append(img[:,:,0],img[:,:,1])
  img = np.append(img1,img[:,:,2])
  img = cv2.resize(img,(1200,1))
  img = img/255.0
  return img

def sigmoid(X):
  return 1.0/(1.0+np.exp(-1*X))

def clip(X):
  if X[0] < 0:
    return -1
  else:
    return 1

def FeedForward(X,W,act = "SGM"):
  out = np.matmul(X,W)
  if act == "SGM":
    return sigmoid(out)
  elif act == "clip":
    return clip(out)
  else:
    return np.array([0])

import random

class Network:
  def __init__(self):
    self.H_layers = 4
    self.HL1 = np.random.uniform(low=-0.1, high=0.1, size=(1200,3600))
    self.HL2 = np.random.uniform(low=-0.1, high=0.1, size=(3600,1600))
    self.HL3 = np.random.uniform(low=-0.1, high=0.1, size=(1600,800))
    self.HL4 = np.random.uniform(low=-0.1, high=0.1, size=(800,1))
  def Evaluate(self,X):
    HL1 = FeedForward(X,self.HL1)
    HL2 = FeedForward(HL1,self.HL2)
    HL3 = FeedForward(HL2,self.HL3)
    evaluation = FeedForward(HL3,self.HL4,'clip')
    return evaluation
  def Mutate(self,range_L,range_H,pc):
    def draw(pc):
      k = random.randint(0,100)
      if k <= pc:
        return
    self.HL1 += np.random.uniform(low=-1.0, high=1.0, size=(1200,3600))
    self.HL2 += np.random.uniform(low=-1.0, high=1.0, size=(3600,1600))
    self.HL3 += np.random.uniform(low=-1.0, high=1.0, size=(1600,800))
    self.HL4 += np.random.uniform(low=-1.0, high=1.0, size=(800,1))
  def Breed(self,Ntwk):
    self.HL2 = Ntwk.HL2
    self.HL4 = Ntwk.HL4

import os
def Train_Generator():
  cats = os.listdir("./cats")
  ci = -1
  mc = True
  dogs = os.listdir("./dogs")
  di = -1
  md = True
  k = random.random()
  for i in range(500):
    yield [extract("./dogs/" + dogs[di]),-1]
  for i in range(500):
    yield [extract("./cats/" + cats[ci]),1]

NGen = 100
PopSize = 21
MPc = 100

def fitness(model):
  corr = 0
  tots = 0
  G = Train_Generator()
  while True:
    try:
      X = next(G)
    except:
      break
    tots += 1
    k = model.Evaluate(X[0])
    if k == X[1]:
      corr += 1
  tots = float(tots) / 100.0
  return (float(float(corr)/float(tots)))

def Fitnesses(models):
  tots = 0
  corr = np.zeros(len(models))
  G = Train_Generator()
  while True:
    try:
      X = next(G)
    except:
      break
    tots+=1
    for i in range(len(models)):
      k = models[i].Evaluate(X[0])
      if k == X[1]:
        corr[i] += 1
  corr = corr / tots
  return corr
  

Population = []
for i in range(PopSize):
  k = Network()
  Population.append(k)

def partition(arr, arr2,low,high): 
  i = ( low-1 )         # index of smaller element 
  pivot = arr[high]     # pivot 
  
  for j in range(low , high): 
  
    # If current element is smaller than the pivot 
    if   arr[j] < pivot: 
          
      # increment index of smaller element 
      i = i+1 
      arr[i],arr[j] = arr[j],arr[i]
      arr2[i],arr2[j] = arr2[j],arr2[i]
  
  arr[i+1],arr[high] = arr[high],arr[i+1]
  arr2[i+1],arr2[high] = arr2[high],arr2[i+1]
  return ( i+1 ) 
  
# The main function that implements QuickSort 
# arr[] --> Array to be sorted, 
# low  --> Starting index, 
# high  --> Ending index 
  
# Function to do Quick sort 
def quickSort(arr, arr2,low,high): 
  if low < high: 
  
    # pi is partitioning index, arr[p] is now 
    # at right place 
    pi = partition(arr, arr2,low,high) 
  
    # Separately sort elements before 
    # partition and after partition 
    quickSort(arr, arr2, low, pi-1) 
    quickSort(arr, arr2, pi+1, high) 

for i in range(NGen):
  #Fitness
  print("Test Start")
  acc = Fitnesses(Population)
  quickSort(acc,Population,0,(len(acc)-1))
  print(acc)
  #Breeding
  fit = Population
  best = Network()
  best.HL1 = fit[0].HL1
  best.HL2 = fit[0].HL2
  best.HL3 = fit[0].HL3
  best.HL4 = fit[0].HL4
  Population = []
  Population.append(best)
  print("Generation : ",i," Best : ",acc[0])
  Tfit = np.sum(acc)
  for k in range(int((PopSize - 1)/2)):
    q = random.randint(0,int(Tfit)-1)
    Cfit = 0
    z = 0
    w = fit[0]
    e = fit[1]
    while (Cfit < q):
      w = fit[z]
      Cfit += acc[z]
    q = random.randint(0,int(Tfit)-1)
    Cfit = 0
    z = 0
    while (Cfit < q):
      e = fit[z]
      Cfit += acc[z]
    w.Breed(e)
    b = Network()
    b.HL1 = w.HL1
    b.HL2 = w.HL2
    b.HL3 = w.HL3
    b.HL4 = w.HL4
    Population.append(b)
    #Mutation
    w.Mutate(-1.0,1.0,MPc)
    Population.append(w)

