# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 09:53:16 2022

@author: rjohn
"""

import six
import sys
sys.modules['sklearn.externals.six'] = six
import numpy as np
import random as rand
import mlrose_hiive as mlroseh
import seaborn as sns
import matplotlib.pyplot as plt
import time

iterations = []
for i in range(0,500):
    iterations.append(i)
    
fitness = mlroseh.Queens()

problem = mlroseh.DiscreteOpt(length = 8, fitness_fn = fitness, maximize = False, max_val = 8)

# Define decay schedule
schedule = mlroseh.ExpDecay()

# Define initial state
init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])

# Solve problem using simulated annealing
best_state, best_fitness, fitness_curve = mlroseh.simulated_annealing(problem, schedule = schedule,
                                                      max_attempts = 100, max_iters = 1000,
                                                      init_state = init_state, random_state = 1, curve = True) 

print(best_fitness)
print(best_state)

fitC = []
for i in iterations:
    if i < len(fitness_curve):
        fitC.append(fitness_curve[i,0])
    else:
        fitC.append(fitness_curve[-1,0])
plt.plot(iterations,fitC,label='SA',drawstyle="steps-post")
plt.legend()


# Solve problem using random hill
best_state, best_fitness, fitness_curve = mlroseh.random_hill_climb(problem, max_iters = 1000,
                                                      init_state = init_state, random_state = 1, curve = True)

print(best_fitness)
print(best_state)

fitC = []
for i in iterations:
    if i < len(fitness_curve):
        fitC.append(fitness_curve[i,0])
    else:
        fitC.append(fitness_curve[-1,0])
plt.plot(iterations,fitC,label='Fitness',drawstyle="steps-post")
plt.legend()

# Solve problem using genetic algorithm
best_state, best_fitness, fitness_curve = mlroseh.genetic_alg(problem, max_attempts = 100, max_iters = 1000,
                                                      pop_size = 200, mutation_prob =.1 , random_state = 1, curve = True)

print(best_fitness)
print(best_state)

fitC = []
for i in iterations:
    if i < len(fitness_curve):
        fitC.append(fitness_curve[i,0])
    else:
        fitC.append(fitness_curve[-1,0])
plt.plot(iterations,fitC,label='Fitness',drawstyle="steps-post")
plt.legend()

# Solve problem using mimic
best_state, best_fitness, fitness_curve = mlroseh.mimic(problem, max_attempts = 100, max_iters = 1000,
                                                      pop_size = 200, keep_pct =.3 , random_state = 1, curve = True)

print(best_fitness)
print(best_state)

fitC = []
for i in iterations:
    if i < len(fitness_curve):
        fitC.append(fitness_curve[i,0])
    else:
        fitC.append(fitness_curve[-1,0])
plt.plot(iterations,fitC,label='Fitness',drawstyle="steps-post")
plt.legend()


############K Colors Problem#################
rand.seed(42)
edges = [(0, 1),
         (1, 3),
         (2, 0),
         (3, 4),
         (4, 5),
         (5, 7),
         (6, 7),
         (6, 8),
         (8, 11),
         (9, 10),
         (10,11),
         (11,12),
         (12,14),
         (13,15),
         (14,15),
         (15,16),
         (16,17),
         (17,18),
         (18,19),
         (19,20),
         (20,21),
         (21,22),
         (22,23),
         (23,24)]

#edges = []
nodes = []
for i in range(0,50):
    nodes.append(i)
    
    for z in range(0,3):
        connection = (i,rand.randint(0, 50))
        
        if connection[0] != connection[1]:
           if (connection not in edges) and ((connection[1],connection[0]) not in edges):
                edges.append(connection)
            

init_state = []
k = 2
for i in range(0,51):
    init_state.append(rand.randint(0,k-1))

fitness = mlroseh.MaxKColor(edges)

fitness.evaluate(init_state)

problem = mlroseh.DiscreteOpt(length = len(init_state), fitness_fn = fitness, maximize = False, max_val = k)

# Define decay schedule
schedule = mlroseh.ExpDecay()

# Solve problem using simulated annealing
SAfitness = [0] * 500
count = 0
for i in range(0,10):
    best_state, best_fitness, fitness_curve = mlroseh.simulated_annealing(problem, schedule = schedule,
                                                          max_attempts = 500, max_iters = 500,
                                                          init_state = init_state, curve = True) 
    
    count+=1
    
    fitC = []
    for i in iterations:
        if i < len(fitness_curve):
            fitC.append(fitness_curve[i,0])
        else:
            fitC.append(fitness_curve[-1,0])
    
    for i in range(0,500):
        SAfitness[i]+=fitC[i]
    
    
print(best_fitness)
print(best_state)

for i in range(0,500):
    SAfitness[i] = -(SAfitness[i]/count)

plt.plot(iterations,SAfitness,label='SA',drawstyle="steps-post")
plt.legend()


# Solve problem using random hill
RHCfitness = [0] * 500
count = 0
for i in range(0,10):
    best_state, best_fitness, fitness_curve = mlroseh.random_hill_climb(problem,max_attempts=500, max_iters = 500,
                                                      init_state = init_state, curve = True,restarts=5)
    
    count+=1
    
    fitC = []
    for i in iterations:
        if i < len(fitness_curve):
            fitC.append(fitness_curve[i,0])
        else:
            fitC.append(fitness_curve[-1,0])
    
    for i in range(0,500):
        RHCfitness[i]+=fitC[i]
    
    
print(best_fitness)
print(best_state)

for i in range(0,500):
    RHCfitness[i] = -(RHCfitness[i]/count)

plt.plot(iterations,RHCfitness,label='RHC',drawstyle="steps-post")
plt.legend()

# Solve problem using genetic algorithm
GAfitness = [0] * 500
count = 0
for i in range(0,10):
    best_state, best_fitness, fitness_curve = mlroseh.genetic_alg(problem, max_attempts = 500, max_iters = 500,
                                                      pop_size = 100, mutation_prob =.1 , curve = True)
    
    count+=1
    
    fitC = []
    for i in iterations:
        if i < len(fitness_curve):
            fitC.append(fitness_curve[i,0])
        else:
            fitC.append(fitness_curve[-1,0])
    
    for i in range(0,500):
        GAfitness[i]+=fitC[i]
    
    
print(best_fitness)
print(best_state)

for i in range(0,500):
    GAfitness[i] = -(GAfitness[i]/count)

plt.plot(iterations,GAfitness,label='GA',drawstyle="steps-post")
plt.legend()

# Solve problem using mimic
best_state, best_fitness, fitness_curve = mlroseh.mimic(problem, max_attempts = 100, max_iters = 500,
                                                      pop_size = 100, keep_pct =.2 , random_state = 1, curve = True)

print(best_fitness)
print(best_state)

fitC = []
for i in iterations:
    if i < len(fitness_curve):
        fitC.append(fitness_curve[i,0])
    else:
        fitC.append(fitness_curve[-1,0])
                                  
plt.plot(iterations,fitC,label='MIMIC',drawstyle="steps-post")
plt.legend()
plt.plot(iterations,SAfitness,label='SA',drawstyle="steps-post")
plt.legend()
plt.plot(iterations,RHCfitness,label='RHC',drawstyle="steps-post")
plt.legend()
plt.plot(iterations,GAfitness,label='GA',drawstyle="steps-post")
plt.legend()




########################################
######################################## GA
############Four Peaks##################
fitness = mlroseh.FourPeaks()

init_state = []
l = 100
for i in range(0,l):
    init_state.append(rand.randint(0,1))

problem = mlroseh.DiscreteOpt(length = len(init_state), fitness_fn = fitness, maximize = True, max_val = 2)

fitness.evaluate(init_state)

# Define decay schedule
schedule = mlroseh.ExpDecay(init_temp=25)


# Solve problem using simulated annealing
SAfitness = [0] * 500
SAtime = []
count = 0
for i in range(0,10):
    start = time.time()
    best_state, best_fitness, fitness_curve = mlroseh.simulated_annealing(problem, schedule = schedule,
                                                          max_attempts = 500, max_iters = 500,
                                                          init_state = init_state, curve = True) 
    
    stop = time.time()
    count+=1
    SAtime.append(stop-start)
    
    fitC = []
    for i in iterations:
        if i < len(fitness_curve):
            fitC.append(fitness_curve[i,0])
        else:
            fitC.append(fitness_curve[-1,0])
    
    for i in range(0,500):
        SAfitness[i]+=fitC[i]
    
    
print(best_fitness)
print(best_state)

for i in range(0,500):
    SAfitness[i] = (SAfitness[i]/count)

#plt.plot(iterations,SAfitness,label='SA',drawstyle="steps-post")
#plt.legend()

# Solve problem using random hill
RHCfitness = [0] * 500
RHCtime = []
count = 0
for i in range(0,10):
    start = time.time()
    best_state, best_fitness, fitness_curve = mlroseh.random_hill_climb(problem,max_attempts=500, max_iters = 500,
                                                      init_state = init_state, curve = True)
    
    stop = time.time()
    RHCtime.append(stop-start)
    count+=1
    
    fitC = []
    for i in iterations:
        if i < len(fitness_curve):
            fitC.append(fitness_curve[i,0])
        else:
            fitC.append(fitness_curve[-1,0])
    
    for i in range(0,500):
        RHCfitness[i]+=fitC[i]
    
    
print(best_fitness)
print(best_state)

for i in range(0,500):
    RHCfitness[i] = (RHCfitness[i]/count)

#plt.plot(iterations,RHCfitness,label='RHC',drawstyle="steps-post")
#plt.legend()

# Solve problem using genetic algorithm
GAfitness = [0] * 500
GAtime = []
count = 0
for i in range(0,10):
    start = time.time()
    best_state, best_fitness, fitness_curve = mlroseh.genetic_alg(problem, max_attempts = 500, max_iters = 500,
                                                      pop_size = 100, mutation_prob =.1 , curve = True)
    
    stop = time.time()
    count+=1
    GAtime.append(stop - start)
    
    fitC = []
    for i in iterations:
        if i < len(fitness_curve):
            fitC.append(fitness_curve[i,0])
        else:
            fitC.append(fitness_curve[-1,0])
    
    for i in range(0,500):
        GAfitness[i]+=fitC[i]
    
    
print(best_fitness)
print(best_state)

for i in range(0,500):
    GAfitness[i] = (GAfitness[i]/count)


# Solve problem using mimic
MIMfitness = [0] * 500
MIMtime = []
count = 0
for i in range(0,3):
    print('MIM iter: ', i)
    start = time.time()
    best_state, best_fitness, fitness_curve = mlroseh.mimic(problem, max_attempts = 7, max_iters = 500,
                                                      pop_size = 100, keep_pct =.2 , curve = True)
    
    stop = time.time()
    count+=1
    MIMtime.append(stop - start)
    
    fitC = []
    for i in iterations:
        if i < len(fitness_curve):
            fitC.append(fitness_curve[i,0])
        else:
            fitC.append(fitness_curve[-1,0])
    
    for i in range(0,500):
        MIMfitness[i]+=fitC[i]
    
    
print(best_fitness)
print(best_state)

for i in range(0,500):
    MIMfitness[i] = (MIMfitness[i]/count)

plt.close()
plt.title('Four Peaks Optimization: Length 100')
plt.xlabel('Iterations')
plt.ylabel('Fitness')       
plt.plot(iterations,SAfitness,label='SA',drawstyle="steps-post")
plt.plot(iterations,RHCfitness,label='RHC',drawstyle="steps-post")
plt.plot(iterations,GAfitness,label='GA',drawstyle="steps-post")
plt.plot(iterations,MIMfitness,label='MIMIC',drawstyle="steps-post")
plt.legend()
plt.show()
plt.close()

timing = [round(np.mean(SAtime),3), round(np.mean(RHCtime),3), round(np.mean(GAtime),3), round(np.mean(MIMtime),3)]
labels = ['SA','RHC','GA','MIM']
timeDict = {}
timeDict['x'] = labels
timeDict['y'] = timing
ax = sns.barplot(x='x',y='y',data=timeDict)

for i in ax.containers:
    ax.bar_label(i,fontsize=12)
    
ax.set(title='Four Peaks Wall Clock Time (seconds): Length 100', 
       xlabel='Algorithm', 
       ylabel='Time in Seconds')



############FlipFlop Problem################# SA
fitness = mlroseh.FlipFlop()

init_state = []
l = 100
for i in range(0,l):
    init_state.append(rand.randint(0,1))

problem = mlroseh.DiscreteOpt(length = len(init_state), fitness_fn = fitness, maximize = True, max_val = 2)

fitness.evaluate(init_state)

# Define decay schedule
schedule = mlroseh.ExpDecay()

# Solve problem using simulated annealing
SAfitness = [0] * 500
SAtime = []
count = 0
for i in range(0,10):
    start = time.time()
    best_state, best_fitness, fitness_curve = mlroseh.simulated_annealing(problem, schedule = schedule,
                                                          max_attempts = 500, max_iters = 500,
                                                          init_state = init_state, curve = True) 
    
    stop = time.time()
    count+=1
    SAtime.append(stop-start)
    
    fitC = []
    for i in iterations:
        if i < len(fitness_curve):
            fitC.append(fitness_curve[i,0])
        else:
            fitC.append(fitness_curve[-1,0])
    
    for i in range(0,500):
        SAfitness[i]+=fitC[i]
    
    
print(best_fitness)
print(best_state)

for i in range(0,500):
    SAfitness[i] = (SAfitness[i]/count)

#plt.plot(iterations,SAfitness,label='SA',drawstyle="steps-post")
#plt.legend()

# Solve problem using random hill
RHCfitness = [0] * 500
RHCtime = []
count = 0
for i in range(0,10):
    start = time.time()
    best_state, best_fitness, fitness_curve = mlroseh.random_hill_climb(problem,max_attempts=500, max_iters = 500,
                                                      init_state = init_state, curve = True, restarts=3)
    
    stop = time.time()
    RHCtime.append(stop-start)
    count+=1
    
    fitC = []
    for i in iterations:
        if i < len(fitness_curve):
            fitC.append(fitness_curve[i,0])
        else:
            fitC.append(fitness_curve[-1,0])
    
    for i in range(0,500):
        RHCfitness[i]+=fitC[i]
    
    
print(best_fitness)
print(best_state)

for i in range(0,500):
    RHCfitness[i] = (RHCfitness[i]/count)

#plt.plot(iterations,RHCfitness,label='RHC',drawstyle="steps-post")
#plt.legend()

# Solve problem using genetic algorithm
GAfitness = [0] * 500
GAtime = []
count = 0
for i in range(0,10):
    start = time.time()
    best_state, best_fitness, fitness_curve = mlroseh.genetic_alg(problem, max_attempts = 500, max_iters = 500,
                                                      pop_size = 50, mutation_prob =.1 , curve = True)
    
    stop = time.time()
    count+=1
    GAtime.append(stop - start)
    
    fitC = []
    for i in iterations:
        if i < len(fitness_curve):
            fitC.append(fitness_curve[i,0])
        else:
            fitC.append(fitness_curve[-1,0])
    
    for i in range(0,500):
        GAfitness[i]+=fitC[i]
    
    
print(best_fitness)
print(best_state)

for i in range(0,500):
    GAfitness[i] = (GAfitness[i]/count)

plt.plot(iterations,SAfitness,label='SA',drawstyle="steps-post")
plt.legend()
plt.plot(iterations,RHCfitness,label='RHC',drawstyle="steps-post")
plt.legend()
plt.plot(iterations,GAfitness,label='GA',drawstyle="steps-post")
plt.legend()

# Solve problem using mimic
MIMfitness = [0] * 500
MIMtime = []
count = 0
for i in range(0,5):
    print("MIMIC iter: ", i)
    start = time.time()
    best_state, best_fitness, fitness_curve = mlroseh.mimic(problem, max_attempts = 15, max_iters = 500,
                                                      pop_size = 100, keep_pct =.2 , curve = True)
    
    stop = time.time()
    count+=1
    MIMtime.append(stop - start)
    
    fitC = []
    for i in iterations:
        if i < len(fitness_curve):
            fitC.append(fitness_curve[i,0])
        else:
            fitC.append(fitness_curve[-1,0])
    
    for i in range(0,500):
        MIMfitness[i]+=fitC[i]
    
    
print(best_fitness)
print(best_state)

for i in range(0,500):
    MIMfitness[i] = (MIMfitness[i]/count)
        

plt.close()
plt.title('FlipFlop Optimization: length 100')
plt.xlabel('Iterations')
plt.ylabel('Fitness')       
plt.plot(iterations,SAfitness,label='SA',drawstyle="steps-post")
plt.plot(iterations,RHCfitness,label='RHC',drawstyle="steps-post")
plt.plot(iterations,GAfitness,label='GA',drawstyle="steps-post")
plt.plot(iterations,MIMfitness,label='MIMIC',drawstyle="steps-post")
plt.legend()
plt.show()
plt.close()

timing = [round(np.mean(SAtime),3), round(np.mean(RHCtime),3), round(np.mean(GAtime),3), round(np.mean(MIMtime),3)]
labels = ['SA','RHC','GA','MIM']
timeDict = {}
timeDict['x'] = labels
timeDict['y'] = timing
ax = sns.barplot(x='x',y='y',data=timeDict)

for i in ax.containers:
    ax.bar_label(i,fontsize=12)
    
ax.set(title='FlipFlop Wall Clock Time (seconds): Length 100', 
       xlabel='Algorithm', 
       ylabel='Time in Seconds')




############Knapsack Problem############### MIMIC
weights = [10, 5, 2, 8, 15]
values = [1, 2, 3, 4, 5]
max_weight_pct = 0.6
fitness = mlroseh.Knapsack(weights, values, max_weight_pct)

max_weight_pct = 0.6
fitness = mlroseh.Knapsack(weights, values, max_weight_pct)
init_state = np.array([1, 0, 2, 1, 0])

problem = mlroseh.DiscreteOpt(length = len(values), fitness_fn = fitness, maximize = True, max_val=len(values))

# Define decay schedule
schedule = mlroseh.ExpDecay()

# Solve problem using simulated annealing
SAfitness = [0] * 500
SAtime = []
count = 0
for i in range(0,10):
    start = time.time()
    best_state, best_fitness, fitness_curve = mlroseh.simulated_annealing(problem, schedule = schedule,
                                                          max_attempts = 500, max_iters = 500,
                                                          init_state = init_state, curve = True) 
    
    stop = time.time()
    count+=1
    SAtime.append(stop-start)
    
    fitC = []
    for i in iterations:
        if i < len(fitness_curve):
            fitC.append(fitness_curve[i,0])
        else:
            fitC.append(fitness_curve[-1,0])
    
    for i in range(0,500):
        SAfitness[i]+=fitC[i]
    
    
print(best_fitness)
print(best_state)

for i in range(0,500):
    SAfitness[i] = (SAfitness[i]/count)

#plt.plot(iterations,SAfitness,label='SA',drawstyle="steps-post")
#plt.legend()

# Solve problem using random hill
RHCfitness = [0] * 500
RHCtime = []
count = 0
for i in range(0,10):
    start = time.time()
    best_state, best_fitness, fitness_curve = mlroseh.random_hill_climb(problem,max_attempts=500, max_iters = 500,
                                                      init_state = init_state, curve = True)
    
    stop = time.time()
    RHCtime.append(stop-start)
    count+=1
    
    fitC = []
    for i in iterations:
        if i < len(fitness_curve):
            fitC.append(fitness_curve[i,0])
        else:
            fitC.append(fitness_curve[-1,0])
    
    for i in range(0,500):
        RHCfitness[i]+=fitC[i]
    
    
print(best_fitness)
print(best_state)

for i in range(0,500):
    RHCfitness[i] = (RHCfitness[i]/count)

#plt.plot(iterations,RHCfitness,label='RHC',drawstyle="steps-post")
#plt.legend()

# Solve problem using genetic algorithm
GAfitness = [0] * 500
GAtime = []
count = 0
for i in range(0,10):
    start = time.time()
    best_state, best_fitness, fitness_curve = mlroseh.genetic_alg(problem, max_attempts = 500, max_iters = 500,
                                                      pop_size = 100, mutation_prob =.1 , curve = True)
    
    stop = time.time()
    count+=1
    GAtime.append(stop - start)
    
    fitC = []
    for i in iterations:
        if i < len(fitness_curve):
            fitC.append(fitness_curve[i,0])
        else:
            fitC.append(fitness_curve[-1,0])
    
    for i in range(0,500):
        GAfitness[i]+=fitC[i]
    
    
print(best_fitness)
print(best_state)

for i in range(0,500):
    GAfitness[i] = (GAfitness[i]/count)


# Solve problem using mimic
MIMfitness = [0] * 500
MIMtime = []
count = 0
for i in range(0,5):
    start = time.time()
    best_state, best_fitness, fitness_curve = mlroseh.mimic(problem, max_attempts = 25, max_iters = 500,
                                                      pop_size = 200, keep_pct =.3 , curve = True)
    
    stop = time.time()
    count+=1
    MIMtime.append(stop - start)
    
    fitC = []
    for i in iterations:
        if i < len(fitness_curve):
            fitC.append(fitness_curve[i,0])
        else:
            fitC.append(fitness_curve[-1,0])
    
    for i in range(0,500):
        MIMfitness[i]+=fitC[i]
    
    
print(best_fitness)
print(best_state)

for i in range(0,500):
    MIMfitness[i] = (MIMfitness[i]/count)
        

plt.close()
plt.title('Knapsack Optimization: 5 Items')
plt.xlabel('Iterations')
plt.ylabel('Fitness')       
plt.plot(iterations,SAfitness,label='SA',drawstyle="steps-post")
plt.plot(iterations,RHCfitness,label='RHC',drawstyle="steps-post")
plt.plot(iterations,GAfitness,label='GA',drawstyle="steps-post")
plt.plot(iterations,MIMfitness,label='MIMIC',drawstyle="steps-post")
plt.legend()
plt.show()
plt.close()

timing = [round(np.mean(SAtime),3), round(np.mean(RHCtime),3), round(np.mean(GAtime),3), round(np.mean(MIMtime),3)]
labels = ['SA','RHC','GA','MIM']
timeDict = {}
timeDict['x'] = labels
timeDict['y'] = timing
ax = sns.barplot(x='x',y='y',data=timeDict)

for i in ax.containers:
    ax.bar_label(i,fontsize=12)
    
ax.set(title='Knapsack Wall Clock Time (seconds): 5 Items', 
       xlabel='Algorithm', 
       ylabel='Time in Seconds')
