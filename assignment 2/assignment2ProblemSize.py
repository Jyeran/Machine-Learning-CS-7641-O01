# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 21:59:00 2022

@author: rjohn
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 21:59:00 2022

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
    
fitness = mlroseh.FourPeaks()

init_state = []

lList = [5,25,50,75,100,125,150,175,200,250]
SAfit = []
RHCfit = []
GAfit = []
MIMfit = []

SAclock = []
RHCclock = []
GAclock = []
MIMclock = []

for l in lList:
    print(l)
    for i in range(0,l):
        init_state.append(rand.randint(0,1))
    
    problem = mlroseh.DiscreteOpt(length = len(init_state), fitness_fn = fitness, maximize = True, max_val = 2)
    
    fitness.evaluate(init_state)
    
    # Define decay schedule
    schedule = mlroseh.ExpDecay()
    
    
    # Solve problem using simulated annealing
    SAfitness = []
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
        
        SAfitness.append(best_fitness)
    
    #plt.plot(iterations,SAfitness,label='SA',drawstyle="steps-post")
    #plt.legend()
    
    # Solve problem using random hill
    RHCfitness = []
    RHCtime = []
    count = 0
    for i in range(0,10):
        start = time.time()
        best_state, best_fitness, fitness_curve = mlroseh.random_hill_climb(problem,max_attempts=500, max_iters = 500,
                                                          init_state = init_state, curve = True)
        
        stop = time.time()
        RHCtime.append(stop-start)
        count+=1
        
        RHCfitness.append(best_state)
    
    #plt.plot(iterations,RHCfitness,label='RHC',drawstyle="steps-post")
    #plt.legend()
    
    # Solve problem using genetic algorithm
    GAfitness = []
    GAtime = []
    count = 0
    for i in range(0,2):
        start = time.time()
        best_state, best_fitness, fitness_curve = mlroseh.genetic_alg(problem, max_attempts = 500, max_iters = 500,
                                                          pop_size = 100, mutation_prob =.1 , curve = True)
        
        stop = time.time()
        count+=1
        GAtime.append(stop - start)
        
        GAfitness.append(best_fitness)
    
    
    # Solve problem using mimic
    MIMfitness = []
    MIMtime = []
    count = 0
    for i in range(0,1):
        print('MIM iter: ', i)
        start = time.time()
        best_state, best_fitness, fitness_curve = mlroseh.mimic(problem, max_attempts = 15, max_iters = 500,
                                                          pop_size = 100, keep_pct =.2 , curve = True)
        
        stop = time.time()
        count+=1
        MIMtime.append(stop - start)
        
        MIMfitness.append(best_fitness)
        
    SAfit.append(np.mean(SAfitness))
    RHCfit.append(np.mean(RHCfitness))
    GAfit.append(np.mean(GAfitness))
    MIMfit.append(np.mean(MIMfitness))
    
    SAclock.append(np.mean(SAtime))
    RHCclock.append(np.mean(RHCtime))
    GAclock.append(np.mean(GAtime))
    MIMclock.append(np.mean(MIMtime))

        
plt.close()
plt.title('Four Peaks Optimization as Problem Grows')
plt.xlabel('Problem Size')
plt.ylabel('Fitness')       
plt.plot(lList,SAfit,label='SA')
plt.plot(lList,RHCfit,label='RHC')
plt.plot(lList,GAfit,label='GA')
plt.plot(lList,MIMfit,label='MIMIC')
plt.legend()
plt.show()
plt.close()

plt.close()
plt.title('Four Peaks Clock Time as Problem Grows')
plt.xlabel('Problem Size: State Space')
plt.ylabel('Time in Seconds')       
plt.plot(lList,SAclock,label='SA')
plt.plot(lList,RHCclock,label='RHC')
plt.plot(lList,GAclock,label='GA')
plt.plot(lList,MIMclock,label='MIMIC')
plt.legend()
plt.show()
plt.close()

plt.close()
plt.title('Four Peaks Clock Time as Problem Grows (No MIMIC)')
plt.xlabel('Problem Size: State Space')
plt.ylabel('Time in Seconds')       
plt.plot(lList,SAclock,label='SA')
plt.plot(lList,RHCclock,label='RHC')
plt.plot(lList,GAclock,label='GA')
plt.legend()
plt.show()
plt.close()


fitness = mlroseh.FlipFlop()

init_state = []

problem = mlroseh.DiscreteOpt(length = len(init_state), fitness_fn = fitness, maximize = True, max_val = 2)


lList = [5,25,50,75,100,125,150,175,200,250]
SAfit = []
RHCfit = []
GAfit = []
MIMfit = []

SAclock = []
RHCclock = []
GAclock = []
MIMclock = []

for l in lList:
    print(l)
    for i in range(0,l):
        init_state.append(rand.randint(0,1))
    
    problem = mlroseh.DiscreteOpt(length = len(init_state), fitness_fn = fitness, maximize = True, max_val = 2)
    
    fitness.evaluate(init_state)
    
    # Define decay schedule
    schedule = mlroseh.ExpDecay()
    
    
    # Solve problem using simulated annealing
    SAfitness = []
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
        
        SAfitness.append(best_fitness)
    
    #plt.plot(iterations,SAfitness,label='SA',drawstyle="steps-post")
    #plt.legend()
    
    # Solve problem using random hill
    RHCfitness = []
    RHCtime = []
    count = 0
    for i in range(0,10):
        start = time.time()
        best_state, best_fitness, fitness_curve = mlroseh.random_hill_climb(problem,max_attempts=500, max_iters = 500,
                                                          init_state = init_state, curve = True)
        
        stop = time.time()
        RHCtime.append(stop-start)
        count+=1
        
        RHCfitness.append(best_state)
    
    #plt.plot(iterations,RHCfitness,label='RHC',drawstyle="steps-post")
    #plt.legend()
    
    # Solve problem using genetic algorithm
    GAfitness = []
    GAtime = []
    count = 0
    for i in range(0,2):
        start = time.time()
        best_state, best_fitness, fitness_curve = mlroseh.genetic_alg(problem, max_attempts = 500, max_iters = 500,
                                                          pop_size = 100, mutation_prob =.1 , curve = True)
        
        stop = time.time()
        count+=1
        GAtime.append(stop - start)
        
        GAfitness.append(best_fitness)
    
    
    # Solve problem using mimic
    MIMfitness = []
    MIMtime = []
    count = 0
    for i in range(0,1):
        print('MIM iter: ', i)
        start = time.time()
        best_state, best_fitness, fitness_curve = mlroseh.mimic(problem, max_attempts = 15, max_iters = 500,
                                                          pop_size = 100, keep_pct =.2 , curve = True)
        
        stop = time.time()
        count+=1
        MIMtime.append(stop - start)
        
        MIMfitness.append(best_fitness)
        
    SAfit.append(np.mean(SAfitness))
    RHCfit.append(np.mean(RHCfitness))
    GAfit.append(np.mean(GAfitness))
    MIMfit.append(np.mean(MIMfitness))
    
    SAclock.append(np.mean(SAtime))
    RHCclock.append(np.mean(RHCtime))
    GAclock.append(np.mean(GAtime))
    MIMclock.append(np.mean(MIMtime))

        
plt.close()
plt.title('FlipFlop Optimization as Problem Grows')
plt.xlabel('Problem Size: String Length')
plt.ylabel('Fitness')       
plt.plot(lList,SAfit,label='SA')
plt.plot(lList,RHCfit,label='RHC')
plt.plot(lList,GAfit,label='GA')
plt.plot(lList,MIMfit,label='MIMIC')
plt.legend()
plt.show()
plt.close()

plt.close()
plt.title('FlipFlop Clock Time as Problem Grows')
plt.xlabel('Problem Size: String Length')
plt.ylabel('Time in Seconds')       
plt.plot(lList,SAclock,label='SA')
plt.plot(lList,RHCclock,label='RHC')
plt.plot(lList,GAclock,label='GA')
plt.plot(lList,MIMclock,label='MIMIC')
plt.legend()
plt.show()
plt.close()

plt.close()
plt.title('FlipFlop Clock Time as Problem Grows (No MIMIC)')
plt.xlabel('Problem Size: String Length')
plt.ylabel('Time in Seconds')       
plt.plot(lList,SAclock,label='SA')
plt.plot(lList,RHCclock,label='RHC')
plt.plot(lList,GAclock,label='GA')
plt.legend()
plt.show()
plt.close()






fitness = mlroseh.FlipFlop()

init_state = []

problem = mlroseh.DiscreteOpt(length = len(init_state), fitness_fn = fitness, maximize = True, max_val = 2)


weights = [
    [10, 5, 2, 8, 15],
    [10, 5, 2, 8, 15, 4, 3, 14, 12, 1],
    [10, 5, 2, 8, 15, 4, 3, 14, 12, 1, 5, 6, 13, 11, 7],
    [10, 5, 2, 8, 15, 4, 3, 14, 12, 1, 5, 6, 13, 11, 7, 9, 3, 10, 6, 9],
    [10, 5, 2, 8, 15, 4, 3, 14, 12, 1, 5, 6, 13, 11, 7, 9, 3, 10, 6, 9, 14, 15, 4, 1, 12]
    ]

values = [
    [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    ]

states = [
    [1, 0, 2, 1, 0],
    [1, 0, 2, 1, 0, 7, 0, 0, 0, 2],
    [1, 0, 2, 1, 0, 7, 0, 0, 0, 2, 6, 6, 0, 0, 0],
    [1, 0, 2, 1, 0, 7, 0, 0, 0, 2, 6, 6, 0, 0, 0, 0, 0, 0, 12, 13],
    [1, 0, 2, 1, 0, 7, 0, 0, 0, 2, 6, 6, 0, 0, 0, 0, 0, 0, 12, 13, 24, 0, 24, 0, 24]
    ]
SAfit = []
RHCfit = []
GAfit = []
MIMfit = []

SAclock = []
RHCclock = []
GAclock = []
MIMclock = []

for i in range(0,len(values)):
    print(i)
    
    w = weights[i]
    v = values[i]
    max_weight_pct = 0.6
    fitness = mlroseh.Knapsack(w, v, max_weight_pct)

    max_weight_pct = 0.6
    fitness = mlroseh.Knapsack(w, v, max_weight_pct)
    init_state = np.array(states[i])

    problem = mlroseh.DiscreteOpt(length = len(v), fitness_fn = fitness, maximize = True, max_val=len(v))

    
    fitness.evaluate(init_state)
    
    # Define decay schedule
    schedule = mlroseh.ExpDecay()
    
    
    # Solve problem using simulated annealing
    SAfitness = []
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
        
        SAfitness.append(best_fitness)
    
    #plt.plot(iterations,SAfitness,label='SA',drawstyle="steps-post")
    #plt.legend()
    
    # Solve problem using random hill
    RHCfitness = []
    RHCtime = []
    count = 0
    for i in range(0,10):
        start = time.time()
        best_state, best_fitness, fitness_curve = mlroseh.random_hill_climb(problem,max_attempts=500, max_iters = 500,
                                                          init_state = init_state, curve = True)
        
        stop = time.time()
        RHCtime.append(stop-start)
        count+=1
        
        RHCfitness.append(best_state)
    
    #plt.plot(iterations,RHCfitness,label='RHC',drawstyle="steps-post")
    #plt.legend()
    
    # Solve problem using genetic algorithm
    GAfitness = []
    GAtime = []
    count = 0
    for i in range(0,2):
        start = time.time()
        best_state, best_fitness, fitness_curve = mlroseh.genetic_alg(problem, max_attempts = 500, max_iters = 500,
                                                          pop_size = 100, mutation_prob =.1 , curve = True)
        
        stop = time.time()
        count+=1
        GAtime.append(stop - start)
        
        GAfitness.append(best_fitness)
    
    
    # Solve problem using mimic
    MIMfitness = []
    MIMtime = []
    count = 0
    for i in range(0,1):
        print('MIM iter: ', i)
        start = time.time()
        best_state, best_fitness, fitness_curve = mlroseh.mimic(problem, max_attempts = 15, max_iters = 500,
                                                          pop_size = 150, keep_pct =.2 , curve = True)
        
        stop = time.time()
        count+=1
        MIMtime.append(stop - start)
        
        MIMfitness.append(best_fitness)
        
    SAfit.append(np.mean(SAfitness))
    RHCfit.append(np.mean(RHCfitness))
    GAfit.append(np.mean(GAfitness))
    MIMfit.append(np.mean(MIMfitness))
    
    SAclock.append(np.mean(SAtime))
    RHCclock.append(np.mean(RHCtime))
    GAclock.append(np.mean(GAtime))
    MIMclock.append(np.mean(MIMtime))

        
knapItems = [5,10,15,20,25]       
plt.close()
plt.title('Knapsack Optimization as Problem Grows')
plt.xlabel('Problem Size: Items for Knapsack')
plt.ylabel('Fitness')       
plt.plot(knapItems,SAfit,label='SA')
plt.plot(knapItems,RHCfit,label='RHC')
plt.plot(knapItems,GAfit,label='GA')
plt.plot(knapItems,MIMfit,label='MIMIC')
plt.legend()
plt.show()
plt.close()

plt.close()
plt.title('Knapsack Clock Time as Problem Grows')
plt.xlabel('Problem Size: Items for Knapsack')
plt.ylabel('Time in Seconds')       
plt.plot(knapItems,SAclock,label='SA')
plt.plot(knapItems,RHCclock,label='RHC')
plt.plot(knapItems,GAclock,label='GA')
plt.plot(knapItems,MIMclock,label='MIMIC')
plt.legend()
plt.show()
plt.close()

plt.close()
plt.title('Knapsack Clock Time as Problem Grows (No MIMIC)')
plt.xlabel('Problem Size: Items for Knapsack')
plt.ylabel('Time in Seconds')       
plt.plot(knapItems,SAclock,label='SA')
plt.plot(knapItems,RHCclock,label='RHC')
plt.plot(knapItems,GAclock,label='GA')
plt.legend()
plt.show()
plt.close()