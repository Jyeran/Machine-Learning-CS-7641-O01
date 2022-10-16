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

fitness = mlroseh.FlipFlop()

init_state = []

problem = mlroseh.DiscreteOpt(length = len(init_state), fitness_fn = fitness, maximize = True, max_val = 2)


lList = [5,25,50,75,100,125,150,175,200,225]
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
                                                          init_state = init_state, curve = True, restarts = 3)
        
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
                                                          pop_size = 50, mutation_prob =.1 , curve = True)
        
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
        best_state, best_fitness, fitness_curve = mlroseh.mimic(problem, max_attempts = 5, max_iters = 500,
                                                          pop_size = 50, keep_pct =.2 , curve = True)
        
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
