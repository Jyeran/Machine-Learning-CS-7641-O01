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
l = 100
for i in range(0,l):
    init_state.append(rand.randint(0,1))

problem = mlroseh.DiscreteOpt(length = len(init_state), fitness_fn = fitness, maximize = True, max_val = 2)

# Solve problem using genetic algorithm
GAfit= [] 
GAclock= []
count = 0

pops = [5,25,50,100,125,150,175,200]

for p in pops:
    GAfitness = [] 
    GAtime = []
    for i in range(0,10):
        print(p, i)
        start = time.time()
        best_state, best_fitness, fitness_curve = mlroseh.genetic_alg(problem, max_attempts = 500, max_iters = 500,
                                                          pop_size = p, mutation_prob =.1 , curve = True)
        
        stop = time.time()
        count+=1
        GAtime.append(stop - start)
        GAfitness.append(best_fitness)
        
    GAfit.append(np.mean(GAfitness))
    GAclock.append(np.mean(GAtime))
           
plt.close()
plt.title('Four Peaks GA Tuning Fitness')
plt.xlabel('Population Size')
plt.ylabel('Fitness')
plt.plot(pops[:-1],GAfit[:-1],label='GA')
plt.legend()
plt.show()
plt.close()

plt.close()
plt.title('Four Peaks GA Tuning Wall Clock')
plt.xlabel('Population Size')
plt.ylabel('Time in Seconds')       
plt.plot(pops[:-1],GAclock[:-1],label='GA')
plt.legend()
plt.show()
plt.close()


GAfit= [] 
GAclock= []
count = 0

# Solve problem using simulated annealing
SAfitness = [0] * 500
count = 0
temp = [.1,1,5,10,25]
for t in temp:
    schedule = mlroseh.ExpDecay(init_temp=t)
    for i in range(0,10):
        start = time.time()
        best_state, best_fitness, fitness_curve = mlroseh.simulated_annealing(problem, schedule = schedule,
                                                              max_attempts = 500, max_iters = 10000,
                                                              init_state = init_state, curve = True) 
        
        stop = time.time()
        count+=1
        GAtime.append(stop - start)
        GAfitness.append(best_fitness)
        
    GAfit.append(np.mean(GAfitness))
    GAclock.append(np.mean(GAtime))



plt.plot(iterations,SAfitness,label='SA',drawstyle="steps-post")
plt.legend()

plt.close()
plt.title('Four Peaks SA Tuning Fitness')
plt.xlabel('Temperature')
plt.ylabel('Fitness')
plt.plot(temp,GAfit,label='SA')
plt.legend()
plt.show()
plt.close()

plt.close()
plt.title('Four Peaks GA Tuning Wall Clock')
plt.xlabel('Population Size')
plt.ylabel('Time in Seconds')       
plt.plot(temp[],GAclock[],label='GA')
plt.legend()
plt.show()
plt.close()
