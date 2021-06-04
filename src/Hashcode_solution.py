#!/usr/bin/env python
# coding: utf-8

# In[126]:


import pandas as pd
import numpy as np
import pulp as pulp
import time,os,sys


# ## Read File contents

# In[131]:


#filename = "a_example.in"
#filename = "b_small.in"
#filename = "c_medium.in"
#filename = "d_quite_big.in"
#filename = "e_also_big.in"
filename = sys.argv[1]
verbose = int(sys.argv[2])

#read file contents
f= open(filename,"r")
contents = f.read()
print("File read successfully !!! File contents :\n")
print("Print complete solution logs:",bool(verbose))

#extract line-wise information
line1=contents.split('\n')[0]
line2=contents.split('\n')[-2]
#print("line 1: ",line1)
#print("line 2: ",line2)

max_pizza_pieces = int(line1.split(' ')[0])
pizza_types = int(line1.split(' ')[1])
slices_per_pizza = [int(x) for x in line2.split(' ')]
slices_per_pizza = np.array(slices_per_pizza)

assert pizza_types == len(slices_per_pizza)
print("\n========== INFORMATION FROM THE FILE ===========\n")
print("Types of pizzas available (N):",pizza_types)
print("Number of Slices per pizza (X):",slices_per_pizza)
print("\nMax pizza pieces to be distributed (M):",max_pizza_pieces)
print("Sum of pizza pieces available :",np.sum(slices_per_pizza))


# ## Linear Optimization

# In[137]:


my_lp_problem = pulp.LpProblem("Pizza_Pieces_Optimization", pulp.LpMaximize)

# Definition of decision variables
X = slices_per_pizza
Y =  [None]*pizza_types
M = max_pizza_pieces

for i in range(len(Y)):
    var_name="y"+str(i)
    Y[i] = pulp.LpVariable(var_name, lowBound=0, upBound=1, cat='Binary')

##### FORMULATION #####
# Objective function
my_lp_problem += np.dot(X,Y), "Z"

# Constraints
my_lp_problem += np.dot(X,Y) <= M
if verbose==1:
    print("\n============== LP Problem Formulation ============\n")
    print(my_lp_problem)
print("Solving the optmization problem now...")
start_time=time.time()
my_lp_problem.solve()
if pulp.LpStatus[my_lp_problem.status]=='Optimal':
    print("Problem Solved !!!")
    print("Time taken :",time.time()-start_time,"sec")
    if verbose==1:
        print("Solution variables:")
        for variable in my_lp_problem.variables():
            print("{} = {}".format(variable.name, variable.varValue))
    
    sol_pizza_types=[int(variable.name[1:]) for variable in my_lp_problem.variables() if variable.varValue == 1]
    sorted_types = sorted(sol_pizza_types)
    print("\n==============  Submission Solution  ==============\n")
    print("[Problem] Max pizza pieces to be distributed (M):",max_pizza_pieces)
    print("[Problem] Types of pizzas available (N):",pizza_types,"\n")
    print("[Solution] Maximized value of pizza pieces :",pulp.value(my_lp_problem.objective))
    print("[Solution] Number of pizza types required:", len(sorted_types))
    print("[Solution] Ordering pizza types:")
    print(' '.join(map(str,sorted_types)))
    sol_filename = "sol_"+filename
    f1=open(sol_filename, 'w+')
    f1.write(str(len(sorted_types)))
    f1.write("\n")
    f1.write(' '.join(map(str,sorted_types)))
    f1.close()
    print("\nThe submission solution has been saved into file '{}' in this directory '{}'.".format(sol_filename,os.getcwd()))
else:
    print("Solution could not be found !!!")


# END OF SCRIPT #



