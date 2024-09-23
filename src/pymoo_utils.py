import numpy as np


def recover_pymoo(dirname,pop_size):
    par = [[] for x in range(pop_size)]
    #par = []
    for i in range(pop_size):
        with open(dirname + str(i) + "/parameters.txt",'r') as f:
            for lines in f:

                par[i].append(lines)
                
    par = np.array(par)

    x = []
    for i in range(pop_size):
        s = par[i,0].split(" ")
        s = np.array(s)
        s = s.astype(float).astype(int)
        x.append(s)
    x = np.array(x)

    acc = [[] for x in range(pop_size)]
    #par = []
    for i in range(pop_size):
        with open(dirname + str(i) + "/accuracy.txt",'r') as f:
            for lines in f:
                acc[i].append(lines)           
    acc = np.array(acc)
    f1 = []
    for i in range(pop_size):
        s = acc[i,0]
        s = np.array(s)
        s = s.astype(float)
        f1.append(s)
    f1 = np.array(f1)


    size = [[] for x in range(pop_size)]
    for i in range(pop_size):
        with open(dirname + str(i) + "/size.txt",'r') as f:
            for lines in f:
                size[i].append(lines)

    size = np.array(size)
    f2 = []
    for i in range(pop_size):
        s = size[i,0]
        s = np.array(s)
        s = s.astype(float)
        f2.append(s)
    f2 = np.array(f2)
    f2.shape

    f = np.concatenate(([f1],[f2]))
    return x,f
