# ==============================
# Randomized response.
# ==============================


from __future__ import division
import numpy as np


def epsilon2probability(epsilon):
    return (np.e ** epsilon) / (np.e ** epsilon + 1)

def perturbation(p):

    rnd = np.random.random()

    return 1 if rnd < p else 0

def Em(n,m,p):

    return (2*p-1)*m-(p-1)*n

def RR(n,m,s,epsilon):

    p = epsilon2probability(epsilon)

    em = Em(n,m,p)

    participate = []

    client = np.zeros((n),dtype=int)

    for _ in s:
        client[_]=1

    for i in range(n):
        client[i] = 1 - client[i] ^ perturbation(p)
        if client[i]==1:
            participate.append(int(i))

    _m = len(participate)

    return em,_m,participate


def Coincidence(n,m,s,eps):

    em, _m, participate = RR(n, m, s, eps)

    coincidence = (len(set(participate) & set(s)))

    coincidence_rate = float(coincidence) / n

    return participate, coincidence, coincidence_rate



if __name__ == '__main__':

    n=100
    m=30
    eps=1.386

    o=[]

    # a=[50]*100
    # b=[46]*100
    # c=[42]*100
    # d=[38]*100

    while (1):

        for i in range(100):

            perm = np.random.permutation(n)
            s = perm[0:m].tolist()

            participate,coincidence,coincidence_rate = Coincidence(n,m,s,eps)


            o.append(len(participate))

        if np.var(o) < 12:

            print(o)
            exit()

    # print(list(a))
    # print(list(b))
    # print(list(c))
    # print(list(d))


    # print(coincidence)

    # print(coincidence_rate)


