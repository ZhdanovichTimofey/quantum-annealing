import pandas_datareader.data as web
import pandas as pd
from dwave.system import DWaveSampler, EmbeddingComposite
import dimod
import pyqubo
import neal
names = ['KRKNP', 'MFGS', 'USBN', 'BSPB', 'KOGK', 'UKUZ',
         'AMEZ', 'URKZ', 'ACKO', 'SFIN', 'OBUV', 'FEES', 'MRKV']
neftegaz = [['KRKNP', 2.84, 11700], ['MFGS', 3.15, 284], ['GAZP', 3.25, 165.27]]
banks = [['USBN', 1.53, 0.0656], ['BSPB', 2.70, 44.9], ['VTBR', 4.76, 0.034005]]
gornodob = [['KOGK', 1.90, 45000], ['UKUZ', 2.67, 496], ['ALNU', 2.72, 56800]]
metal = [['AMEZ', 1.52, 4.64], ['URKZ', 1.62, 14920], ['CHMK', 2.04, 2525]]
finance = [['ACKO', 2.77, 4.66], ['SFIN', 5.68, 468.2], ['AFKS', 5.72, 31.649]]
retail = [['OBUV', 2.12, 31.65], ['MEAN', 5.71, 0], ['MEAN', 5.71, 0]]
elseti = [['FEES', 2.92, 0.1984], ['MRKV', 3.93, 0.06975],
          ['MRKP', 3.92, 0.2327]]
B = 1000000
sampler = EmbeddingComposite(DWaveSampler())
Ef = []
f = []
cost_max = [0 for k in range(len(names))]
matrix_covariation = [[] for k in range(len(names))]


def efficiency(E):
    for i in range(2):
        E += [neftegaz[2][1] - neftegaz[i][1]]
    for i in range(2):
        E += [banks[2][1] - banks[i][1]]
    for i in range(2):
        E += [gornodob[2][1] - gornodob[i][1]]
    for i in range(2):
        E += [metal[2][1] - metal[i][1]]
    for i in range(2):
        E += [finance[2][1] - finance[i][1]]
    for i in range(2):
        E += [retail[2][1] - retail[i][1]] \
    for i in range(2):
        E += [elseti[2][1] - elseti[i][1]]
    return E


def covariation(x, y):
    s = pd.Series((x - M(x)) * (y - M(y)))
    cov = M(pd.Series((x - M(x)) * (y - M(y))))
    return cov


def M(x):
    m = x.mean()
    return m


def data_read(f):
    for i, n in enumerate(names):
        f += [web.DataReader(n, 'moex', start='2019-01-01',
                             end='2019-12-31')]
        cost_max[i] = f[i]['HIGH']
    return f


def create_matrix_covariation(matrix_covariation):
    for i in range(len(names)):
        for j in range(len(names)):
            matrix_covariation[i] += [covariation(cost_max[i], cost_max[j])]
    return matrix_covariation


B1 = B / len(names)
Ef = B1 * efficiency(Ef)
f = data_read(f)
matrix_covariation = create_matrix_covariation(matrix_covariation)
'''A = []
for i in range(len(Ef)):
    A += [B*2 / len(Ef)]'''
x = pyqubo.Array.create('x', shape=len(Ef), vartype='BINARY')
H1 = -sum(a * e for a, e in zip(x, Ef))
for i in range(len(Ef)):
    for j in range(len(Ef)):
        H1 += x[i] * x[j] * matrix_covariation[i][j]
#H2 = (sum(a * A for a, A in zip(x, A)) - B)**2
H = H1 + H2
Q, offset = H.compile().to_qubo()
sampleset = sampler.sample_qubo(Q, num_reads=5000)
print(sampleset)
