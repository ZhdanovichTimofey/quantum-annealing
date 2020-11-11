import pandas_datareader.data as web
import pandas as pd
from dwave.system import DWaveSampler, EmbeddingComposite
import dimod
import pyqubo
import neal
names = ['KRKNP', 'MFGS', 'GAZP', 'SIBN', 'BANEP', 'RNFT', 'ROSN', 'LKOH',
         'TATN', 'USBN', 'BSPB', 'VTBR', 'SBER', 'KUZB', 'PRMB', 'ROSB', 'CBOM',
         'TCSG', 'KOGK', 'UKUZ', 'ALNU', 'RASP']
neftegaz = [['KRKNP', 2.84, 11700], ['MFGS', 3.15, 284], ['GAZP', 3.25, 165.27],
            ['SIBN', 3.41, 288.1], ['BANEP', 3.48, 1567], ['RNFT', 3.61, 299.2],
            ['ROSN', 4.42, 382.8], ['LKOH', 4.50, 4451.5],
            ['TATN', 5.62, 453.7], ['MEAN', 7.57, 0]]
banks = [['USBN', 1.53, 0.0656], ['BSPB', 2.70, 44.9], ['VTBR', 4.76, 0.034005],
         ['SBER', 5.89, 220.8], ['KUZB', 6.62, 0.0143], ['PRMB', 7.53, 25400],
         ['ROSB', 11.1, 78], ['CBOM', 11.3, 5.665], ['TCSG', 11.9, 2184.4],
         ['MEAN', 13.8, 0]]
gornodob = [['KOGK', 1.90, 45000], ['UKUZ', 2.67, 496], ['ALNU', 2.72, 56800],
            ['RASP', 5.98, 116.06], ['MEAN', 7.30, 0]]
B = 1000000
sampler = EmbeddingComposite(DWaveSampler())
Ef = []
f = []
cost_max = [0 for k in range(len(names))]
matrix_covariation = [[] for k in range(len(names))]


def efficiency(E):
    for i in range(len(neftegaz) - 1):
        E += [neftegaz[i][2] * (neftegaz[i + 1][1] - neftegaz[i][1])]
    for i in range(len(banks) - 1):
        E += [banks[i][2] * (banks[i + 1][1] - banks[i][1])]
    for i in range(len(gornodob) - 1):
        E += [gornodob[i][2] * (gornodob[i + 1][1] - gornodob[i][1])]
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


Ef = efficiency(Ef)
f = data_read(f)
matrix_covariation = create_matrix_covariation(matrix_covariation)
A = []
for i in range(len(Ef)):
    A += [B*2 / len(Ef)]
x = pyqubo.Array.create('x', shape=len(Ef), vartype='BINARY')
H1 = -sum(a * e for a, e in zip(x, Ef))
for i in range(len(Ef)):
    for j in range(len(Ef)):
        H1 += x[i] * x[j] * matrix_covariation[i][j]
H2 = (sum(a * A for a, A in zip(x, A)) - B)**2
H = H1 + H2
Q, offset = H.compile().to_qubo()
sampleset = sampler.sample_qubo(Q, num_reads=5000)
print(sampleset)
