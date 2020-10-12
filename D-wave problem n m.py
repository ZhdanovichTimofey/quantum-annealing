from dwave.system import DWaveSampler, EmbeddingComposite
import dimod
import pyqubo
import neal
sampler = EmbeddingComposite(DWaveSampler())
n = int(input())
m = int(input())
a = [0 for i in range(n)]
b = [0 for j in range(n)]
x = pyqubo.Array.create('x', shape=n, vartype='BINARY')
a = list(map(int, input().split()))
for i in range(n):
    b[i] = -4*m*max(a) - a[i]
H = sum(w * s for s, w in zip(x, b)) + 2 * max(a) * sum(s for s in x)**2
Q, offset = H.compile().to_qubo()
sampleset = sampler.sample_qubo(Q, num_reads=5000)
print(sampleset)
