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
b = list(map(abs, a))
H = 100 * max(b) * (sum(b for b in x) - m)**2  - max (b) * sum(k * b for k, b in zip(a, x))
Q, offset = H.compile().to_qubo()
sampleset = sampler.sample_qubo(Q, num_reads=5000)
print(sampleset)
