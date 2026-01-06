import numpy as np

Ntomo = 10
Ncl = 20

size = int((Ntomo * (Ntomo + 1)/2 + Ntomo + Ntomo * Ntomo ) * Ncl)
print('Size = ', size)
datav = np.zeros(size)
mask = np.zeros(size)

index = np.arange(size)
dv_KL = np.loadtxt('Roman_Ntomo10_KL.datavector', usecols=(1))
size_KL = len(dv_KL)
datav[:size_KL] = dv_KL
mask[:size_KL] = 1
np.savetxt('roman_kl.datavector', np.column_stack((index, datav)), fmt='%d %1.6e')
np.savetxt('roman_kl.mask', np.column_stack((index, mask)), fmt='%d %1.1f')

cov_fname = 'Roman_ssss_cov_Ncl20_Ntomo10'
rows = []
with open(cov_fname, 'r') as f:
    for line in f:
        parts = line.strip().split()
        rows.append(parts[:10])
arr = np.array(rows, dtype=object)
i_in = arr[:, 0].astype(int)
j_in = arr[:, 1].astype(int)

data = {}
for r in rows:
    i = int(r[0])
    j = int(r[1])
    data[(i, j)] = r

ell1, ell2, tomo1, tomo2, tomo3, tomo4 = (1.0, 1.0, 0, 0, 0, 0)
def fmt_float(x: float) -> str:
    return f"{float(x):.6e}"

out_fname = 'Roman_cov_Ncl20_Ntomo10'
with open(out_fname, 'w') as f:
    for i in range(size):
        for j in range(i, size):
            key = (i, j)
            if key in data:
                f.write(' '.join(data[key]) + '\n')
            else:
                f.write(f'{i:d} {j:d} {fmt_float(ell1)} {fmt_float(ell2)} {tomo1:d} {tomo2:d} {tomo3:d} {tomo4:d} ' +
                        f'{fmt_float(0.0)} {fmt_float(0.0)}\n')
