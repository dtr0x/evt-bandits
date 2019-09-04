shape = [0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.75, 2.5, 5.0]
alph = [0.99, 0.999]
tp_init = [0.7, 0.8, 0.9]

for i in shape:
    for j in alph:
        for k in tp_init:
            print("bash fraction_closer.sh weibull_{0}_1_{1}_{2}_40_0.1_0.9".format(i, j, k))