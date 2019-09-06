shape = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2]
alph = [0.99, 0.999]
tp_init = [0.7, 0.8, 0.9]

for i in shape:
    for j in alph:
        for k in tp_init:
            print("bash fraction_closer.sh lnorm_{0}_0_{1}_{2}_40_0.1_0.9".format(i, j, k))