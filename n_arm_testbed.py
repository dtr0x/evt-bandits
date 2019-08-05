from tce import *
import matplotlib.pyplot as plt
import time

np.set_printoptions(precision=3, threshold=np.inf, suppress=True)

nB = 2000
nA = 10
nP = 10000

alph = 0.99

np.random.seed(0)

armSigmas = np.random.uniform(size = nA)
armTCEs = np.zeros(nA)
trueTCEs = tce_lnorm(alph, 0, armSigmas)
bestArm = np.argmin(trueTCEs)

percentBestAction = np.zeros(nP)
allSamples = np.zeros((nA, nP))
armSelected = np.zeros(nP)

eps = np.concatenate((np.linspace(1, 0.1, 1000, endpoint=False), np.linspace(0.1, 0, 9000)))

start = time.time()
for i in range(nP):
    if np.random.uniform() <= eps[i]: # pick random arm
        arm = np.random.randint(nA)
    else:
        arm = np.argmin(armTCEs)

    allSamples[arm, i] = lognorm.rvs(armSigmas[arm])
    armSelected[i] = arm
    currentSamp = allSamples[arm, np.nonzero(allSamples[arm, ])]
    try:
        tce = tce_ad(currentSamp, alph)
        if np.isnan(tce):
            print("NaN encountered at timestep", i)
            armTCEs[arm] = tce_sa(currentSamp, alph)
        else:
            armTCEs[arm] = tce
    except ValueError:
        print("Error encountered at timestep", i)
        armTCEs[arm] = tce_sa(currentSamp, alph)

    percentBestAction[i] = sum(armSelected[:i] == bestArm)/(i+1)
end = time.time()

print("Took {:.3f} seconds.".format(end - start))

plt.plot(percentBestAction)
plt.xlabel("Time Steps")
plt.ylabel("% Best Action")
plt.title("10 Arm Testbed")
plt.show()






