import numpy as np

class Distribution:
    def rand(self, n):
        p = np.random.uniform(size=n)
        return self.var(p)
