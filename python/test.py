import sys
import numpy as np

print("Hello world!")

testArray = ([0, 1, 2, 300])

# Export txt file
np.savetxt('../python/outputs/output.txt', testArray, delimiter=',')