import sys
import numpy as np

print("Hello world!")

output_vector = np.array([0.123, 1.234, 2.345, 3.456])
print(type(output_vector))

# Export txt file
np.savetxt('../python/outputs/output.txt', output_vector, delimiter=',')