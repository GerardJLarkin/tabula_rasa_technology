# import numpy as np
# from scipy.special import bernoulli, zeta
# ouput = bernoulli(1000)
# from time import perf_counter, process_time, clock_gettime_ns, CLOCK_REALTIME

# test_time = clock_gettime_ns(CLOCK_REALTIME)

# print(int(str(test_time)[-1]))

import random
import string

# get random password pf length 8 with letters, digits, and symbols
characters = string.ascii_letters + string.digits
password = ''.join(random.choice(characters) for i in range(6))
print("Random password is:", password)