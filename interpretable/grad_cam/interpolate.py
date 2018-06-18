import numpy as np
from scipy import interpolate


from_size1 = 5
to_size1 = 5

from_size2 = 5
to_size2 = 5

x,y=np.mgrid[0:1:(from_size1*1j), 0:1:(from_size2*1j)]
z = np.arange(from_size1*from_size2).reshape(from_size1, -1)

xn,yn=np.mgrid[0:1:(to_size1*1j), 0:1:(to_size2*1j)]
tck = interpolate.bisplrep(x,y,z,s=0)
znew = interpolate.bisplev(xn[:,0], yn[0,:], tck)
