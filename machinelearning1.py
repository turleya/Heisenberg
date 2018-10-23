from tempfile import TemporaryFile
import numpy as np

outfile = TemporaryFile()

x = np.arange(10)
#print(x)

np.save(outfile, x)

outfile.seek(0)

y = np.load(outfile)
print(y)
