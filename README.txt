This resampling code is intended for resampling one dimensional spectra between arbitrary wavelength solutions.

to use simply either add the directory containing resampling.py to the PYTHONPATH or add a path file resampling.pth to a directory already in your python path probably /lib/site-packages  To see what your python path actually is you can fire up the interpreter and type

import sys
sys.path

The workhorse function is resampling.get_resampling_matrix look to the documentation in the code.