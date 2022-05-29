# Wilson-Cowan custom implementation  
Custom implementation of W-C neural mass model (single node) that can perform parallel simulations in numpy array format (single core) accelerated with numba JIT compiler. Integration schemes available are forward euler and heun (possibility of additive white noise in both).

Built to explore the potential ways to implement the model as to maximize sampling of the parameter space for single-node parameter estimation/parameter recovery. 
Now also includes a version that simulates a set of coupled/connected neural masses.

Attempted to implement the model in C to maximize speed, but it is about 10x slower than the optimized python code implemented with numpy and numba JIT when used to perform a set of independent simulations. Most likely due to numpy being optimized for array operations. Would need to further optimize the C implementation to get the most of it in that regard. 

As is, the C implementation should be faster for small sets of simulations while the python (numpy with numba) should be faster for larger sets. The python implementation should then be more appropriate for simulating large samples from the parameter space.
