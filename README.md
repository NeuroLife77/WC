# Wilson-Cowan custom implementation  
Custom implementation of W-C neural mass model (single node) that can perform parallel simulations in numpy array format (single core) accelerated with numba JIT compiler. Integration schemes available are forward euler and heun (possibility of additive white noise in both).

Built to explore the potential ways to implement the model as to maximize sampling of the parameter space for single-node parameter estimation/parameter recovery. 
Will potentially add some multi-scale interactions in the future.

Attempted to implement the model in C to maximize speed, but it apparently is about 10x slower than the optimized python code implemented with numpy and numba JIT for independent parallel (single core) simulations.
