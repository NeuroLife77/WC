# Wilson-Cowan custom implementation w/ forward euler 
Custom implementation of W-C neural mass model (single node) that can perform parallel simulations in numpy array format (single core) accelerated with numba JIT compiler.

Built to explore the potential ways to implement the model as to maximize sampling of the parameter space for single-node parameter estimation/parameter recovery. Intentions to replace the forward euler with Heun and add some noise.
Will potentially add some multi-scale interactions in the future.
