"""A Parallel Implementation of MCMC in CUDA"""

from numba import jit, cuda, float64
from numba.cuda.random import xoroshiro128p_uniform_float64, xoroshiro128p_normal_float64, init_xoroshiro128p_state, xoroshiro128p_jump, xoroshiro128p_dtype

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


class pMCMC:        
    """Host code of our parallel MCMC implementation.
    """
    def __init__(self, data, block_size, n_iter, seed=0):
        self.data = cuda.to_device(data)
        self.n_iter = n_iter
        
        # Parameters for the kernel launch 
        self.block_size = block_size
        self.n_samples = data.shape[0]
        self.n_blocks = self.n_samples // block_size
        
        # Allocate an output array on the GPU
        self.output = cuda.device_array((n_iter,self.n_blocks,2))
        
        # Create random number generators for each thread
        # NOTE: The threads within the same block should generate the same random numbers
        rng_states = np.empty(self.n_samples, dtype=xoroshiro128p_dtype)
        for i in range(self.n_samples):
            init_xoroshiro128p_state(rng_states, i, seed)  # Init to a fixed state
            for j in range(i//block_size):  # Jump forward block_index*2^64 steps
                xoroshiro128p_jump(rng_states, i)
        self.rng_states = cuda.to_device(rng_states)  # Copy it to the GPU
        
    def launch(self):
        """Launches the kernel and returns the MCMC samples.
        """
        mcmc[ self.n_blocks, self.block_size ]( self.data, self.output, self.rng_states, self.n_iter)
        return self.output.copy_to_host()
        
    @staticmethod
    def generate_data(n_samples):
        """Generates and returns a hyperparameter theta and n_samples noisy observations of theta."""
        theta =  np.random.multivariate_normal([1,1],cov=[[1, 0],[0, 1]])
        data = np.random.multivariate_normal(theta,cov=[[0.1, 0],[0, 0.1]],size=n_samples)
        return theta, data



@cuda.jit
def mcmc(data, output, rng_states, n_iter):
    """Device code of our parallel MCMC implementation.
    """
    shared = cuda.shared.array(shape=(2**9,), dtype=float64)  # Shared Memory
    tx = cuda.threadIdx.x  # Thread ID
    ty = cuda.blockIdx.x  # Block ID
    bw = cuda.blockDim.x  # Block Size
    idx = bw*ty+tx  # Global ID
    
    theta = (0.,0.)  # Initialize theta
    x = data[idx]  # Fetch the data point
    logp_x = -(((theta[0]-x[0])**2)/(2*0.1) + ((theta[1]-x[1])**2)/(2*0.1))  # Log-likelihood of the data point
    shared[tx] = logp_x  # Put the log-likelihood to the shared memory
    cuda.syncthreads()
    
    # Reduction using sequential addressing. NOTE: Increasing the data points per thread might increase the performance
    s = bw//2
    while s>0:
        if tx < s:
            shared[tx] += shared[tx+s]
        cuda.syncthreads()
        s>>=1
    # Get the log-likelihood of the sub-dataset from the first position
    logp = shared[0]  #  NOTE: Might cause some performance issues
    
    # Add the log-prior
    log_prior = -(((theta[0]-1)**2)/2 + ((theta[1]-1)**2)/2) 
    logp += log_prior/2
    
    # Main MCMC Loop
    for i in range(n_iter):
        # Propose a new theta
        theta_ = (theta[0] + 0.1*xoroshiro128p_normal_float64(rng_states, idx), theta[1] + 0.1*xoroshiro128p_normal_float64(rng_states, idx))
        logp_x = -(((theta_[0]-x[0])**2)/(2*0.1) + ((theta_[1]-x[1])**2)/(2*0.1))  # Log-likelihood of the data point
        shared[tx] = logp_x  # Put the log-likelihood to the shared memory
        cuda.syncthreads()
        
        # Reduction using sequential addressing
        s = bw//2
        while s>0:
            if tx < s:
                shared[tx] += shared[tx+s]
            cuda.syncthreads()
            s>>=1
        # Get the log-likelihood;
        # this will trigger a "broadcast", see https://devblogs.nvidia.com/using-shared-memory-cuda-cc/   
        logp_ = shared[0]
        
        # Add the log-prior
        log_prior = -(((theta_[0]-1)**2)/2 + ((theta_[1]-1)**2)/2) 
        logp_ += log_prior/2
        
        # Acceptance ratio
        alpha = math.exp(min(0,logp_-logp))
        # Draw a uniform random number
        u = xoroshiro128p_uniform_float64(rng_states, idx)
        # Accept/Reject?
        if u < alpha:
            theta = theta_
            logp = logp_
        
        # Write the sample to the memory
        if tx == 0:
            output[i,ty] = theta

def confidence_ellipse(mean, cov, ax, n_std=1.0, edgecolor='black', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        edgecolor=edgecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean[0], mean[1])

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)