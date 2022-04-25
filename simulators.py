import numpy as np
from scipy import signal as sgl
import numba
from torch import as_tensor
from parameters_info import sample_uniform_within_range

####################### No noise #######################

@numba.njit 
def simulate_euler(parameters,length: int = 302, dt: int = 1, num_sim: int = 2000, initial_conditions = np.array([0.25,0.25])):
    # Input parameters are of shape (num_nodes, num_parameters) to match parameter estimation output, but we need (num_parameters,num_nodes) to simulate
    params = parameters.T 
    # Equivalent to allocating memory
    time_series_E = np.empty((int(1000/dt*length),int(num_sim)))
    time_series_I = np.empty((int(1000/dt*length),int(num_sim)))
    # Set initial conditions
    time_series_E[0] = initial_conditions[0]
    time_series_I[0] = initial_conditions[1]
    # Forward euler performed in-place within the time_series_X arrays to maximize speed
    for i in range(int(1000/dt*length)-1):
        time_series_E[i+1] = params[0] * time_series_E[i] - params[2] * time_series_I[i] + params[18] - params[9]
        time_series_I[i+1] = params[1] * time_series_E[i] - params[3] * time_series_I[i] + params[19] - params[13]
        time_series_E[i+1] = params[8] / (1 + np.exp(-params[6]* (params[20] * time_series_E[i+1] - params[7])))
        time_series_I[i+1] = params[12] / (1 + np.exp(-params[10]* (params[21] * time_series_I[i+1] - params[11])))
        time_series_E[i+1] = dt * (((params[16] - params[14] * time_series_E[i]) * time_series_E[i+1]) - time_series_E[i]) / params[4] 
        time_series_I[i+1] = dt * (((params[17] - params[15] * time_series_I[i]) * time_series_I[i+1]) - time_series_I[i]) / params[5] 
        time_series_E[i+1] += time_series_E[i] 
        time_series_I[i+1] += time_series_I[i] 
    return time_series_E, time_series_I

@numba.njit
def simulate_heun(parameters, length: int = 302, dt: int = 1, num_sim: int = 2000, initial_conditions = np.array([0.25,0.25])):
    # Input parameters are of shape (num_nodes, num_parameters) to match parameter estimation output, but we need (num_parameters,num_nodes) to simulate
    params = parameters.T 
    # Equivalent to allocating memory
    time_series_E = np.empty((int(1000/dt*length),int(num_sim)))
    time_series_I = np.empty((int(1000/dt*length),int(num_sim)))
    time_series_E_temp = np.empty((1,int(num_sim)))
    time_series_I_temp = np.empty((1,int(num_sim)))
    time_series_E_corr = np.empty((1,int(num_sim)))
    time_series_I_corr = np.empty((1,int(num_sim)))
    # Set initial conditions
    time_series_E[0] = initial_conditions[0]
    time_series_I[0] = initial_conditions[1]
    # Forward heun performed in-place within the time_series_X arrays to maximize speed
    for i in range(int(1000/dt*length)-1):
        # Forward euler
        time_series_E[i+1] = params[0] * time_series_E[i] - params[2] * time_series_I[i] + params[18] - params[9]
        time_series_I[i+1] = params[1] * time_series_E[i] - params[3] * time_series_I[i] + params[19] - params[13]
        time_series_E[i+1] = params[8] / (1 + np.exp(-params[6]* (params[20] * time_series_E[i+1] - params[7])))
        time_series_I[i+1] = params[12] / (1 + np.exp(-params[10]* (params[21] * time_series_I[i+1] - params[11])))
        time_series_E[i+1] = dt*(((params[16] - params[14] * time_series_E[i]) * time_series_E[i+1]) - time_series_E[i]) / params[4] 
        time_series_I[i+1] = dt*(((params[17] - params[15] * time_series_I[i]) * time_series_I[i+1]) - time_series_I[i]) / params[5] 
        time_series_E_temp = time_series_E[i] + time_series_E[i+1] 
        time_series_I_temp = time_series_I[i] + time_series_I[i+1]
        # Corrector point
        time_series_E_corr = params[0] * time_series_E_temp - params[2] * time_series_I_temp + params[18] - params[9]
        time_series_I_corr = params[1] * time_series_E_temp - params[3] * time_series_I_temp + params[19] - params[13]
        time_series_E_corr = params[8] / (1 + np.exp(-params[6]* (params[20] * time_series_E_corr - params[7])))
        time_series_I_corr = params[12] / (1 + np.exp(-params[10]* (params[21] * time_series_I_corr - params[11])))
        time_series_E_corr = dt*(((params[16] - params[14] * time_series_E_temp) * time_series_E_corr) - time_series_E_temp) / params[4] 
        time_series_I_corr = dt*(((params[17] - params[15] * time_series_I_temp) * time_series_I_corr) - time_series_I_temp) / params[5]
        # Heun point
        time_series_E[i+1] = time_series_E[i] + (time_series_E[i+1]+time_series_E_corr)/2 
        time_series_I[i+1] = time_series_I[i] + (time_series_I[i+1]+time_series_I_corr)/2         
    return time_series_E, time_series_I


####################### With noise #######################

@numba.njit 
def simulate_euler_noise(parameters, length: int = 302, dt: int = 1, num_sim: int = 2000, initial_conditions = np.array([0.25,0.25]), noise_seed: int = 42):
    # Input parameters are of shape (num_nodes, num_parameters) to match parameter estimation output, but we need (num_parameters,num_nodes) to simulate
    params = parameters.T 
    # Set seed
    np.random.seed(noise_seed)
    # White noise
    DE, DI = np.sqrt(2*params[-1]), np.sqrt(2*params[-2])
    # Equivalent to allocating memory
    time_series_E = np.empty((int(1000/dt*length),int(num_sim)))
    time_series_I = np.empty((int(1000/dt*length),int(num_sim)))
    # Set initial conditions
    time_series_E[0] = initial_conditions[0]
    time_series_I[0] = initial_conditions[1]
    # Forward euler performed in-place within the time_series_X arrays to maximize speed
    for i in range(int(1000/dt*length)-1):
        time_series_E[i+1] = params[0] * time_series_E[i] - params[2] * time_series_I[i] + params[18] - params[9]
        time_series_I[i+1] = params[1] * time_series_E[i] - params[3] * time_series_I[i] + params[19] - params[13]
        time_series_E[i+1] = params[8] / (1 + np.exp(-params[6]* (params[20] * time_series_E[i+1] - params[7])))
        time_series_I[i+1] = params[12] / (1 + np.exp(-params[10]* (params[21] * time_series_I[i+1] - params[11])))
        time_series_E[i+1] = dt * (((params[16] - params[14] * time_series_E[i]) * time_series_E[i+1]) - time_series_E[i]) / params[4] 
        time_series_I[i+1] = dt * (((params[17] - params[15] * time_series_I[i]) * time_series_I[i+1]) - time_series_I[i]) / params[5] 
        time_series_E[i+1] += time_series_E[i] + np.random.normal(0,1,size=num_sim) *  DE
        time_series_I[i+1] += time_series_I[i] + np.random.normal(0,1,size=num_sim) *  DI 
    return time_series_E, time_series_I

@numba.njit
def simulate_heun_noise(parameters, length: int = 302, dt: int = 1, num_sim: int = 2000, initial_conditions = np.array([0.25,0.25]), noise_seed: int = 42):
    # Input parameters are of shape (num_nodes, num_parameters) to match parameter estimation output, but we need (num_parameters,num_nodes) to simulate
    params = parameters.T 
    # Set seet
    np.random.seed(noise_seed)
    # White noise
    DE, DI = np.sqrt(2*params[-1]), np.sqrt(2*params[-2])
    # Equivalent to allocating memory
    time_series_E = np.empty((int(1000/dt*length),int(num_sim)))
    time_series_I = np.empty((int(1000/dt*length),int(num_sim)))
    time_series_E_temp = np.empty((1,int(num_sim)))
    time_series_I_temp = np.empty((1,int(num_sim)))
    time_series_E_corr = np.empty((1,int(num_sim)))
    time_series_I_corr = np.empty((1,int(num_sim)))
    time_series_E_noise = np.empty((1,int(num_sim)))
    time_series_I_noise = np.empty((1,int(num_sim)))
    # Set initial conditions
    time_series_E[0] = initial_conditions[0]
    time_series_I[0] = initial_conditions[1]
    # Heun performed in-place within the time_series_X arrays to maximize speed
    for i in range(int(1000/dt*length)-1):
        # Forward Euler
        time_series_E[i+1] = params[0] * time_series_E[i] - params[2] * time_series_I[i] + params[18] - params[9]
        time_series_I[i+1] = params[1] * time_series_E[i] - params[3] * time_series_I[i] + params[19] - params[13]
        time_series_E[i+1] = params[8] / (1 + np.exp(-params[6]* (params[20] * time_series_E[i+1] - params[7])))
        time_series_I[i+1] = params[12] / (1 + np.exp(-params[10]* (params[21] * time_series_I[i+1] - params[11])))
        time_series_E[i+1] = dt*(((params[16] - params[14] * time_series_E[i]) * time_series_E[i+1]) - time_series_E[i]) / params[4] 
        time_series_I[i+1] = dt*(((params[17] - params[15] * time_series_I[i]) * time_series_I[i+1]) - time_series_I[i]) / params[5] 
        time_series_E_noise = np.random.normal(0,1,size=num_sim) *  DE
        time_series_I_noise = np.random.normal(0,1,size=num_sim) *  DI
        time_series_E_temp = time_series_E[i] + time_series_E[i+1] + time_series_E_noise
        time_series_I_temp = time_series_I[i] + time_series_I[i+1] + time_series_I_noise
        # Corrector point
        time_series_E_corr = params[0] * time_series_E_temp - params[2] * time_series_I_temp + params[18] - params[9]
        time_series_I_corr = params[1] * time_series_E_temp - params[3] * time_series_I_temp + params[19] - params[13]
        time_series_E_corr = params[8] / (1 + np.exp(-params[6]* (params[20] * time_series_E_corr - params[7])))
        time_series_I_corr = params[12] / (1 + np.exp(-params[10]* (params[21] * time_series_I_corr - params[11])))
        time_series_E_corr = dt*(((params[16] - params[14] * time_series_E_temp) * time_series_E_corr) - time_series_E_temp) / params[4] 
        time_series_I_corr = dt*(((params[17] - params[15] * time_series_I_temp) * time_series_I_corr) - time_series_I_temp) / params[5]
        # Heun point
        time_series_E[i+1] = time_series_E[i] + (time_series_E[i+1]+time_series_E_corr)/2 + time_series_E_noise
        time_series_I[i+1] = time_series_I[i] + (time_series_I[i+1]+time_series_I_corr)/2 + time_series_I_noise  
    return time_series_E, time_series_I

def compute_PSD(timeseries, dt):
    freq , psds = sgl.welch(timeseries,fs=(1000/dt), nperseg=2000/dt, axis = 0)
    return psds[:100,:].T, freq[:100]

def WC_stochastic_heun_PSD(parameters,
                           length: int = 302,
                           dt: int = 1,
                           initial_conditions = np.array([0.25,0.25]),
                           noise_seed: int = 42,
                           PSD_cutoff = 100,
                           out_tensor = True,
                           get_psd_I = True,
                           filter_bad = True,
                           remove_bad = False):
    try:
        parameters = parameters.detach().cpu().numpy()
    except: 
        pass
    time_series_E, time_series_I = simulate_heun_noise(parameters,length = length, dt=dt, num_sim = parameters.shape[0],noise_seed=noise_seed, initial_conditions=initial_conditions)
    bad_sims = np.logical_or(np.logical_or(np.amax(time_series_E,axis=0)>1.1, np.amax(time_series_I,axis=0)>1.1), np.logical_or(np.amin(time_series_E,axis=0)<-0.1, np.amin(time_series_I,axis=0)<-0.1))
    if filter_bad:
        time_series_E[:,bad_sims] *= 0
        time_series_I[:,bad_sims] *= 0
    if remove_bad:
        time_series_E = time_series_E[:,np.logical_not(bad_sims)]
        time_series_I = time_series_I[:,np.logical_not(bad_sims)]
        try:
            if time_series_E.shape[1] < 1:
                if get_psd_I:
                    return np.array([[-1 for i in range(PSD_cutoff)]]), np.array([[-1 for i in range(PSD_cutoff)]]), np.array([[-1 for i in range(PSD_cutoff)]])
                else:
                    return np.array([[-1 for i in range(PSD_cutoff)]]), np.array([[-1 for i in range(PSD_cutoff)]])
        except:
            if get_psd_I:
                return np.array([[-1 for i in range(PSD_cutoff)]]), np.array([[-1 for i in range(PSD_cutoff)]]), np.array([[-1 for i in range(PSD_cutoff)]])
            else:
                return np.array([[-1 for i in range(PSD_cutoff)]]), np.array([[-1 for i in range(PSD_cutoff)]])
    freq , psd_E = sgl.welch(time_series_E,fs=(1000/dt), nperseg=2000/dt, axis = 0)
    psd_E, freq = psd_E.T[:,:PSD_cutoff], freq[:PSD_cutoff]
    if get_psd_I:
        _ , psd_I = sgl.welch(time_series_I,fs=(1000/dt), nperseg=2000/dt, axis = 0)
        psd_I = psd_I.T[:,:PSD_cutoff]
        if out_tensor:
            psd_E, psd_I, freq = as_tensor(psd_E), as_tensor(psd_I), as_tensor(freq)
        return psd_E, psd_I, freq
    else:
        time_series_I = None
        if out_tensor:
            psd_E, freq = as_tensor(psd_E), as_tensor(freq)
        return psd_E, freq