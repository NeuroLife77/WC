import numpy as np
from parameters_info import pars

def wilson_cowan_step_dict(E,I,dt,pars):
    E_input = pars['c_ee']*E - pars['c_ie']*I + pars['P'] - pars['theta_e']
    I_input = pars['c_ei']*E - pars['c_ii']*I + pars['Q'] - pars['theta_i']
    E_input = pars['c_e']/(1 + np.exp(-pars['a_e']* (pars['alpha_e'] * E_input - pars['b_e'])))
    I_input = pars['c_i']/(1 + np.exp(-pars['a_i']* (pars['alpha_i'] * I_input - pars['b_i'])))
    dE = (((pars['k_e'] - pars['r_e'] * E) * E_input) - E) / pars['tau_e'] 
    dI = (((pars['k_i'] - pars['r_i'] * I) * I_input) - I) / pars['tau_i'] 
    return E + dE*dt, I + dI*dt



# 34.1 seconds for 302s of sim with dt = 1ms and 2000 parallel unconnected nodes (pure python)
def simulate_wc():
    length = 302
    dt = 1
    time_series_E = np.zeros((1000//dt*length,2000))
    time_series_I = np.zeros((1000//dt*length,2000))
    time_series_I[0] = 0.25
    time_series_E[0] = 0.25

    for i in range(1000//dt*length-1):
        time_series_E[i+1] = pars['c_ee']*time_series_E[i] - pars['c_ie']*time_series_I[i] + pars['P'] - pars['theta_e']
        time_series_I[i+1] = pars['c_ei']*time_series_E[i] - pars['c_ii']*time_series_I[i] + pars['Q'] - pars['theta_i']
        time_series_E[i+1] = pars['c_e']/(1 + np.exp(-pars['a_e']* (pars['alpha_e'] * time_series_E[i+1] - pars['b_e'])))
        time_series_I[i+1] = pars['c_i']/(1 + np.exp(-pars['a_i']* (pars['alpha_i'] * time_series_I[i+1] - pars['b_i'])))
        time_series_E[i+1] = dt * (((pars['k_e'] - pars['r_e'] * time_series_E[i]) * time_series_E[i+1]) - time_series_E[i]) / pars['tau_e'] 
        time_series_I[i+1] = dt * (((pars['k_i'] - pars['r_i'] * time_series_I[i]) * time_series_I[i+1]) - time_series_I[i]) / pars['tau_i'] 
        time_series_E[i+1] += time_series_E[i] 
        time_series_I[i+1] += time_series_I[i] 
