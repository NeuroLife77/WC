import numpy as np

def wilson_cowan_step_dict(E,I,dt,pars):
    E_input = pars['c_ee']*E - pars['c_ie']*I + pars['P'] - pars['theta_e']
    I_input = pars['c_ei']*E - pars['c_ii']*I + pars['Q'] - pars['theta_i']
    E_input = pars['c_e']/(1 + np.exp(-pars['a_e']* (pars['alpha_e'] * E_input - pars['b_e'])))
    I_input = pars['c_i']/(1 + np.exp(-pars['a_i']* (pars['alpha_i'] * I_input - pars['b_i'])))
    dE = (((pars['k_e'] - pars['r_e'] * E) * E_input) - E) / pars['tau_e'] 
    dI = (((pars['k_i'] - pars['r_i'] * I) * I_input) - I) / pars['tau_i'] 
    return E + dE*dt, I + dI*dt

pars = {
    'c_ee': 16.0, #Ex-to-Ex coupling coefficient, index = 0
    'c_ei': 12.0, #Ex-to-In coupling coefficient, index = 1
    'c_ie': 15.0, #In-to-Ex coupling coefficient, index = 2
    'c_ii': 3.0, #In-to-In coupling coefficient, index = 3
    'tau_e': 8.0, #Ex membrane time-constant, index = 4
    'tau_i': 18.0, #In membrane time-constant, index = 5
    'a_e': 1.3, #Ex Value of max slope of sigmoid function (1/a_e) is related to variance of distribution of thresholds, index = 6
    'b_e': 4.0, #Sigmoid function threshold, index = 7
    'c_e': 1.0, #Amplitude of Ex response function, index = 8
    'theta_e': 0.0, #Position of max slope of S_e, index = 9
    'a_i': 2.0, #In Value of max slope of sigmoid function (1/a_e) is related to variance of distribution of thresholds, index = 10
    'b_i': 3.7, #Sigmoid function threshold, index = 11
    'c_i': 1.0, #Amplitude of In response function, index = 12
    'theta_i': 0.0, #Position of max slope of S_i, index = 13
    'r_e': 1.0, #Ex refractory period, index = 14
    'r_i': 1.0, #In refractory period, index = 15
    'k_e': 1.0, #Max value of the Ex response function, index = 16
    'k_i': 1.0, #Max value of the In response function, index = 17
    'P': 1.25, #Balance between Ex and In masses, index = 18
    'Q': 0.0, #Balance between Ex and In masses, index = 19
    'alpha_e': 1.0, #Balance between Ex and In masses, index = 20
    'alpha_i': 1.0, #Balance between Ex and In masses, index = 21

}

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
