import numpy as np

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

parameters_alpha_peak = np.array([1.6000e+01, 1.2000e+01, 1.5000e+01, 3.0000e+00, 28.0000e+00, 28.0000e+00,
        1.3000e+00, 4.0000e+00, 1.0000e+00, 0.0000e+00, 2.0000e+00, 3.7000e+00,
        1.0000e+00, 0.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00,
        1.500e+00, 0.0000e+00, 1.0000e+00, 1.0000e+00,5.0000e-07,5.0000e-07])

parameters_original = np.array([1.6000e+01, 1.2000e+01, 1.5000e+01, 3.0000e+00, 8.0000e+00, 8.0000e+00,
        1.3000e+00, 4.0000e+00, 1.0000e+00, 0.0000e+00, 2.0000e+00, 3.7000e+00,
        1.0000e+00, 0.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00,
        1.500e+00, 0.0000e+00, 1.0000e+00, 1.0000e+00,5.0000e-07,5.0000e-07])

parameters_lower_bound = np.array([
    1.1000e+01, #c_ee
    0.2000e+01, #c_ei
    0.2000e+01, #c_ie
    2.0000e+00, #c_ii
    1.0000e+00, #tau_e
    1.0000e+00, #tau_i
    0.0000e+00, #a_e
    1.4000e+00, #b_e
    1.0000e+00, #c_e
    0.0000e+00, #theta_e
    0.0000e+00, #a_i
    2.0000e+00, #b_i
    1.0000e+00, #c_i
    0.0000e+00, #theta_i
    0.5000e+00, #r_e
    0.5000e+00, #r_i
    0.5000e+00, #k_e
    0.0000e+00, #k_i
    0.0000e+00, #P
    0.0000e+00, #Q
    0.0000e+00, #alpha_e
    0.0000e+00, #alpha_i
    5.0000e-10, #Noise E
    5.0000e-10  #Noise I 
    ])


parameters_upper_bound = np.array([
    1.6000e+01, #c_ee
    1.5000e+01, #c_ei
    1.3000e+01, #c_ie
    1.1000e+01, #c_ii
    1.5000e+02, #tau_e
    1.5000e+02, #tau_i
    1.4000e+00, #a_e
    6.0000e+00, #b_e
    2.0000e+01, #c_e
    6.0000e+01, #theta_e
    2.0000e+00, #a_i
    6.0000e+00, #b_i
    2.0000e+01, #c_i
    6.0000e+01, #theta_i
    2.0000e+00, #r_e
    2.0000e+00, #r_i
    2.0000e+00, #k_e
    2.0000e+00, #k_i
    2.0000e+01, #P
    2.0000e+01, #Q
    2.0000e+01, #alpha_e
    2.0000e+01, #alpha_i
    5.0000e-04, #Noise E
    5.0000e-04  #Noise I 
    ])

parameters_range_bounds = parameters_upper_bound-parameters_lower_bound

def sample_uniform_within_range(num_samples, upper = parameters_upper_bound, lower = parameters_lower_bound):
    bounds = upper - lower
    samples = np.random.uniform(size = (num_samples, bounds.shape[0])) * bounds
    return samples + lower

def sample_uniform_around(num_samples, point = parameters_alpha_peak, width = parameters_range_bounds * 0.01):
    bound_up = point + width/2
    bound_up = np.where(bound_up>parameters_upper_bound,parameters_upper_bound,bound_up)
    bound_down = point - width/2
    bound_down = np.where(bound_down<parameters_lower_bound,parameters_lower_bound,bound_down)
    bounds = bound_up - bound_down
    samples = np.random.uniform(size = (num_samples, point.shape[0])) * bounds
    return samples + bound_down