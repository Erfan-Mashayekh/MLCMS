# parameters
t_0     = 0
t_end   = 1000
steps = 15000
# if these error tolerances are set too high, the solution will be qualitatively (!) wrong
rtol = 1e-8
atol = 1e-8

# SIR model parameters
beta= 11.5  # Average number of adequate contacts per unit time with infectious individuals
A   = 20    # birth rate
d   = 0.1   # per capita natural death rate
nu  = 1     # per capita disease-induced death rate
b   = 0.01  # try to set this to 0.01, 0.020, ..., 0.022, ..., 0.03
mu0 = 10    # minimum recovery rate
mu1 = 10.45 # maximum recovery rate