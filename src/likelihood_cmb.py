import numpy as np

def loadlike_cmb(path):
    """
    Lee archivo con dos números:
    theta_obs   sigma_theta
    """
    arr = np.loadtxt(path)
    theta_obs = float(arr[0])
    sigma_theta = float(arr[1])
    return theta_obs, sigma_theta


def loglike_cmb(params, model, theta_obs, sigma_theta):
    """
    Likelihood para el CMB acoustic scale usando CETΩ:
    
        theta* = r_s(z*) / D_A(z*)

    donde:
        - z* se calcula con el fit de Hu & Sugiyama
        - r_s se integra con H(z) CETΩ
        - D_A se calcula también con CETΩ
    """
    theta_th = model.theta_star(params)
    chi2 = (theta_obs - theta_th)**2 / sigma_theta**2
    return -0.5 * chi2