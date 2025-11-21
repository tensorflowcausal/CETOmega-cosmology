import numpy as np


def loadlike_hz(path):
    """
    Carga datos H(z) (cosmic chronometers).

    Formato esperado:
        z   H_obs   sigma
    """
    data = np.loadtxt(path)
    return {
        "z": data[:, 0],
        "H": data[:, 1],
        "sigma": data[:, 2],
    }


def loglike_hz(params, model, hz_data):
    """
    Likelihood para H(z).

        ln L = -0.5 * Σ [ (H_obs - H_th)^2 / sigma^2 ]
    """
    z = hz_data["z"]
    Hobs = hz_data["H"]
    sigma = hz_data["sigma"]

    Hth = model.H_of_z(z, params)

    chi2 = np.sum(((Hobs - Hth) / sigma)**2)
    return -0.5 * chi2