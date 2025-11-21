import numpy as np
from src.ceto_model import CETOmegaModel


def loadlike_sne(path):
    """
    Carga datos de supernovas.

    Formato mínimo esperado:
        z   mu_obs   sigma

    Devuelve:
        {"z": z, "mu": mu, "sigma": sigma}
    """
    data = np.loadtxt(path)
    data = np.atleast_2d(data)

    if data.shape[1] < 3:
        raise ValueError(
            f"Se esperaban al menos 3 columnas en {path} (z, mu, sigma). "
            f"Shape={data.shape}"
        )

    z = data[:, 0]
    mu = data[:, 1]
    sigma = data[:, 2]

    return {"z": z, "mu": mu, "sigma": sigma}


def _DM_of_z_vec(model, params, z):
    """
    Helper vectorizado porque DM_of_z integra escalar.
    """
    z = np.atleast_1d(z)
    DM = np.array([model.DM_of_z(zi, params) for zi in z])
    return DM


def sne_mu_theory(model, params, z):
    """
    Módulo de distancia cosmológico estándar:

        μ(z) = 5 log10(d_L / Mpc) + 25

    con d_L(z) = (1+z) * D_M(z).

    Se protege contra d_L = 0 usando un mínimo ~0.
    """
    z = np.atleast_1d(z)
    DM = _DM_of_z_vec(model, params, z)   # Mpc
    dL = (1.0 + z) * DM                   # Mpc

    # Evitar log10(0): clamp a un valor muy pequeño
    dL_safe = np.maximum(dL, 1e-6)

    mu_th = 5.0 * np.log10(dL_safe) + 25.0
    return mu_th


def loglike_sne(params, model, sne_data):
    """
    Log-verosimilitud SNe:

        ln L = -0.5 * sum_i [ (mu_obs_i - mu_th_i)/sigma_i ]^2
    """
    z = sne_data["z"]
    mu_obs = sne_data["mu"]
    sigma = sne_data["sigma"]

    mu_th = sne_mu_theory(model, params, z)

    chi2 = np.sum(((mu_obs - mu_th) / sigma) ** 2)
    return -0.5 * chi2