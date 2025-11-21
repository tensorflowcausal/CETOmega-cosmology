import numpy as np
from src.ceto_model import c  # velocidad de la luz (km/s) definida en ceto_model

# Elegí uno: "DM_over_rd", "DH_over_rd", "DV_over_rd"
BAO_OBSERVABLE = "DV_over_rd"


def loadlike_bao(path):
    """
    Carga datos BAO en formato:

        z   observable   sigma

    Ejemplos de 'observable' según BAO_OBSERVABLE:
        - DV_over_rd  -> D_V(z)/r_d
        - DM_over_rd  -> D_M(z)/r_d
        - DH_over_rd  -> D_H(z)/r_d

    Devuelve:
        {
          "z": z,
          "obs": obs,
          "sigma": sigma
        }
    """
    data = np.loadtxt(path)
    data = np.atleast_2d(data)

    if data.shape[1] < 3:
        raise ValueError(
            f"Se esperaban al menos 3 columnas en {path} (z, obs, sigma). "
            f"Shape={data.shape}"
        )

    z = data[:, 0]
    obs = data[:, 1]
    sigma = data[:, 2]

    return {"z": z, "obs": obs, "sigma": sigma}


def bao_theory(model, params, z, rd=147.0):
    """
    Calcula el observable BAO teórico en CETΩ.

    Por ahora usamos un r_d fijo (147 Mpc) estándar Planck.
    Más adelante se puede reemplazar por r_s^Ω del modelo.
    """
    z = np.atleast_1d(z)

    # H(z) vectorial
    H = model.H_of_z(z, params)          # km/s/Mpc

    # D_M(z) requiere integración escalar → vectorizamos a mano
    DM = np.array([model.DM_of_z(zi, params) for zi in z])  # Mpc

    if BAO_OBSERVABLE == "DM_over_rd":
        return DM / rd

    elif BAO_OBSERVABLE == "DH_over_rd":
        DH = c / H                       # Mpc
        return DH / rd

    elif BAO_OBSERVABLE == "DV_over_rd":
        # D_V = [ z * D_M^2 * D_H ]^(1/3)
        DH = c / H                       # Mpc
        DV = (z * DM**2 * DH) ** (1.0 / 3.0)
        return DV / rd

    else:
        raise ValueError(
            f"Observable BAO desconocido: {BAO_OBSERVABLE}. "
            f"Usa 'DV_over_rd', 'DM_over_rd' o 'DH_over_rd'."
        )


def loglike_bao(params, model, bao_data, rd=147.0):
    """
    Log-verosimilitud BAO:

        ln L = -0.5 * sum_i [ (obs_i - th_i) / sigma_i ]^2
    """
    z = bao_data["z"]
    obs = bao_data["obs"]
    sigma = bao_data["sigma"]

    th = bao_theory(model, params, z, rd=rd)

    chi2 = np.sum(((obs - th) / sigma) ** 2)
    return -0.5 * chi2