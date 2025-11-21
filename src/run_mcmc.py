import numpy as np
import emcee

from src.ceto_model import CETOmegaModel
from src.likelihood_cmb import loadlike_cmb, loglike_cmb
from src.likelihood_bao import loadlike_bao, loglike_bao
from src.likelihood_sne import loadlike_sne, loglike_sne
from src.likelihood_hz import loadlike_hz, loglike_hz

# ---------------------------------------------------------
# RUTAS DE DATOS
# ---------------------------------------------------------

PATH_CMB = "data/theta_star_ACTPlanck2024.txt"
PATH_BAO = "data/desi_dr2_bao.txt"
PATH_SNE = "data/union30_sne.txt"
PATH_HZ  = "data/H_z.txt"   # si no existe, se ignora H(z)


# ---------------------------------------------------------
# PARÁMETROS CETΩ
# ---------------------------------------------------------

PARAM_NAMES = [
    "Omega_m",
    "alpha0",
    "alpha1",
    "alpha_log",
    "beta_psi",
    "gamma_Omega",
    "xi_psi",
]


def pack_params(theta):
    """
    Convierte el vector theta en dict de parámetros para CETOmegaModel
    y los likelihoods.
    """
    return {
        "Omega_m":     float(theta[0]),
        "alpha0":      float(theta[1]),
        "alpha1":      float(theta[2]),
        "alpha_log":   float(theta[3]),
        "beta_psi":    float(theta[4]),
        "gamma_Omega": float(theta[5]),
        "xi_psi":      float(theta[6]),
    }


# ---------------------------------------------------------
# MODELO CETΩ
# ---------------------------------------------------------

model = CETOmegaModel()


# ---------------------------------------------------------
# PRIORS
# ---------------------------------------------------------

def logprior(theta):
    """
    Priors uniformes (top-hat) razonables para un primer análisis.
    Ajustables después.
    """
    (Omega_m,
     alpha0,
     alpha1,
     alpha_log,
     beta_psi,
     gamma_Omega,
     xi_psi) = theta

    if not (0.05 < Omega_m < 0.6):
        return -np.inf

    if not (-5.0 < alpha0 < 5.0):
        return -np.inf

    if not (-5.0 < alpha1 < 5.0):
        return -np.inf

    if not (-5.0 < alpha_log < 5.0):
        return -np.inf

    if not (-5.0 < beta_psi < 5.0):
        return -np.inf

    if not (-5.0 < gamma_Omega < 5.0):
        return -np.inf

    if not (-5.0 < xi_psi < 5.0):
        return -np.inf

    return 0.0


# ---------------------------------------------------------
# CARGA DE DATOS
# ---------------------------------------------------------

theta_cmb_obs, sigma_cmb = loadlike_cmb(PATH_CMB)
bao_data = loadlike_bao(PATH_BAO)
sne_data = loadlike_sne(PATH_SNE)

# H(z) es opcional: si falta el archivo, seguimos sin rompernos
USE_HZ = True
try:
    hz_data = loadlike_hz(PATH_HZ)
except FileNotFoundError:
    print(f"[run_mcmc] Aviso: no se encontró {PATH_HZ}, se omite likelihood H(z).")
    USE_HZ = False


# ---------------------------------------------------------
# LOG-POSTERIOR
# ---------------------------------------------------------

def log_prob(theta):
    """
    log posterior = log prior + sum log-likelihoods
    """
    lp = logprior(theta)
    if not np.isfinite(lp):
        return -np.inf

    params = pack_params(theta)

    ll_cmb = loglike_cmb(params, model, theta_cmb_obs, sigma_cmb)
    ll_bao = loglike_bao(params, model, bao_data)
    ll_sne = loglike_sne(params, model, sne_data)

    logL = ll_cmb + ll_bao + ll_sne

    if USE_HZ:
        logL += loglike_hz(params, model, hz_data)

    return lp + logL


# ---------------------------------------------------------
# MCMC
# ---------------------------------------------------------

def run_sampler(
    nwalkers=32,
    nsteps=8000,
    outfile="chains.npy",
    seed=42,
):
    """
    Ejecuta el MCMC con emcee y guarda:
    - cadena completa en `outfile`
    - log-probabilidades en `logprob.npy`
    """
    ndim = len(PARAM_NAMES)
    rng = np.random.default_rng(seed)

    # Punto inicial razonable
    theta0 = np.array([
        0.3,   # Omega_m
        1.0,   # alpha0
        0.5,   # alpha1
        0.0,   # alpha_log
        0.0,   # beta_psi
        0.0,   # gamma_Omega
        0.0,   # xi_psi
    ])

    # Desplazamos walkers alrededor de theta0
    p0 = theta0 + 1e-2 * rng.standard_normal(size=(nwalkers, ndim))

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)

    print(f"[run_mcmc] Iniciando MCMC: nwalkers={nwalkers}, nsteps={nsteps}, ndim={ndim}")
    sampler.run_mcmc(p0, nsteps, progress=True)

    chain = sampler.get_chain()
    logp_chain = sampler.get_log_prob()

    np.save(outfile, chain)
    np.save("logprob.npy", logp_chain)

    print(f"[run_mcmc] Cadena guardada en '{outfile}', log-prob en 'logprob.npy'.")

    # (Opcional) intentar estimar el tiempo de autocorrelación
    try:
        tau = sampler.get_autocorr_time()
        print("[run_mcmc] Autocorrelation time tau ~", tau)
        print("[run_mcmc] nsteps / tau ~", nsteps / np.mean(tau))
    except Exception as e:
        print("[run_mcmc] No se pudo estimar tau:", e)

    return chain, logp_chain


def main():
    run_sampler()


if __name__ == "__main__":
    main()