import os
import numpy as np
import matplotlib.pyplot as plt
import corner

from src.run_mcmc import (
    PARAM_NAMES,
    pack_params,
    model,
    PATH_BAO,
    PATH_SNE,
)
from src.likelihood_bao import loadlike_bao, bao_theory, BAO_OBSERVABLE
from src.likelihood_sne import loadlike_sne, sne_mu_theory


def _latex_label(name: str) -> str:
    """
    Traduce nombre interno -> label en LaTeX para las figuras.
    """
    mapping = {
        "Omega_m":     r"$\Omega_m$",
        "alpha0":      r"$\alpha_0$",
        "alpha1":      r"$\alpha_1$",
        "alpha_log":   r"$\alpha_{\log}$",
        "beta_psi":    r"$\beta_{\psi}$",
        "gamma_Omega": r"$\gamma_{\Omega}$",
        "xi_psi":      r"$\xi_{\psi}$",
    }
    return mapping.get(name, name)


# ---------------------------------------------------------
# CARGAR LA CADENA
# ---------------------------------------------------------

def load_chain(path="chains.npy", burnin_frac=0.5, thin=10):
    """
    Carga cadena MCMC y devuelve:
      - chain completa
      - chain_post: posterior con burn-in + thinning
    """
    chain = np.load(path)

    if chain.ndim != 3:
        raise ValueError(f"Shape inesperado de la cadena: {chain.shape}")

    nwalkers, nsteps, ndim = chain.shape
    burn = int(burnin_frac * nsteps)

    # Recortamos burn-in
    chain_post = chain[:, burn:, :]

    # Thinning
    chain_post = chain_post[:, ::thin, :]

    # Aplanar todas las muestras
    chain_post = chain_post.reshape(-1, ndim)

    return chain, chain_post


# ---------------------------------------------------------
# TRAZAS
# ---------------------------------------------------------

def make_trace_plots(chain, outdir="output"):
    os.makedirs(outdir, exist_ok=True)
    nwalkers, nsteps, ndim = chain.shape

    for i, name in enumerate(PARAM_NAMES):
        fig, ax = plt.subplots()
        for w in range(nwalkers):
            ax.plot(chain[w, :, i], alpha=0.2, lw=0.5)

        ax.set_xlabel("step")
        ax.set_ylabel(_latex_label(name))
        ax.set_title(f"Trace plot de {name}")

        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"trace_{name}.png"), dpi=150)
        plt.close(fig)


# ---------------------------------------------------------
# CORNER
# ---------------------------------------------------------

def make_corner_plot(chain_post, outdir="output"):
    os.makedirs(outdir, exist_ok=True)

    labels = [_latex_label(p) for p in PARAM_NAMES]

    fig = corner.corner(
        chain_post,
        labels=labels,
        show_titles=True,
        title_fmt=".4f",
        title_kwargs={"fontsize": 10},
    )

    fig.savefig(os.path.join(outdir, "corner.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------
# BAO PLOT
# ---------------------------------------------------------

def make_bao_plot(params_dict, outdir="output"):
    os.makedirs(outdir, exist_ok=True)

    bao_data = loadlike_bao(PATH_BAO)
    z = bao_data["z"]
    obs = bao_data["obs"]
    sigma = bao_data["sigma"]

    th = bao_theory(model, params_dict, z)

    fig, ax = plt.subplots()
    ax.errorbar(z, obs, yerr=sigma, fmt="o", ms=3, capsize=3, label="Datos BAO")
    ax.plot(
        z,
        th,
        "-",
        lw=1.5,
        label=r"CET$\Omega$ (mediana)",
    )

    # Label del eje Y según observable
    if BAO_OBSERVABLE == "DV_over_rd":
        ylabel = r"$D_V(z)/r_d$"
    elif BAO_OBSERVABLE == "DM_over_rd":
        ylabel = r"$D_M(z)/r_d$"
    elif BAO_OBSERVABLE == "DH_over_rd":
        ylabel = r"$D_H(z)/r_d$"
    else:
        ylabel = "BAO observable"

    ax.set_xlabel(r"$z$")
    ax.set_ylabel(ylabel)
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "bao_fit.png"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------
# SNE PLOT
# ---------------------------------------------------------

def make_sne_plot(params_dict, outdir="output"):
    os.makedirs(outdir, exist_ok=True)

    sne_data = loadlike_sne(PATH_SNE)
    z = sne_data["z"]
    mu_obs = sne_data["mu"]
    sigma = sne_data["sigma"]

    # Curva suave
    z_grid = np.linspace(0.0, z.max() * 1.05, 300)
    mu_th_grid = sne_mu_theory(model, params_dict, z_grid)

    fig, ax = plt.subplots()
    ax.errorbar(z, mu_obs, yerr=sigma, fmt="o", ms=3, capsize=3, label="SNe datos")
    ax.plot(
        z_grid,
        mu_th_grid,
        "-",
        lw=1.5,
        label=r"CET$\Omega$ (mediana)",
    )

    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$\mu(z)$")
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "sne_fit.png"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    outdir = "output"
    chain, chain_post = load_chain("chains.npy", burnin_frac=0.5, thin=10)

    # Parámetros CETΩ = mediana del posterior
    theta_med = np.median(chain_post, axis=0)
    params_med = pack_params(theta_med)

    # 1) Trazas
    make_trace_plots(chain, outdir=outdir)

    # 2) Corner
    make_corner_plot(chain_post, outdir=outdir)

    # 3) BAO fit
    make_bao_plot(params_med, outdir=outdir)

    # 4) SNe fit
    make_sne_plot(params_med, outdir=outdir)

    print("[make_figures] Figuras guardadas en carpeta 'output/'.")


if __name__ == "__main__":
    main()