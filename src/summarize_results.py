import numpy as np
from src.run_mcmc import PARAM_NAMES, pack_params


def summarize_chain(chain_path="chains.npy", burnin_frac=0.5, thin=10, outfile="params_summary.txt"):
    chain = np.load(chain_path)   # (nwalkers, nsteps, ndim)

    if chain.ndim != 3:
        raise ValueError(f"Cadena con shape raro: {chain.shape}")

    nwalkers, nsteps, ndim = chain.shape
    burn = int(burnin_frac * nsteps)

    # Recortar burn-in
    chain_post = chain[:, burn:, :]

    # Thinning
    chain_post = chain_post[:, ::thin, :]

    # Aplanar
    samples = chain_post.reshape(-1, ndim)

    results = {}
    lines = []

    for i, name in enumerate(PARAM_NAMES):
        vals = samples[:, i]

        mean = np.mean(vals)
        median = np.median(vals)
        p16 = np.percentile(vals, 16)
        p84 = np.percentile(vals, 84)

        err_minus = median - p16
        err_plus  = p84 - median

        results[name] = {
            "mean": mean,
            "median": median,
            "p16": p16,
            "p84": p84,
            "err_minus": err_minus,
            "err_plus": err_plus,
        }

        line = (
            f"{name:12s}  median = {median: .5f}  "
            f"-{err_minus: .5f}  +{err_plus: .5f}  "
            f"(mean = {mean: .5f})"
        )
        lines.append(line)
        print(line)

    # Guardar tabla
    with open(outfile, "w") as f:
        for L in lines:
            f.write(L + "\n")

    print(f"[summarize_results] Resumen guardado en '{outfile}'.")

    return results


def main():
    summarize_chain()


if __name__ == "__main__":
    main()