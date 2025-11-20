# CETΩmega Cosmology
A Reproducible Multi-Probe Validation of the CETΩ Informational Cosmological Model

This repository contains the complete reproducible pipeline for the
multi-probe empirical validation of the **CETΩ (Causal-Entropic Thermodynamics Ω)**
informational cosmology model.
It includes:

- The full cosmological model implementation (`src/ceto_model.py`)
- MCMC sampling and parameter inference (`src/run_mcmc.py`)
- Automatic figure generation (`src/make_figures.py`)
- Posterior summary tools (`src/summarize_results.py`)
- DESI BAO, cosmic chronometers, and CMB acoustic-scale consistency tests


The project is designed as a fully open, transparent, and reproducible scientific workflow.

---

## 📂 Repository Structure

---

## 🔧 Installation

### Clone the repository
```bash
git clone https://github.com/tensorflowcausal/CETOmega-cosmology.git
cd CETOmega-cosmology

Install dependencies


pip install -r requirements.txt

Running the Full CETΩ Pipeline



1. Run the MCMC sampler



This generates posterior samples for all CETΩ cosmological parameters.

python -m src.run_mcmc

Outputs (saved in /output/):

Chains (chains.npy)

Log-likelihood (logprob.npy)

Trace plots

Acceptance ratios

Generate all scientific figures

python -m src.make_figures
Figures saved in /figures/, including:

H(z) vs cosmic chronometers

Radial BAO predictions vs DESI DR2

Sound horizon integrals

Posterior corner plots

Trace plots

3. Summarize posterior distributions

python -m src.summarize_results
This prints and saves:

Median parameter values

1σ credible intervals

Best-fit point estimates



Saved as:

output/params_summary.txt

Observables Computed



The CETΩ pipeline computes:

Expansion rate: H(z) = H₀ E(z)

Comoving distance: D₍C₎(z)

Angular-diameter distance: D₍A₎(z)

Sound horizon at recombination rₛ(z*)

Drag-epoch sound horizon r_d = rₛ(z_d)

BAO observables:

D_H(z)/r_d

D_M(z)/r_d

CMB acoustic angle:

θ* = rₛ(z*) / D_A(z*)



Fully consistent with:

DESI DR2 BAO

Cosmic chronometers

Planck 2018 acoustic scale

About the CETΩ Cosmology



CETΩ is an informational extension of standard cosmology.

Its key idea is that informational pressure acts as a small correction to the

Friedmann equation:



[

H^2(z) \to H^2(z),D_\Omega(z)

]



This project implements the complete CETΩ framework as described in:

General CETΩ Theory

DOI: https://doi.org/10.5281/zenodo.17464238

Empirical BAO Validation

DOI: https://doi.org/10.5281/zenodo.17653205

