import numpy as np
import warnings
from scipy.integrate import quad, IntegrationWarning

c = 299792.458  # km/s


class CETOmegaModel:
    """
    CETΩ background cosmology:
    - Includes matter + radiation
    - Informational deformation with alpha0, alpha1, alpha_log, gamma_Omega, xi_psi
    - Computes: H(z), DM(z), DA(z), rs(z), z_star, theta_star
    """

    def __init__(self, H0=70.0, Omega_b=0.049, Tcmb=2.7255):
        self.H0 = H0
        self.Omega_b = Omega_b
        self.Tcmb = Tcmb

        # Radiation density today (photons + neutrinos)
        Omega_gamma = 2.47e-5 / (H0 / 100.0) ** 2
        self.Omega_r = Omega_gamma * (1 + 0.2271 * 3.046)

    # ---------------------------------------------------------
    # INTERNAL ROBUST INTEGRATORS
    # ---------------------------------------------------------

    def _integral_0_to_z(
        self,
        integrand,
        z,
        z_split=1e-4,
        epsabs=1e-7,
        epsrel=1e-7,
        limit=200,
    ):
        """
        Robust integral of integrand(zp) from 0 to z.

        - Handles z < 0 by flipping the sign.
        - Avoids potential numerical issues near zp = 0 by splitting [0,z]
          into [0,z_split] and [z_split,z] when z > z_split.
        - Captures IntegrationWarning and returns +inf in that case so that
          the likelihood can simply reject those parameter points.
        """
        if z == 0:
            return 0.0

        z = float(z)
        sgn = 1.0
        if z < 0:
            z = -z
            sgn = -1.0

        # Choose a small sub-interval near zero (but not larger than ~30% of z)
        z0 = min(z_split, 0.3 * z)

        def safe_f(u):
            val = integrand(u)
            # Protect against NaNs/Infs in the integrand
            if not np.isfinite(val):
                return 0.0
            return val

        with warnings.catch_warnings():
            warnings.simplefilter("error", IntegrationWarning)
            try:
                if z <= z0 or z0 == 0.0:
                    val, _ = quad(
                        safe_f,
                        0.0,
                        z,
                        epsabs=epsabs,
                        epsrel=epsrel,
                        limit=limit,
                    )
                else:
                    val1, _ = quad(
                        safe_f,
                        0.0,
                        z0,
                        epsabs=epsabs,
                        epsrel=epsrel,
                        limit=limit,
                    )
                    val2, _ = quad(
                        safe_f,
                        z0,
                        z,
                        epsabs=epsabs,
                        epsrel=epsrel,
                        limit=limit,
                    )
                    val = val1 + val2
            except IntegrationWarning:
                return np.inf

        return sgn * val

    def _safe_quad(
        self,
        integrand,
        a,
        b,
        epsabs=1e-7,
        epsrel=1e-7,
        limit=300,
    ):
        """
        Generic robust wrapper around scipy.integrate.quad on [a,b].
        Captures IntegrationWarning and returns +inf in that case.
        """
        if a == b:
            return 0.0

        a = float(a)
        b = float(b)
        sgn = 1.0
        if b < a:
            a, b = b, a
            sgn = -1.0

        def safe_f(u):
            val = integrand(u)
            if not np.isfinite(val):
                return 0.0
            return val

        with warnings.catch_warnings():
            warnings.simplefilter("error", IntegrationWarning)
            try:
                val, _ = quad(
                    safe_f,
                    a,
                    b,
                    epsabs=epsabs,
                    epsrel=epsrel,
                    limit=limit,
                )
            except IntegrationWarning:
                return np.inf

        return sgn * val

    # ---------------------------------------------------------
    # PARAM PARSING
    # ---------------------------------------------------------

    def _parse_params(self, params):
        return (
            params["Omega_m"],
            params["alpha0"],
            params["alpha1"],
            params.get("alpha_log", 0.0),
            params.get("beta_psi", 0.0),
            params.get("gamma_Omega", 0.0),
            params.get("xi_psi", 0.0),
        )

    # ---------------------------------------------------------
    # EXPANSION
    # ---------------------------------------------------------

    def dark_deform(
        self,
        z,
        alpha0,
        alpha1,
        alpha_log,
        beta_psi,
        gamma_Omega,
        xi_psi,
    ):
        zp1 = 1 + z
        return (
            1
            + alpha0 * z / zp1
            + alpha1 * z**2 / (zp1**2)
            + alpha_log * np.log(zp1)
            + gamma_Omega * z
            + xi_psi * z**2
        )

    def E_of_z(self, z, params):
        (
            Omega_m,
            a0,
            a1,
            aL,
            bp,
            gO,
            xpsi,
        ) = self._parse_params(params)

        z = np.asarray(z)
        deform = self.dark_deform(z, a0, a1, aL, bp, gO, xpsi)

        Ez2 = (
            self.Omega_r * (1 + z) ** 4
            + Omega_m * (1 + z) ** 3
            + (1 - Omega_m - self.Omega_r) * deform
        )

        # Guard against negative numerical noise
        return np.sqrt(np.maximum(Ez2, 1e-15))

    def H_of_z(self, z, params):
        return self.H0 * self.E_of_z(z, params)

    # ---------------------------------------------------------
    # DISTANCES
    # ---------------------------------------------------------

    def Dc_of_z(self, z, params):
        integrand = lambda zp: 1.0 / self.E_of_z(zp, params)
        val = self._integral_0_to_z(integrand, z)
        return (c / self.H0) * val

    def DM_of_z(self, z, params):
        return self.Dc_of_z(z, params)

    def DA_of_z(self, z, params):
        return self.DM_of_z(z, params) / (1 + z)

    # ---------------------------------------------------------
    # EARLY UNIVERSE
    # ---------------------------------------------------------

    def cs_of_z(self, z, params):
        R = 3 * self.Omega_b / (4 * self.Omega_r) / (1 + z)
        return c / np.sqrt(3 * (1 + R))

    def rs_of_z(self, z, params):
        """
        Sound horizon from z to very high redshift.
        We keep the upper limit at 1e6 as in the original CETΩ setup,
        but with a robust integrator wrapper.
        """
        integrand = lambda zp: self.cs_of_z(zp, params) / self.H_of_z(zp, params)
        val = self._safe_quad(integrand, z, 1e6, epsabs=1e-7, epsrel=1e-7, limit=500)
        return val

    def z_star(self, params):
        """Recombination redshift (Hu & Sugiyama fit)."""
        Omega_m, _, _, _, _, _, _ = self._parse_params(params)
        h = self.H0 / 100
        Obh2 = self.Omega_b * h**2
        Omh2 = Omega_m * h**2

        g1 = 0.0783 * Obh2 ** -0.238 / (1 + 39.5 * Obh2**0.763)
        g2 = 0.560 / (1 + 21.1 * Obh2**1.81)

        return 1048 * (1 + 0.00124 * Obh2 ** -0.738) * (1 + g1 * Omh2**g2)

    def theta_star(self, params):
        zstar = self.z_star(params)
        rs = self.rs_of_z(zstar, params)
        DA = self.DA_of_z(zstar, params)
        return rs / DA