"""The five solvers compared in the project."""

from svmdual.solvers.cvx import cvx_dual, cvx_primal
from svmdual.solvers.l1svm import l1_svm
from svmdual.solvers.pegasos import PegasosConfig, pegasos
from svmdual.solvers.smo import SMOConfig, smo

__all__ = ["PegasosConfig", "SMOConfig", "cvx_dual", "cvx_primal", "l1_svm", "pegasos", "smo"]
