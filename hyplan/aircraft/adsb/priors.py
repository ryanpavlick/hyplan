"""Optional prior/reference model comparison (v1 stubs).

Future versions will support blending fitted schedules with prior
aircraft models (e.g. from OpenAP or manufacturer data) and scoring
fit quality against a reference.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from .models import FitResult
    from .._base import Aircraft

logger = logging.getLogger(__name__)


def apply_prior(
    fit_result: FitResult,
    prior_aircraft: Optional[Aircraft] = None,
    blend_weight: float = 0.3,
) -> FitResult:
    """Blend fitted schedules with a prior aircraft model.

    Not yet implemented in v1.  Returns *fit_result* unchanged.
    """
    logger.info("apply_prior: no-op in v1, returning fit_result unchanged.")
    return fit_result


def score_fit(
    fit_result: FitResult,
    reference_aircraft: Optional[Aircraft] = None,
) -> Dict[str, float]:
    """Score a fit result against a reference aircraft.

    Not yet implemented in v1.  Returns an empty dict.
    """
    logger.info("score_fit: not yet implemented, returning empty scores.")
    return {}
