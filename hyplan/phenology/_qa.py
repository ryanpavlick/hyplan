"""QA bit-unpacking and filtering for MODIS vegetation products."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


def apply_vi_qa_mask(
    data: npt.NDArray[Any],
    pixel_reliability: npt.NDArray[np.integer[Any]],
    max_reliability: int = 1,
) -> np.ma.MaskedArray[Any, np.dtype[Any]]:
    """Apply QA filter for MOD13A1/MYD13A1 vegetation indices.

    MODIS pixel reliability band values::

        0 = Good data
        1 = Marginal data
        2 = Snow/Ice
        3 = Cloudy
       -1 = Fill/No data (255 unsigned)

    Parameters
    ----------
    data : np.ndarray
        Raw vegetation index data array (int16, pre-scale-factor).
    pixel_reliability : np.ndarray
        Pixel reliability band values (same shape as *data*).
    max_reliability : int
        Maximum acceptable reliability value.  ``0`` keeps only good
        pixels; ``1`` (default) keeps good and marginal.

    Returns
    -------
    np.ma.MaskedArray
        Data with unreliable pixels masked.
    """
    bad = (pixel_reliability > max_reliability) | (pixel_reliability < 0)
    return np.ma.masked_array(data, mask=bad)  # type: ignore[no-untyped-call]


def apply_lai_qa_mask(
    data: npt.NDArray[Any],
    qa: npt.NDArray[np.integer[Any]],
) -> np.ma.MaskedArray[Any, np.dtype[Any]]:
    """Apply QA filter for MOD15A2H LAI/FPAR.

    Uses the FparLai_QC bitfield (per the MOD15 C6.1 User's Guide):

    * Bit 0 — MODLAND_QC: ``0`` = good quality (main RT algorithm),
      ``1`` = other quality.
    * Bits 3-4 — CloudState (``00`` = clear).  **Not** filtered directly
      here; see below.
    * Bits 5-7 — SCF_QC: the retrieval algorithm-path / confidence score,
      ``000`` = main (RT) method used, best result possible, no
      saturation; ``001`` = main method with saturation; ``010``/``011``
      = main method failed, empirical fallback used; ``100`` = pixel not
      produced.

    This filter keeps a pixel only when ``MODLAND_QC == 0`` **and**
    ``SCF_QC == 000`` — i.e. the highest-confidence main-algorithm
    retrieval with no saturation.  That is intentionally stricter than a
    cloud-only screen: it also rejects saturated and empirical-fallback
    retrievals.  Because the SCF_QC == 000 class already implies a
    successful clear-sky main retrieval, CloudState (bits 3-4) is not
    tested separately.  Fill values (``255``) are also masked.

    Parameters
    ----------
    data : np.ndarray
        Raw LAI or FPAR data array (uint8, pre-scale-factor).
    qa : np.ndarray
        FparLai_QC band values (same shape as *data*).

    Returns
    -------
    np.ma.MaskedArray
        Data with low-quality pixels masked.
    """
    # Bit 0 — MODLAND_QC: 0 = good (main algorithm), 1 = other.
    algo_bad = (qa & 0b1) != 0
    # Bits 5-7 — SCF_QC: keep only 000 (best main-RT retrieval, no
    # saturation); any other value is rejected.
    scf_qc = (qa >> 5) & 0b111
    not_best_retrieval = scf_qc != 0
    # Fill value
    fill = data == 255

    bad = algo_bad | not_best_retrieval | fill
    return np.ma.masked_array(data, mask=bad)  # type: ignore[no-untyped-call]


def apply_phenology_qa_mask(
    data_dict: dict[str, npt.NDArray[Any]],
    qa: npt.NDArray[np.integer[Any]],
    max_quality: int = 1,
) -> dict[str, np.ma.MaskedArray[Any, np.dtype[Any]]]:
    """Apply QA filter for MCD12Q2 phenology transitions.

    Uses the QA_Detailed bitfield:

    * Bits 0-1: overall quality (``00`` = best, ``01`` = good,
      ``10`` = fair, ``11`` = poor)

    Parameters
    ----------
    data_dict : dict[str, np.ndarray]
        Mapping of stage name to raw date-value arrays.
    qa : np.ndarray
        QA_Detailed band values.
    max_quality : int
        Maximum acceptable quality code.  ``0`` keeps only best;
        ``1`` (default) keeps best and good.

    Returns
    -------
    dict[str, np.ma.MaskedArray]
        Same keys as *data_dict*, with low-quality pixels masked.
    """
    quality_bits = qa & 0b11
    bad = quality_bits > max_quality

    return {
        name: np.ma.masked_array(arr, mask=bad)  # type: ignore[no-untyped-call]
        for name, arr in data_dict.items()
    }


def convert_mcd12q2_dates(raw_values: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    """Convert MCD12Q2 date values to day-of-year.

    MCD12Q2 stores phenological transition dates as the number of
    days since January 1, 1970.  Fill values (``32767``) become ``NaN``.

    Parameters
    ----------
    raw_values : np.ndarray
        Integer array of days since 1970-01-01.

    Returns
    -------
    np.ndarray
        Float array of day-of-year values (1–366), with ``NaN``
        for fill/invalid entries.
    """
    import datetime as dt

    epoch = dt.date(1970, 1, 1)
    result = np.full(raw_values.shape, np.nan, dtype=np.float64)

    valid = (raw_values != 32767) & (raw_values > 0)
    if np.any(valid):
        days = raw_values[valid].astype(int)
        dates = np.array([epoch + dt.timedelta(days=int(d)) for d in days])
        doys = np.array([d.timetuple().tm_yday for d in dates], dtype=np.float64)
        result[valid] = doys

    return result
