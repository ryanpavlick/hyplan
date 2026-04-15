"""QA bit-unpacking and filtering for MODIS vegetation products."""

from __future__ import annotations

import numpy as np


def apply_vi_qa_mask(
    data: np.ndarray,
    pixel_reliability: np.ndarray,
    max_reliability: int = 1,
) -> np.ma.MaskedArray:
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
    return np.ma.masked_array(data, mask=bad)


def apply_lai_qa_mask(
    data: np.ndarray,
    qa: np.ndarray,
) -> np.ma.MaskedArray:
    """Apply QA filter for MOD15A2H LAI/FPAR.

    Uses the FparLai_QC bitfield:

    * Bit 0: ``0`` = good quality main algorithm, ``1`` = other
    * Bits 5-7: cloud state (``000`` = clear)

    Also masks fill values (``255``).

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
    # Bit 0: algorithm quality
    algo_bad = (qa & 0b1) != 0
    # Bits 5-7: cloud state (must be 000 = clear)
    cloud_bits = (qa >> 5) & 0b111
    cloudy = cloud_bits != 0
    # Fill value
    fill = data == 255

    bad = algo_bad | cloudy | fill
    return np.ma.masked_array(data, mask=bad)


def apply_phenology_qa_mask(
    data_dict: dict[str, np.ndarray],
    qa: np.ndarray,
    max_quality: int = 1,
) -> dict[str, np.ma.MaskedArray]:
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
        name: np.ma.masked_array(arr, mask=bad)
        for name, arr in data_dict.items()
    }


def convert_mcd12q2_dates(raw_values: np.ndarray) -> np.ndarray:
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

    return result  # type: ignore[no-any-return]
