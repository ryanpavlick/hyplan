# Planned ↔ as-flown sortie pairs

Mapping between NASA ER-2 mission planning artifacts (KML route +
"Green Card" mission data card) and the matching IWG1 in-situ recording.
The IWG1 timestamps below are the start of the airborne window
(post `trim_ground_taxi`, second header line of the raw file).

## In scope

All three pairs are fully supported by the parser
([`hyplan/aircraft/_planned_sortie.py`](../../hyplan/aircraft/_planned_sortie.py));
PDF Green Cards parse via `pdfplumber` (install with
`pip install hyplan[planned]`), XLSX via `openpyxl`.

| Planned | KML | Green Card | IWG1 file | IWG1 start (Z) | Match confidence | Reason |
| --- | --- | --- | --- | --- | --- | --- |
| NM17 B | `data/er2/NM17 B KML.kml` | `data/er2/NM17 B ER2 Green Card1.xlsx` | `data/er2/n806_2026-04-19.txt` | 2026-04-19T21:50:07Z | high | only NM17 B sortie cached; date is the next ER-2 flight day after the planned product was issued |
| CO07v4 | `data/er2/CO07v4 KML.kml` | `data/er2/CO07v4 ER2 Green Card1.pdf` | `data/er2/n806_2026-04-21.txt` | 2026-04-21T19:52:47Z | high | sole CO07-series IWG1 cached in this window |
| CO06 | `data/er2/CO06 KML.kml` | `data/er2/CO06 ER2 Green Card1.pdf` | `data/er2/n806_2026-04-24.txt` | 2026-04-24T18:31:24Z | high | sole CO06-series IWG1 cached in this window |

## Held out (no matching as-flown yet)

| Planned | KML | Green Card | Note |
| --- | --- | --- | --- |
| NM09 | `data/er2/NM09 KML.kml` | `data/er2/NM 09 ER2 Green Card1.pdf` | no matching IWG1 cached; defer until one arrives |
| NM10Bv3 | `data/er2/NM10Bv3 KML.kml` | `data/er2/NM10Bv3 ER2 Green Card1.xlsx` | no matching IWG1 cached; defer |

## Unmatched IWG1 traces

| File | Start (Z) | Note |
| --- | --- | --- |
| `data/er2/n806_2026-04-27.txt` | 2026-04-27T17:47:47Z | no planned product paired |
| `data/er2/69f22a875886b06d1b13c252.txt` | 2026-04-29T18:13:41Z | no planned product paired |

## Convention

`load_planned_sortie(kml_path, gc_path)` ([`hyplan/aircraft/_planned_sortie.py`](../../hyplan/aircraft/_planned_sortie.py))
returns a normalized waypoint table; `load_iwg1(path)` returns the as-flown
trace.  Pair them via the table above when running planned-vs-flown
comparisons in [`sortie_replay.ipynb`](sortie_replay.ipynb).
