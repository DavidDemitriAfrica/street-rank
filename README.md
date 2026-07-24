# Street rank

## How structured is your city?

**Site and interactive map:** https://daviddemitriafrica.github.io/street-rank/

Some cities follow a small number of clear street directions. Others
contain curves, interruptions, overlapping grids, and many local
patterns. We draw the central streets of each city in the same 5 km
square, rotate the image so its main street direction points up, and
measure how many stripe patterns are necessary to reconstruct it. A
simple grid needs few. An irregular plan needs many. We call this number
the effective rank (the participation ratio of the squared singular
values, the n2 estimator in Del Giudice 2021). The sample is the largest
cities of each continent, 71 hand-picked plans, and the largest
Philippine cities. The full explanation, the interactive explainer, and
the caveats are on the site.

![71 world cities sorted by effective rank](figures/gallery_global.png)

## Data

`data/clean/city_rank.csv`, one row per city. Key columns:

| Column | What it is |
|---|---|
| `erank` | The headline number. Low is grid-like, high is organic |
| `erank_norm` | Rank on a fixed 200 km street sample. Has a known artifact, see the site |
| `block_m` | Block spacing from the Fourier peak (Manhattan: 81 m) |
| `orient_entropy`, `orient_order`, `grid_share`, `straightness` | Orientation and shape measures |
| `intersection_km2`, `four_way_share`, `deadend_share` | Junction density and mix |
| `street_km`, `pop2024`, `pop`, `netmig_rate`, `income_class`, `region`, `flag` | Context and caveats |

`data/clean/synthetic_anchors.csv` holds synthetic plans measured the
same way (perfect grid 1.9, random curves about 250). Streets are
OpenStreetMap drivable plus pedestrian ways; footpath alleys are
excluded. Every download is logged in `data/provenance.jsonl`.

## Rebuild

```bash
python3 pipeline/run_all.py        # fetch and measure, writes the CSV
python3 pipeline/figures.py        # figures
python3 pipeline/make_site_data.py # site data and thumbnails
```

## Related work

- Boeing (2019), [street orientation entropy](https://doi.org/10.1007/s41109-019-0189-1)
- Louf and Barthelemy (2014), [a typology of street patterns](https://doi.org/10.1098/rsif.2014.0924)
- Barrington-Leigh and Millard-Ball (2020), [street-network sprawl (SNDi)](https://sprawlmap.org)
- Roy and Vetterli (2007), the effective rank
- Del Giudice (2021), [effective dimensionality: a tutorial](https://doi.org/10.1080/00273171.2020.1743631)
- Laziou and Lemoy (2025), [the density structure of world cities](https://doi.org/10.1038/s42949-025-00262-4)
- Boeing (2026), [street networks for every urban area](https://doi.org/10.1177/23998083261446991)

## Citing this

```bibtex
@misc{africa2026streetrank,
  author = {Africa, David Demitri},
  title  = {Street rank: how structured is your city?},
  year   = {2026},
  url    = {https://github.com/DavidDemitriAfrica/street-rank}
}
```

Code MIT. Data ODbL, (c) OpenStreetMap contributors. Philippine
populations from the
[Philippine Internal Migration Dataset](https://github.com/DavidDemitriAfrica/philippines-internal-migration).
