# Astrofuzzy: additive sky-background transfer study

A complete runnable research artifact inspired by Kumar et al.'s MRI work. This is an additive astronomical adaptation, **not an exact reproduction of the published MRI algorithm**. The main empirical finding is that background correction helps, but the simpler polynomial-background plus Gaussian-filter comparator wins. Do not present this as established state-of-the-art performance.

## Executed results

Ten independent synthetic images, 96 × 96, seeds 0–9; a separate validation image selects each method's Dice threshold. See `results/ablation.csv` and `results/summary.json`.

| Method | Mean AP | Mean Dice | Background RMSE |
|---|---:|---:|---:|
| FCM | .4467 | .4087 | 134.181 |
| Spatial FCM | .4467 | .4086 | 134.181 |
| BC-FCM | .8753 | .7591 | 4.672 |
| Additive BC-IFCM | .8751 | .7606 | 4.265 |
| Polynomial + Gaussian | **.9734** | **.9011** | **2.502** |

Paired mean AP(full) minus AP(comparator): −.09834; image-bootstrap 95% interval [−.10784, −.08798]. The original reported 18% / 24.6% Dice results cannot be reproduced without their inputs and code and are not asserted here.

## Install and execute

Python 3.9 or later, preferably a fresh virtual environment:

```sh
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
python -m pytest -q -W error
python -m astrofuzzy.experiments all --out results --seeds 10 --size 96
python -m astrofuzzy.experiments hubble --out results --size 256
```

The `all` command includes ablation, the 81-run sensitivity grid, and Hubble visualization. Hubble may require a first-use data download; an explicitly supplied local image works offline:

```sh
python -m astrofuzzy.experiments hubble --image /absolute/path/hubble.png --size 256
```

Outputs include raw CSV, JSON metadata, PNG panels, and compressed Hubble score maps. The supplied Hubble output is a 256 × 256 crop. Model fitting never uses the simulator truth. Percentile clipping is for display only.

## Files

- `astrofuzzy/core.py`: simulator; polynomial design; robust pilot; Base FCM, Spatial FCM, corrected FCM, additive BC-IFCM; Gaussian-filter comparator.
- `astrofuzzy/experiments.py`: complete CLI for ablation, threshold selection, sensitivity, Hubble, bootstrap, and plots.
- `tests/test_core.py`: mathematical and numerical checks.
- `paper/main.pdf`: compiled seven-page IEEE-format paper, visually checked.
- `paper/main.tex`: full IEEE conference-format manuscript with exact implementation equations, results, limitations, and references.
- `paper/manuscript.md`: readable full manuscript text.
- `results/`: executed measurements and image artifacts.
- `SOURCE_BUNDLE.md`: every source file collected into one readable file.

## Mathematical contracts

Skyglow is additive; the polynomial fit is anchored to low-centroid pixels to fix the background/centroid gauge ambiguity. Spatial cost averages neighboring **residuals**, not neighboring memberships. Sugeno nonmembership is `(1-u)/(1+lambda*u)` with lambda ≥ 0. Hesitation is `1-u-nu`. It only modulates background reliability here.

The rank score is unclipped centroid-normalized corrected intensity. It is not a probability. Consequently Base and Spatial FCM have identical AP by rank invariance. Their memberships still differ. The background update and spatial clustering optimize separate conditional objectives; no common descent or global convergence guarantee is claimed.

AP avoids selecting a prediction threshold but still requires a binary target. Synthetic target: noiseless summed source intensity ≥ twice the background-only noise scale. Dice uses a threshold selected on seed 10000 and frozen for the test. Hubble has no reference labels: there is no claimed Hubble AP or Dice.

## Limits

This is a tested research implementation at the supplied scales, not a certified production pipeline. Gaussian galaxies are approximations. No deblending, photometry, FITS calibration, cosmic-ray treatment, masks, or instrument validation is implemented. Full-source broadcasting uses O(number of sources × pixels) memory. Small image-scale tests do not prove large-frame performance. The primary main-table ablation changes spatial and hesitation together; the sensitivity grid exposes them separately.

## Verified inspiration

D. Kumar, R. K. Agrawal, and P. Kumar, IEEE Transactions on Fuzzy Systems 30(3), 687–700 (2022), DOI [10.1109/TFUZZ.2020.3044253](https://doi.org/10.1109/TFUZZ.2020.3044253). Bibliographic metadata verified using the author's public code release. No citation-count or first-ever-application claim is made.
