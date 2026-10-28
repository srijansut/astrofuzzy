"""CLI: python -m astrofuzzy.experiments {ablation,sensitivity,hubble,all}."""
import argparse
import csv
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
from scipy import __version__ as scipy_version
from sklearn.metrics import average_precision_score, precision_recall_curve
from .core import Config, simulate, segment

METHODS = ('fcm', 'spatial_fcm', 'bc_fcm', 'bc_ifcm', 'matched')


def choose_threshold(truth, score):
    p, r, t = precision_recall_curve(truth.ravel(), score.ravel())
    f = 2*p[:-1]*r[:-1]/np.maximum(p[:-1]+r[:-1], 1e-12)
    return float(t[np.argmax(f)])


def metrics(scene, result, threshold):
    y, s = scene['truth'], result['score']
    mask = s >= threshold
    tp = np.count_nonzero(mask & y)
    denom = np.count_nonzero(mask)+np.count_nonzero(y)
    return dict(ap=float(average_precision_score(y.ravel(), s.ravel())),
                dice=float(2*tp/max(denom, 1)), prevalence=float(y.mean()),
                background_rmse=float(np.sqrt(np.mean((result['background']-scene['background'])**2))),
                converged=bool(result['converged']), iterations=int(result['iterations']))


def write_rows(path, rows):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ablation(out, seeds=10, size=128):
    thresholds = {}
    # Validation is a separate image; never maximize Dice on test labels.
    validation = simulate(10000, shape=(size, size))
    for method in METHODS:
        thresholds[method] = choose_threshold(validation['truth'], segment(validation['image'], method)['score'])
    rows = []
    for seed in range(seeds):
        scene = simulate(seed, shape=(size, size))
        for method in METHODS:
            start = time.perf_counter()
            result = segment(scene['image'], method)
            row = dict(seed=seed, method=method, threshold=thresholds[method], **metrics(scene, result, thresholds[method]))
            row['seconds'] = time.perf_counter()-start
            rows.append(row)
    write_rows(out/'ablation.csv', rows)
    rng = np.random.default_rng(782)
    summary = {}
    for method in METHODS:
        group = [r for r in rows if r['method'] == method]
        summary[method] = {key: float(np.mean([r[key] for r in group])) for key in ('ap', 'dice', 'background_rmse', 'seconds', 'converged')}
    differences = np.array([r['ap'] for r in rows if r['method']=='bc_ifcm'])-np.array([r['ap'] for r in rows if r['method']=='matched'])
    boot = differences[rng.integers(0, seeds, (2000, seeds))].mean(1)
    summary['paired_AP_bc_ifcm_minus_matched'] = dict(mean=float(differences.mean()), ci95=np.quantile(boot, [.025,.975]).tolist())
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    scene = simulate(0, shape=(size,size))
    plot_scene(out/'synthetic.png', scene['image'], {m: segment(scene['image'],m)['score'] for m in METHODS}, scene['truth'])
    return summary


def sensitivity(out, size=128):
    rows = []
    # A diagnostic sweep, not a test-set hyperparameter selection procedure.
    for gradient in (0., 40., 120.):
        for alpha in (0., .15, .6):
            for lam in (0., 1., 4.):
                for seed in (200, 201, 202):
                    scene = simulate(seed, shape=(size,size), gradient=gradient)
                    result = segment(scene['image'], config=Config(spatial=alpha, hesitation=lam))
                    rows.append(dict(seed=seed, gradient=gradient, spatial=alpha, hesitation=lam,
                                     ap=float(average_precision_score(scene['truth'].ravel(), result['score'].ravel())),
                                     background_rmse=float(np.sqrt(np.mean((result['background']-scene['background'])**2)))))
    write_rows(out/'sensitivity.csv', rows)


def plot_scene(path, image, scores, truth=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels = {'Observed (display stretch)': image, **scores}
    if truth is not None:
        panels['Reference: signal >= 2 sigma'] = truth
    fig, axes = plt.subplots(2, 4, figsize=(13, 6))
    for ax, (name, value) in zip(axes.ravel(), panels.items()):
        lo, hi = np.percentile(value.astype(float), [1, 99.5])
        ax.imshow(value, cmap='magma', vmin=lo, vmax=max(hi, lo+1e-6))
        ax.set_title(name, fontsize=9)
        ax.axis('off')
    for ax in axes.ravel()[len(panels):]:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def hubble(out, size=256, image_path=None):
    from skimage import data, io, img_as_float
    rgb = io.imread(image_path) if image_path else data.hubble_deep_field()
    image = np.einsum('...c,c->...', img_as_float(rgb[..., :3]), np.array([.2125,.7154,.0721]), optimize=False) if rgb.ndim == 3 else rgb.astype(float)
    # Deterministic native-resolution crop; no intensity clipping for inference.
    image = image[:size,:size]
    scores = {m: segment(image,m)['score'] for m in METHODS}
    np.savez_compressed(out/'hubble_scores.npz', image=image, **scores)
    plot_scene(out/'hubble.png', image, scores)
    (out/'hubble.json').write_text(json.dumps(dict(shape=list(image.shape), source='skimage.data.hubble_deep_field' if not image_path else str(image_path), quantitative_accuracy=None, reason='No independent ground-truth labels; display RGB is not calibrated FITS photometry.'), indent=2))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('experiment', choices=['ablation','sensitivity','hubble','all'])
    parser.add_argument('--out', type=Path, default=Path('results'))
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--image', type=Path)
    args = parser.parse_args()
    if args.seeds < 2 or args.size < 16:
        parser.error('at least two seeds and size >=16 required')
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/f'environment_{args.experiment}.json').write_text(json.dumps(dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy_version, configuration=asdict(Config()), command=vars(args)), default=str, indent=2))
    if args.experiment in ('ablation','all'):
        print(json.dumps(ablation(args.out,args.seeds,args.size),indent=2))
    if args.experiment in ('sensitivity','all'):
        sensitivity(args.out,args.size)
    if args.experiment in ('hubble','all'):
        hubble(args.out, max(128,args.size), args.image)

if __name__ == '__main__':
    main()
