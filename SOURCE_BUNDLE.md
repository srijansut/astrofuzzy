# Complete source bundle

## astrofuzzy/__init__.py

```python
from .core import Config, simulate, segment, sugeno
__all__ = ['Config', 'simulate', 'segment', 'sugeno']

```

## astrofuzzy/core.py

```python
"""Vectorized additive-background fuzzy source segmentation; not an MRI reproduction."""
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter

def mv(p, a):
    return np.einsum("ij,j->i", p, a, optimize=False)

def weighted_fit(p, w, z, ridge):
    gram = np.einsum("ni,n,nj->ij", p, w, p, optimize=False)
    rhs = np.einsum("ni,n,n->i", p, w, z, optimize=False)
    return np.linalg.solve(gram+ridge*np.eye(p.shape[1]), rhs)

@dataclass(frozen=True)
class Config:
    clusters: int = 3
    m: float = 2.0
    spatial: float = 0.15
    hesitation: float = 1.0
    degree: int = 2
    ridge: float = 1e-5
    iterations: int = 80
    tolerance: float = 1e-5


def design(shape, degree=2):
    if degree not in (0, 1, 2, 3):
        raise ValueError('degree must be 0..3')
    y, x = np.meshgrid(np.linspace(-1, 1, shape[0]), np.linspace(-1, 1, shape[1]), indexing='ij')
    return np.stack([x**a * y**b for a in range(degree+1) for b in range(degree+1-a)], -1).reshape(-1, (degree+1)*(degree+2)//2)


def simulate(seed=0, shape=(128, 128), sources=30, gradient=80., read_noise=3., gain=2.):
    """Electron-equivalent intensities; elliptical Gaussian source tensor, no pixel loops."""
    if min(shape) < 16 or sources < 1 or gradient < 0 or read_noise < 0 or gain <= 0:
        raise ValueError('invalid scene parameters')
    rng = np.random.default_rng(seed)
    y, x = np.indices(shape, dtype=float)
    cx = rng.uniform(5, shape[1]-5, sources)
    cy = rng.uniform(5, shape[0]-5, sources)
    sx = rng.uniform(.9, 2.4, sources)
    sy = sx * rng.uniform(.8, 1.8, sources)
    angle = rng.uniform(0, np.pi, sources)
    amp = np.exp(rng.uniform(np.log(12), np.log(180), sources))
    dx, dy = x[None]-cx[:, None, None], y[None]-cy[:, None, None]
    xx = dx*np.cos(angle[:, None, None])+dy*np.sin(angle[:, None, None])
    yy = -dx*np.sin(angle[:, None, None])+dy*np.cos(angle[:, None, None])
    signal = (amp[:, None, None]*np.exp(-.5*((xx/sx[:, None, None])**2+(yy/sy[:, None, None])**2))).sum(0)
    p = design(shape)
    coeff = np.array([40+gradient, .25*gradient, .2*gradient, .55*gradient, -.2*gradient, .2*gradient])
    background = (mv(p, coeff)).reshape(shape)
    image = rng.poisson((background+signal)*gain)/gain + rng.normal(0, read_noise, shape)
    sigma = np.sqrt(background/gain+read_noise**2)
    # Operational label defined from the noiseless summed source signal, not observed data.
    truth = signal >= 2*sigma
    return dict(image=image, signal=signal, background=background, sigma=sigma, truth=truth,
                centers=np.column_stack([cy, cx]), amplitudes=amp)


def sugeno(u, lam):
    if lam < 0:
        raise ValueError('Sugeno lambda must be nonnegative')
    nu = (1-u)/(1+lam*u)
    return nu, np.maximum(0, 1-u-nu)


def background_fit(image, degree=2, ridge=1e-5):
    """Upper-tail robust polynomial pilot; IRLS iterations are over models, not pixels."""
    p = design(image.shape, degree)
    z = image.ravel()
    a = np.linalg.lstsq(p, z, rcond=None)[0]
    for _ in range(12):
        r = z-mv(p, a)
        scale = max(1.4826*np.median(np.abs(r-np.median(r))), 1e-6)
        w = np.where(r > 2*scale, (2*scale/np.maximum(r, 1e-9))**2, 1.)
        a = weighted_fit(p, w, z, ridge)
    return (mv(p, a)).reshape(image.shape)


def segment(image, method='bc_ifcm', config=None):
    """Return continuous scores, background and diagnostics. Iterative spatial surrogate.

    Bright-class membership is not monotone with intensity outside the centroid range;
    the rank score uses an unclipped centroid-normalized intensity instead.
    """
    c = config or Config()
    image = np.asarray(image, dtype=float)
    if image.ndim != 2 or min(image.shape) < 3 or not np.isfinite(image).all():
        raise ValueError('image must be finite 2D with dimensions >= 3')
    if method not in ('fcm', 'spatial_fcm', 'bc_fcm', 'bc_ifcm', 'matched'):
        raise ValueError('unknown method')
    if c.m <= 1 or c.clusters < 2 or c.spatial < 0 or c.ridge <= 0 or c.iterations < 1 or c.tolerance <= 0:
        raise ValueError('invalid configuration')
    sugeno(np.array([.5]), c.hesitation)
    corrected = method in ('bc_fcm', 'bc_ifcm', 'matched')
    bg = background_fit(image, c.degree, c.ridge) if corrected else np.zeros_like(image)
    if method == 'matched':
        return dict(score=gaussian_filter(image-bg, 1.3), background=bg, iterations=0, converged=True)
    offset = float(np.median(image-bg))
    scale = max(float(np.std(image-bg)), 1e-8)
    z = (image-bg-offset)/scale
    centers = np.quantile(z, np.linspace(.25, .99, c.clusters))
    u = np.full((*image.shape, c.clusters), 1/c.clusters)
    alpha = c.spatial if method in ('spatial_fcm', 'bc_ifcm') else 0.
    p = design(image.shape, c.degree) if corrected else None
    trace = []
    converged = False
    for iteration in range(c.iterations):
        old = u.copy()
        d = (z[..., None]-centers)**2
        # Reflected 3x3 mean excludes the central pixel.
        neighbor = (9*uniform_filter(d, size=(3, 3, 1), mode='reflect')-d)/8
        cost = np.maximum(d+alpha*neighbor, 1e-12)
        logw = -np.log(cost)/(c.m-1)
        logw -= logw.max(-1, keepdims=True)
        u = np.exp(logw)
        u /= u.sum(-1, keepdims=True)
        # Spatial energy includes neighboring corrected observations.
        nz = (9*uniform_filter(z, size=3, mode='reflect')-z)/8
        um = u**c.m
        centers = (um*(z+alpha*nz)[..., None]).sum((0, 1))/((1+alpha)*um.sum((0, 1))+1e-12)
        if corrected:
            # Anchor background to low-intensity class; fixes additive gauge.
            k = np.argmin(centers)
            _, pi = sugeno(u[..., k], c.hesitation if method == 'bc_ifcm' else 0.)
            weights = (u[..., k]**c.m*(1-pi)).ravel()
            weights = np.maximum(weights, 1e-8)
            a = weighted_fit(p, weights, image.ravel(), c.ridge)
            candidate = (mv(p, a)).reshape(image.shape)
            bg = .5*bg+.5*candidate
            z = (image-bg-offset)/scale
        trace.append(float(np.mean(np.sum(um*cost, -1))))
        if np.max(np.abs(u-old)) < c.tolerance:
            converged = True
            break
    # Monotone projection avoids FCM's bright-tail membership reversal.
    order = np.argsort(centers)
    ordered = centers[order]
    span = max(ordered[-1]-ordered[0], 1e-8)
    score = (z-ordered[0])/span
    # This is a centroid-based intensity score, not a calibrated posterior.
    nu, pi = sugeno(u, c.hesitation if method == 'bc_ifcm' else 0.)
    return dict(score=score, membership=u, nonmembership=nu, hesitation=pi,
                background=bg, centers=centers*scale+offset, iterations=iteration+1,
                converged=converged, surrogate_trace=np.array(trace))

```

## astrofuzzy/experiments.py

```python
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

```

## paper/main.tex

```latex
\documentclass[conference]{IEEEtran}
\usepackage{amsmath,amssymb,graphicx,booktabs,url}
\usepackage[hidelinks]{hyperref}
\title{Additive Sky-Background Correction for Fuzzy Astronomical Source Segmentation: A Reproducible Transfer Study}
\author{\IEEEauthorblockN{Anonymous Undergraduate Research Manuscript}\IEEEauthorblockA{Mathematics and Computing\\Reproducible computational study, October 2026}}
\begin{document}
\maketitle
\begin{abstract}
Smooth sky backgrounds interfere with the extraction of faint astronomical sources, while binary segmentation masks discard the graded structure of point-spread-function wings. Inspired by bias-corrected intuitionistic fuzzy clustering for magnetic resonance imaging, this work develops an additive-background adaptation and evaluates whether its extra mathematical machinery improves source ranking. The implementation combines vectorized fuzzy c-means, a local residual penalty, a polynomial background model, and Sugeno hesitation. It explicitly separates fuzzy memberships from an intensity-monotone detection score. Evaluation uses average precision, validation-selected Dice, background error, and paired image-level bootstrap comparisons. On ten seeded synthetic images containing elliptical Gaussian sources, polynomial backgrounds, photon noise, and read noise, the proposed adaptation attains mean average precision 0.875 and Dice 0.761, compared with 0.447 and 0.409 for an uncorrected fuzzy baseline. A robust polynomial-background plus Gaussian-filter comparator performs better, attaining 0.973 and 0.901. Thus the experiment supports background removal but does not establish a benefit from intuitionistic weighting. An unlabeled Hubble image supplies a qualitative domain-transfer example only. The contribution is an auditable transfer formulation and evaluation protocol, rather than a claim of state-of-the-art astronomical extraction.
\end{abstract}
\begin{IEEEkeywords}
Astronomical image segmentation, fuzzy c-means, sky background, intuitionistic fuzzy sets, average precision, reproducibility.
\end{IEEEkeywords}

\section{Introduction}
Astronomical source extraction begins with a difficult separation: photons from a faint object must be distinguished from a background whose level varies across the detector. In ground-based images, sky emission, scattered light, and light pollution can introduce broad intensity structure. A single image-wide threshold can accept bright portions of the background while rejecting legitimate low-surface-brightness source structure elsewhere. The problem becomes especially visible around diffuse source wings: the underlying brightness profile changes continuously, but the segmentation decision is discrete.

Hard clipping is not, by itself, a defect in an astronomical pipeline. Catalog construction ultimately requires operational decisions, and established extraction systems estimate local backgrounds and detection thresholds. The problem is treating a single unvalidated binary mask as a complete measure of success. A Gaussian profile has nonzero support everywhere. Therefore a claim to recover its complete binary footprint is undefined without a surface-brightness or signal-to-noise convention. Changing that convention changes both class prevalence and the apparent difficulty of the task.

Medical image segmentation offers a related computational idea. A smoothly varying nuisance field can be estimated alongside uncertain class assignments. Kumar, Agrawal, and Kumar developed bias-corrected intuitionistic fuzzy c-means with spatial neighborhood information for human brain MRI segmentation \cite{kumar}. Their work motivates the present study, but the physical correspondence is incomplete: astronomical skyglow on a linear detector is principally additive, and astronomical sources vary continuously in brightness and size. An MRI algorithm cannot be transferred by renaming its variables alone.

This paper asks a narrower and testable question: does an additive, spatial, intuitionistic fuzzy adaptation provide useful source rankings beyond uncorrected intensity clustering and a simple background-aware comparator? The question is evaluated using seeded synthetic data with explicit latent truth. A public Hubble display image is used to examine visual behavior, without assigning artificial ground-truth labels to it.

The principal contributions are an explicit additive formulation with identifiable background anchoring; a vectorized implementation without Python loops over pixels; a score definition that avoids bright-tail membership reversal; and an evaluation that separates ranking, threshold selection, and physical background error. Equally important, the negative comparison against the simpler filtered detector is retained. A robust research claim must survive a strong baseline, not merely improve an unsuitable one.

\section{Related Work and Scope of Transfer}
\subsection{Bias correction and intuitionistic clustering}
Kumar et al. \cite{kumar} combine bias correction, intuitionistic fuzzy reasoning, and spatial neighborhood information for MRI segmentation. The verified bibliographic record is IEEE Transactions on Fuzzy Systems, volume 30, number 3, pages 687--700, March 2022, DOI 10.1109/TFUZZ.2020.3044253. The present software is inspired by these concepts; it is not a line-by-line reproduction of that algorithm or a validation of its original MRI results. In particular, our additive background fit and anchored regression are application-specific choices.

An intuitionistic fuzzy representation associates membership, nonmembership, and hesitation with an observation. Sugeno-type generators supply a constrained relationship between these quantities \cite{zenian}. These quantities should not automatically be interpreted as calibrated Bayesian probabilities or physical uncertainty estimates. Here hesitation is an algebraic descriptor and a regression reliability factor. Its contribution is evaluated by removing the factor, rather than inferred from the appearance of a smoother segmentation map.

\subsection{Astronomical background estimation}
Existing astronomy software already treats background estimation as central to source detection. SExtractor describes a smoothly varying background map estimated from local image statistics \cite{sextractor}. Photutils provides background estimation, convolution, segmentation, and deblending components \cite{photutils}. Consequently, the application of smooth-background removal to astronomy is established practice and is not claimed as a new discovery here.

The relevant novelty question concerns the value of the particular fuzzy coupling. This study includes a robust polynomial-background and Gaussian-filter comparator to make that question falsifiable. The comparator is intentionally simple and is not a substitute for full benchmarking against tuned SExtractor, SEP, Photutils, or multiscale methods. None of those packages is represented by the comparator's numbers.

\subsection{Ranking and segmentation evaluation}
Average precision (AP) summarizes a precision--recall ranking without selecting one operating threshold. We use the noninterpolated definition implemented in scikit-learn \cite{ap}. This avoids conflating score quality with an arbitrary membership cutoff. However, AP still requires binary reference labels, depends on the evaluation population, and does not measure photometric accuracy. Reporting AP cannot repair a poorly specified reference mask.

Dice remains useful when a deployed segmentation decision is required. We therefore retain it, select its threshold using a separate validation image, and freeze that threshold for all test images. The resulting AP and Dice answer different questions. An improvement in one need not imply an improvement in the other, and neither establishes that the recovered catalog contains distinct, correctly deblended astronomical objects.

\section{Observation Model and Simulator}
\subsection{Additive photon-counting model}
Let pixel $i$ have spatial coordinate $\boldsymbol r_i$ and noiseless source intensity $s_i$. We generate
\begin{align}
 n_i &\sim \operatorname{Poisson}\{g[b_i+s_i]\},\\
 y_i &= n_i/g+\epsilon_i,\quad
 \epsilon_i\sim\mathcal N(0,\sigma_r^2),\label{eq:obs}
\end{align}
where $g>0$ converts image units into counts, $b_i$ is the nonnegative sky background, and $\sigma_r$ is read-noise standard deviation in image units. Thus $\operatorname{Var}(y_i\mid s_i,b_i)=(b_i+s_i)/g+\sigma_r^2$. The algorithms do not receive the latent background or source map.

The simulator creates a superposition of elliptical Gaussian components,
\begin{equation}
 s_i=\sum_{j=1}^{M} A_j\exp\left[-\frac12\left\{
 \frac{x_{ij}'^2}{a_j^2}+\frac{y_{ij}'^2}{d_j^2}\right\}\right],
\end{equation}
where $(x_{ij}',y_{ij}')$ are source-centered coordinates rotated by $\theta_j$. Source tensors are formed by broadcasting over source index and the two spatial dimensions. Source overlap arises naturally. Elliptical Gaussian components approximate compact extended objects; they are not realistic models of arbitrary galaxy morphology or instrument-specific point-spread functions.

\subsection{Background and source populations}
Normalized detector coordinates $x,y\in[-1,1]$ define a total-degree polynomial basis. For degree two the implementation orders it as
\begin{equation}
 \boldsymbol p_i=(1,y,y^2,x,xy,x^2)^\top.
\end{equation}
The simulated coefficient vector is
\begin{equation}
 \boldsymbol a_{\mathrm{true}}=
 (40+G,\;0.25G,\;0.20G,\;0.55G,\;-0.20G,\;0.20G)^\top,
\end{equation}
with $G\geq0$. This parameterization keeps the tested background positive and changes its overall level as well as its variation. A gradient sweep therefore also changes shot-noise magnitude; it is not an isolated gradient-only intervention.

The executed ablation uses $96\times96$ images, $M=30$, $G=80$, $g=2$, and $\sigma_r=3$. Source centers are uniform inside a five-pixel border. Minor-axis scale $a_j$ is uniform on $[0.9,2.4]$, axis ratio $d_j/a_j$ on $[0.8,1.8]$, orientation on $[0,\pi]$, and peak amplitude log-uniform on $[12,180]$. These choices create both visible sources and faint wings near the noise floor. They do not represent a measured astronomical luminosity function.

\subsection{Operational reference mask}
Define background-only noise scale
\begin{equation}
 \sigma_{b,i}=\sqrt{b_i/g+\sigma_r^2},\qquad
 t_i=\mathbb 1[s_i\geq2\sigma_{b,i}].\label{eq:truth}
\end{equation}
This reference is derived from the latent summed source map, not the observed noisy image or an algorithm's predictions. Overlapping wings can jointly cross the boundary. The factor two defines the scientific evaluation target; it is not a detection threshold supplied to the methods. AP is threshold-free with respect to predicted scores, not with respect to the definition of truth.

This explicit convention resolves the infinite-support ambiguity without suggesting that the chosen contour is uniquely correct. A broader validation should repeat evaluation at multiple latent signal-to-noise boundaries and source populations. Those experiments are not silently substituted for the primary setting reported here.

\section{Vectorized Fuzzy and Background Models}
\subsection{Classical fuzzy clustering}
For $N$ pixels and $K$ classes, memberships $u_{ik}$ obey $u_{ik}\geq0$ and $\sum_k u_{ik}=1$. Classical fuzzy c-means minimizes
\begin{equation}
 J_{\mathrm{FCM}}(U,\boldsymbol c)=
 \sum_{i=1}^{N}\sum_{k=1}^{K}u_{ik}^{m}(z_i-c_k)^2,
 \quad m>1.\label{eq:fcm}
\end{equation}
For uncorrected methods, $z_i=(y_i-o)/q$, with $o$ the image median and $q$ its standard deviation, floored at $10^{-8}$. Corrected methods compute these normalization constants once from a pilot background-subtracted image. They remain fixed throughout iteration, so the centroid and regression calculations share a stable intensity scale.

The default configuration uses $K=3$, $m=2$, at most 80 iterations, and stopping tolerance $10^{-5}$ on the maximum membership change. Centroids start at evenly spaced quantile levels from 0.25 to 0.99. These settings are fixed before test evaluation. The implementation uses log-domain normalization of inverse distances to avoid overflow in membership updates.

\subsection{Spatial residual penalty}
Let $\mathcal N_i$ be the reflected $3\times3$ neighborhood excluding its center, with eight equally weighted positions. Reflection at boundaries can repeat image coordinates; the definition is the numerical stencil, not a variable-size graph. Set
\begin{align}
 d_{ik}&=(z_i-c_k)^2,\\
 D_{ik}&=d_{ik}+\frac{\alpha}{8}\sum_{j\in\mathcal N_i}(z_j-c_k)^2,\\
 J_{\mathrm{sp}}(U,\boldsymbol c\mid z)&=
 \sum_{i,k}u_{ik}^{m}D_{ik}.\label{eq:sp}
\end{align}
The spatial mean is computed as $(9\mathcal H_3[d]-d)/8$, where $\mathcal H_3$ is a reflected uniform filter over spatial dimensions only. No loop visits individual pixels.

For fixed $z$ and centroids, the membership solution is
\begin{equation}
 u_{ik}=\frac{\widetilde D_{ik}^{-1/(m-1)}}
 {\sum_{\ell}\widetilde D_{i\ell}^{-1/(m-1)}},
 \quad\widetilde D_{ik}=\max(D_{ik},10^{-12}).
\end{equation}
With $\overline z_i=\frac18\sum_{j\in\mathcal N_i}z_j$, the corresponding centroid update is
\begin{equation}
 c_k=\frac{\sum_i u_{ik}^m(z_i+\alpha\overline z_i)}
 {(1+\alpha)\sum_i u_{ik}^m+10^{-12}}.\label{eq:center}
\end{equation}
This is a neighboring-observation residual penalty. It is not a Potts penalty on adjacent memberships. Equations \eqref{eq:sp}--\eqref{eq:center} specify precisely which form of spatial FCM is implemented.

\subsection{Sugeno nonmembership and hesitation}
For $\lambda\geq0$, define
\begin{align}
 \nu_{ik}&=\frac{1-u_{ik}}{1+\lambda u_{ik}},\\
 \pi_{ik}&=1-u_{ik}-\nu_{ik}
 =\frac{\lambda u_{ik}(1-u_{ik})}{1+\lambda u_{ik}}.\label{eq:sugeno}
\end{align}
Because $u\in[0,1]$ and $\lambda\geq0$, both $\nu$ and $\pi$ are nonnegative and $u+\nu+\pi=1$. At $\lambda=0$, hesitation is exactly zero. This provides a natural ablation. Negative $\lambda$ is rejected rather than permitted to violate the intuitionistic constraint.

The construction is deterministic given $u$. It supplies no independent observation of uncertainty and does not establish a statistical confidence interval. In this implementation only the inferred background class's hesitation modifies regression weights. We deliberately avoid an unmotivated replacement of normalized memberships by $u+\pi$, which need not sum to one across classes.

\subsection{Pilot estimate and anchored background regression}
An initial polynomial background is fitted by 12 iteratively reweighted least-squares passes. For residual $r_i=y_i-\boldsymbol p_i^\top\boldsymbol a$, let $h$ be $1.4826$ times its median absolute deviation, floored at $10^{-6}$. The pilot weight is
\begin{equation}
 w_i^{(0)}=\begin{cases}
 (2h/\max(r_i,10^{-9}))^2,&r_i>2h,\\
 1,&\text{otherwise}.
 \end{cases}
\end{equation}
This asymmetric rule reduces positive source contamination. It is an algorithmic robust pilot, without a claim that the sequence globally minimizes one fixed loss.

An additive model containing both a constant background term and unconstrained class centers has a gauge ambiguity. Moving a constant from the centers to the background leaves predicted intensities unchanged. We resolve this by treating the lowest-centroid class $k_0=\arg\min_k c_k$ as observations of the sky level. The background regression target is the observed intensity itself, anchoring the residual background class approximately to zero.

At each corrected-method iteration, define
\begin{equation}
 w_i=\max\{u_{ik_0}^{m}(1-\pi_{ik_0}),10^{-8}\}.
\end{equation}
The exact conditional regression objective is
\begin{equation}
 J_b(\boldsymbol a\mid U)=\sum_iw_i
 (y_i-\boldsymbol p_i^\top\boldsymbol a)^2
 +\rho\|\boldsymbol a\|_2^2,\label{eq:bias}
\end{equation}
with solution
\begin{equation}
 \widehat{\boldsymbol a}=(P^\top WP+\rho I)^{-1}P^\top W\boldsymbol y.
\end{equation}
The implementation solves this small linear system rather than forming an inverse. It damps the fitted background,
\begin{equation}
 b^{(t+1)}=\tfrac12 b^{(t)}+\tfrac12P\widehat{\boldsymbol a},
 \quad z_i^{(t+1)}=(y_i-b_i^{(t+1)}-o)/q.
\end{equation}
The default ridge coefficient is $10^{-5}$ and applies to all polynomial coefficients, including the intercept.

These alternating operations minimize conditional objectives, not a proven common global objective: the anchored regression ignores source-class residuals and differs from differentiation of \eqref{eq:sp} with respect to the background. Accordingly, the stored spatial-energy trace is a diagnostic surrogate and is not asserted to decrease monotonically. Membership stabilization is reported separately from any optimization guarantee.

\subsection{Source-ranking score and its limitations}
A direct bright-class membership can fail as a source-ranking statistic. In one-dimensional FCM, when an observed intensity tends far above every centroid, all centroid distances become nearly equal. Memberships then approach $1/K$, so a very bright source may receive a smaller bright-class membership than a moderately bright one.

We therefore report memberships separately and use
\begin{equation}
 R_i=\frac{z_i-c_{\min}}{\max(c_{\max}-c_{\min},10^{-8})}\label{eq:score}
\end{equation}
as an unbounded detection score. It is intentionally not clipped: clipping introduces ties among bright pixels and changes AP. The score is neither a probability nor a membership. Its monotonicity ensures that extreme bright sources retain high ranks.

Equation \eqref{eq:score} also imposes an important limitation. For a fixed background, it is a positive affine transformation of corrected intensity. Thus uncorrected FCM and spatial FCM must have identical AP. Their different memberships can influence a transferred numerical threshold, but cannot change pixel ranking. In corrected variants, spatial or intuitionistic components affect AP only through the estimated background. This property is disclosed as an analytical consequence, not interpreted as evidence that spatial reasoning intrinsically improves detection.

\subsection{Computational properties}
For $Q$ polynomial coefficients, fuzzy updates require $O(NK)$ work per iteration; weighted normal-equation construction costs $O(NQ^2)$ and solving costs $O(Q^3)$. Storage is $O(NK+NQ)$. The Gaussian simulator requires $O(MN)$ temporary storage and is appropriate for the supplied image sizes; very large catalogs require source batching or convolutional rendering. Python loops are confined to model iterations, polynomial powers, experiment configurations, methods, and plotting panels. Tensor operations handle pixels.

\section{Experimental Protocol}
\subsection{Ablations and comparator}
We evaluate uncorrected FCM; spatial FCM with $\alpha=0.15$; background-corrected FCM without spatial or hesitation terms; the full additive adaptation with $\alpha=0.15$ and $\lambda=1$; and a Gaussian-filter comparator. The last method estimates the same robust polynomial pilot and filters the residual with Gaussian standard deviation 1.3 pixels. It is a matched-filter-style comparator with a fixed approximate source scale, not an exact heteroscedastic optimal filter.

The comparison between corrected FCM and the full method jointly changes spatial weighting and hesitation. The separate sensitivity experiment varies $\alpha$ and $\lambda$, including zero, to expose their individual behavior. A fully factorial main ablation would strengthen causal attribution and is not implied by the five-row table.

\subsection{Data separation and metrics}
Validation uses simulator seed 10000. Test seeds are 0 through 9. For each method, a threshold maximizing validation Dice is selected from its precision--recall curve and then held fixed. The latent truth never enters model fitting. There is no per-image test threshold optimization.

For ranked precision $P_n$ and recall $Q_n$, AP is
\begin{equation}
 \operatorname{AP}=\sum_n(Q_n-Q_{n-1})P_n.
\end{equation}
For the frozen threshold, Dice is $2TP/(2TP+FP+FN)$. Background error is
\begin{equation}
 \operatorname{RMSE}_b=\sqrt{N^{-1}\sum_i(\widehat b_i-b_i)^2}.
\end{equation}
For uncorrected methods, $\widehat b=0$ by definition. Their background error is a reference for the absent correction, not a separately fitted estimate. Per-image positive prevalence is recorded because precision--recall behavior depends on class balance. Means are computed across images, rather than by pooling all pixels into a falsely independent sample.

For the principal comparator, differences in AP are paired by seed. A percentile bootstrap resamples ten image indices with replacement 2000 times using RNG seed 782. The 95\% interval is descriptive at this small sample size. Pixels within an image are not treated as independent bootstrap units.

\subsection{Sensitivity and real-image protocol}
The sensitivity grid uses background parameter $G\in\{0,40,120\}$, spatial coefficient $\alpha\in\{0,0.15,0.6\}$, and generator coefficient $\lambda\in\{0,1,4\}$, with seeds 200--202. This yields 81 runs at $96\times96$ resolution. The grid is diagnostic and is not used to select the configuration of the reported primary test.

The real-image experiment uses the top-left $256\times256$ native-resolution crop of \texttt{skimage.data.hubble\_deep\_field()}, documented as a Hubble eXtreme Deep Field display image \cite{hubble}. RGB is converted to grayscale using fixed luminance weights. Percentile stretching applies only to visualization. No synthetic skyglow is added to this example. A space-based image does not directly validate terrestrial light-pollution removal, and an RGB display image is not calibrated linear FITS data.

\section{Results and Discussion}
\subsection{Measured synthetic results}
Table~\ref{tab:main} contains executed measurements, not projected values. The full additive adaptation improves AP from 0.447 to 0.875 and validation-selected Dice from approximately 0.409 to 0.761 compared with uncorrected FCM. Its mean background RMSE is 4.265 image units. These results support the usefulness of explicitly modeling a varying additive background under the simulator's assumptions.

\begin{table}[t]
\caption{Executed ten-image synthetic ablation; means across seeds.}
\label{tab:main}
\centering
\begin{tabular}{lrrr}
\toprule
Method & AP $\uparrow$ & Dice $\uparrow$ & $\mathrm{RMSE}_b\downarrow$\\
\midrule
FCM & .4467 & .4087 & 134.181\\
Spatial FCM & .4467 & .4086 & 134.181\\
BC-FCM & .8753 & .7591 & 4.672\\
Additive BC-IFCM & .8751 & .7606 & 4.265\\
Polynomial + Gaussian & \textbf{.9734} & \textbf{.9011} & \textbf{2.502}\\
\bottomrule
\end{tabular}
\end{table}

The stronger comparison changes the interpretation. The Gaussian-filter comparator outperforms the full adaptation by approximately 0.0983 AP. The paired interval for AP(full) minus AP(comparator) is $[-0.1078,-0.0880]$. Under this population, the added fuzzy machinery does not compensate for the comparator's more effective background estimate and spatial averaging. The result is not evidence that intuitionistic clustering is universally ineffective; it is evidence against superiority for this implementation and experiment.

Corrected FCM and the full adaptation are nearly tied in AP. The difference is approximately $-0.00019$ in favor of corrected FCM, while Dice changes slightly in the other direction. These small changes cannot justify a strong claim about hesitation. The full method's modestly lower background RMSE is worth recording, but the simpler comparator remains better on that metric too.

The equality of uncorrected AP values is predicted by \eqref{eq:score}. It is an implementation-level invariance, not a failed significance test. This observation is useful because it prevents attributing ranking improvements to centroid updates that only rescale scores. Any future claim that the spatial penalty improves ranking directly must use a spatially varying score and evaluate it separately.

\begin{figure*}[t]
\centering
\includegraphics[width=.98\textwidth]{../results/synthetic.png}
\caption{Executed synthetic example, seed 0. Panels show observed intensities, method scores, and the latent operational reference. Each panel has its own display stretch; apparent brightness is not a calibrated probability comparison. The filtered comparator concentrates compact-source response after background subtraction.}
\label{fig:synth}
\end{figure*}

\subsection{Sensitivity evidence}
Across the 27 runs at each background setting, mean AP was 0.9185 for $G=0$, 0.8805 for $G=40$, and 0.8438 for $G=120$. These aggregate values include all tested spatial and hesitation settings. The released \texttt{sensitivity.csv} preserves every run, including background RMSE and AP. The grid should be read as a robustness diagnostic rather than a search for a favorable setting. Increasing $\alpha$ widens the influence of neighboring residuals and can change which pixels anchor the fitted background. Increasing $\lambda$ changes reliability weights through hesitation but does not add measured information. An improvement at one point in this grid would require confirmation on new seeds before becoming a tuned test result.

A special case helps interpretation: $\lambda=0$ sets $\pi=0$ exactly, but leaves spatial weighting active when $\alpha>0$. Setting both to zero removes these additions while retaining background regression. This makes the distinction between background correction and intuitionistic reasoning explicit. The current simulator shares its polynomial family with the fitted background, which favors all polynomial-based methods. A spline sky, scattered-light feature, or diffuse nebula could produce substantially different outcomes.

\subsection{Qualitative Hubble behavior}
Figure~\ref{fig:hubble} illustrates source-emphasizing scores on the Hubble crop. The images expose bright compact and extended structures as well as spatially varying intensity. Because each panel is stretched separately, visual contrast cannot be converted into a quantitative advantage. Smooth subtraction may emphasize compact peaks while absorbing broad galaxy light into the inferred background; the figure alone cannot determine whether such subtraction is scientifically appropriate.

The experiment supplies no independently annotated source masks, source catalog, exposure map, or calibrated fluxes. Consequently no Hubble Dice or AP is reported. A mask generated by another thresholding algorithm would measure agreement with that algorithm, not ground-truth accuracy. Likewise, successful visualization on a space telescope image does not establish performance on ground-based skyglow. These restrictions preserve the distinction between a useful visual example and an independently validated application.

\begin{figure*}[t]
\centering
\includegraphics[width=.98\textwidth]{../results/hubble.png}
\caption{Qualitative Hubble display-image experiment on a deterministic $256\times256$ crop. No reference segmentation is available. Panel-specific stretches aid inspection but do not imply comparable calibrated score ranges. The public image is accessed through scikit-image; it is not used as a labeled accuracy benchmark.}
\label{fig:hubble}
\end{figure*}

\subsection{Unfilled external-validation table}
Table~\ref{tab:pending} is deliberately an unfilled reporting schema for measurements that were not performed. It is separated from Table~\ref{tab:main} so that placeholder content cannot be mistaken for evidence. Completing it requires calibrated independent data and appropriate reference measurements, rather than extrapolating from synthetic pixel AP.
\begin{table}[t]
\caption{External-validation schema. NR means not run.}
\label{tab:pending}
\centering
\setlength{\tabcolsep}{3pt}
\small
\begin{tabular}{lccc}
\toprule
Benchmark & Recall & False detections & Flux error\\
\midrule
Ground-based FITS & NR & NR & NR\\
Injection recovery & NR & NR & NR\\
Photutils & NR & NR & NR\\
SExtractor & NR & NR & NR\\
\bottomrule
\end{tabular}
\end{table}

\subsection{Threats to validity}
The primary restriction is simulator validity. Real point-spread functions can have non-Gaussian wings, vary spatially, and include diffraction structure. Galaxies do not generally follow one elliptical Gaussian. Saturation, cosmic rays, bad pixels, source confusion, correlated resampling noise, and diffuse astrophysical emission are absent. The polynomial model is correctly specified for the main synthetic sky, so its measured accuracy should not be extrapolated to arbitrary backgrounds.

A second restriction is the task itself. Pixel ranking is not catalog extraction. The implementation does not deblend overlapping sources or estimate object identities, completeness as a function of magnitude, astrometric error, or photometric bias. A high AP can coexist with a merged pair of stars. Object-level matching and flux-conserving measurement require a different evaluation layer.

A third restriction concerns statistical breadth. Ten small images and one validation image are sufficient for a reproducible computational demonstration, but not for universal superiority. The bootstrap represents seed variation under this one scene population. It does not cover instrument variation or uncertainty in the astrophysical model. The reported parameter defaults are not claimed optimal.

Finally, the method is an anchored alternating heuristic. The conditional objectives are explicit, but there is no proof that the full sequence descends a single scalar objective or reaches a global minimum. The observed primary runs stabilize under the supplied tolerance. This empirical behavior must not be rewritten as a general convergence theorem.

\section{Reproducibility and Software Verification}
The software package contains a seeded simulator, vectorized models, a command-line experiment runner, tests, raw CSV records, environment metadata, figures, and this manuscript source. The executed environment used Python 3.9.6, NumPy 2.0.2, and SciPy 1.13.1; the dependency snapshot records additional installed packages. Fixed RNG seeds govern scenes and bootstrap resampling. Timing values in the raw output are descriptive and are not controlled hardware benchmarks.

Tests verify deterministic simulation, normalized membership sums, finite scores, recovery of a known smooth background within a stated tolerance, valid Sugeno decomposition, constant-image handling, rejection of invalid inputs, polynomial design dimensions, and AP invariance under positive affine score transformations. The test suite passes with warnings treated as errors for the core paths. Weighted products use explicit tensor contractions to avoid runtime-specific matrix-multiplication warnings observed in the host's numerical stack.

The supplied project is a research implementation with a reproducible tested configuration. It does not claim certification, exhaustive input coverage, or production validation for telescope pipelines. The original user-reported Dice values of 18\% and 24.6\% are not included as reproduced results: the corresponding code, seeds, reference labels, and threshold definitions were not supplied. The measurements in this paper come from the delivered implementation and stated protocol.

\section{Conclusion}
Transferring bias-corrected intuitionistic fuzzy ideas from MRI to astronomical imagery requires changing both the observation model and the evaluation. An additive sky model, explicit latent reference contour, separate score construction, and validation-only threshold selection make the resulting claims auditable. The delivered adaptation substantially improves over uncorrected fuzzy intensity ranking in the tested synthetic setting. However, a simpler robust-background and Gaussian-filter comparator performs better, while hesitation offers no demonstrated AP advantage.

The defensible outcome is therefore a reproducible transfer study and a clear boundary on its claims. Further evidence would require independent ground-based calibrated images, realistic injection recovery, object-level matching, photometric evaluation, and comparisons with established extraction systems. Until that evidence exists, neither the visual Hubble example nor the synthetic improvement over plain FCM supports a claim of a superior astronomical detector.

\begin{thebibliography}{9}
\bibitem{kumar} D. Kumar, R. K. Agrawal, and P. Kumar, ``Bias-corrected intuitionistic fuzzy C-means with spatial neighborhood information approach for human brain MRI image segmentation,'' \emph{IEEE Trans. Fuzzy Syst.}, vol. 30, no. 3, pp. 687--700, Mar. 2022, doi: 10.1109/TFUZZ.2020.3044253.
\bibitem{zenian} S. Zenian, T. Ahmad, and A. Idris, ``The determination of hesitation value for Sugeno type intuitionistic fuzzy generator via fuzzy limit,'' \emph{Int. J. Pure Appl. Math.}, vol. 104, no. 2, pp. 249--264, 2015, doi: 10.12732/ijpam.v104i2.9.
\bibitem{sextractor} SExtractor documentation, ``Modeling the background.'' [Online]. Available: \url{https://sextractor.readthedocs.io/en/stable/Background.html}. Accessed: Oct. 26, 2026.
\bibitem{photutils} Photutils documentation, ``SourceFinder.'' [Online]. Available: \url{https://photutils.readthedocs.io/en/stable/api/photutils.segmentation.SourceFinder.html}. Accessed: Oct. 26, 2026.
\bibitem{ap} Scikit-learn documentation, ``average\_precision\_score.'' [Online]. Available: \url{https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html}. Accessed: Oct. 26, 2026.
\bibitem{hubble} Scikit-image documentation, ``skimage.data: hubble\_deep\_field.'' [Online]. Available: \url{https://scikit-image.org/docs/stable/api/skimage.data.html}. Accessed: Oct. 26, 2026.
\end{thebibliography}
\end{document}

```

## pyproject.toml

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
[project]
name = "astrofuzzy"
version = "0.1.0"
description = "Reproducible additive sky-background fuzzy segmentation experiments"
requires-python = ">=3.9"
dependencies = ["numpy>=1.24", "scipy>=1.10", "scikit-learn>=1.3", "scikit-image>=0.22", "matplotlib>=3.7"]
[project.optional-dependencies]
test = ["pytest>=7"]
[project.scripts]
astrofuzzy = "astrofuzzy.experiments:main"
[tool.setuptools.packages.find]
include = ["astrofuzzy*"]

```

## tests/test_core.py

```python
import numpy as np
import pytest
from astrofuzzy.core import simulate, segment, sugeno, Config, design
from astrofuzzy.experiments import choose_threshold
from sklearn.metrics import average_precision_score


def test_sugeno_simplex():
    u = np.linspace(0,1,1001)
    nu, pi = sugeno(u,4)
    assert np.all(pi >= 0)
    np.testing.assert_allclose(u+nu+pi,1)
    with pytest.raises(ValueError):
        sugeno(u,-.1)


def test_scene_and_background_recovery():
    scene = simulate(42,shape=(64,64),sources=8)
    np.testing.assert_array_equal(scene['image'],simulate(42,shape=(64,64),sources=8)['image'])
    result = segment(scene['image'])
    np.testing.assert_allclose(result['membership'].sum(-1),1,atol=1e-12)
    assert np.isfinite(result['score']).all()
    assert np.sqrt(np.mean((result['background']-scene['background'])**2)) < 15


def test_constant_and_invalid():
    assert np.isfinite(segment(np.ones((16,16)))['score']).all()
    with pytest.raises(ValueError):
        segment(np.full((5,5),np.nan))
    with pytest.raises(ValueError):
        segment(np.ones((5,5)),config=Config(m=1))


def test_ap_rank_invariance_and_threshold():
    scene = simulate(7,shape=(32,32),sources=4)
    y, s = scene['truth'].ravel(), scene['image'].ravel()
    assert average_precision_score(y,s) == average_precision_score(y,3*s+2)
    assert np.isfinite(choose_threshold(y,s))


def test_polynomial_basis():
    assert design((8,9),2).shape == (72,6)


def test_zero_terms_reduce_to_corrected_fcm():
    scene = simulate(11, shape=(24,24), sources=3)
    config = Config(spatial=0, hesitation=0, iterations=20)
    base = segment(scene['image'], 'bc_fcm', config)
    full = segment(scene['image'], 'bc_ifcm', config)
    np.testing.assert_allclose(full['background'], base['background'])
    np.testing.assert_allclose(full['membership'], base['membership'])


def test_spatial_ranking_is_affine_invariant():
    scene = simulate(19, shape=(32,32), sources=4)
    base = segment(scene['image'], 'fcm')
    spatial = segment(scene['image'], 'spatial_fcm')
    truth = scene['truth'].ravel()
    assert average_precision_score(truth,base['score'].ravel()) == average_precision_score(truth,spatial['score'].ravel())

```