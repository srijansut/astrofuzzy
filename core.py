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
