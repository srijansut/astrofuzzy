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
