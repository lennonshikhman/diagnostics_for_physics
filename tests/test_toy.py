from diagnostic_suite.toy import make_periodic_advection


def test_toy_generator_shape_and_dtype():
    x = make_periodic_advection(n_traj=2, n_time=3, size=8, channels=2, seed=1)
    assert x.shape == (2, 3, 8, 8, 2)
    assert x.dtype.name == "float32"
