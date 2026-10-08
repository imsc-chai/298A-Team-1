from p17sim.rng import derive_seed, rng_for


def test_derive_seed_is_stable_and_name_sensitive():
    assert derive_seed(1, "p0000", "facts") == derive_seed(1, "p0000", "facts")
    assert derive_seed(1, "p0000", "facts") != derive_seed(1, "p0000", "routines")


def test_rng_streams_repeat():
    assert rng_for(1, "a").integers(0, 10**9) == rng_for(1, "a").integers(0, 10**9)
