# This is needed as running all tests together now causes the test to fail
# when the same doesn't happen when running tests separately.
# The Error is the same: AttributeError: module 'numpyro.infer' has no
# attribute 'initialization' in arviz_base.io_numpyro

# This may have something to do with the lazy loading used in arviz_base.io_numpyro,
# in accordance with import ordering and different behaviors when using from-imports.
# However, I'm unable to recreate it using a minimal example. This at least enforces
# the numpyro.infer to have the initialization attribute.
import numpyro.infer.initialization  # noqa: F401
