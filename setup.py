from distutils.core import setup

setup(
    name="NonlinearOptControl",
    packages=['nopt'],
    install_requires=['jax','matplotlib','scipy']
)
