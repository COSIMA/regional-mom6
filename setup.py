from setuptools import setup, find_packages

setup(
    name="regional-mom6",
    description="Automatic generation of regional configurations for Modular Ocean Model 6",
    author="Ashley Barnes and contributors",
    use_scm_version=True,
    setup_requires=["setuptools_scm", "setuptools_scm_git_archive"],
    packages=find_packages(),
    extras_require={"build": ["pytest", "findiff"]},
    install_requires=[
        "dask[array]",
        "h5py",
        "numpy>=1.17.0",
        "scipy>=1.2.0",
        "xarray",
        "netCDF4",
    ],
)
