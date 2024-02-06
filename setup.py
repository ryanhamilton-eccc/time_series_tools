from setuptools import setup

setup(
    name='time_series_tools',
    version='0.1.0',
    package_dir={'': 'src'},
    author='Ryan Hamilton',
    author_email='ryan.hamilton@ec.gc.ca',
    install_requires=[
        'earthengine-api',
    ]
)