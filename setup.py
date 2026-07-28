from setuptools import setup, find_packages

setup(
    name='snudd',
    url='https://github.com/SNuDD/SNuDD.git',
    version="1.0.0",
    author='Dorian Amaral, David Cerdeno, Andrew Cheek, Valeria Costa, Patrick Foldenauer',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "snudd": ["data/*", "data/**/*"],
        },
)