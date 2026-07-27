from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='snudd',
    url='https://github.com/SNuDD/SNuDD.git',
    version="0.1",
    author='Dorian Amaral, David Cerdeno, Andrew Cheek, Patrick Foldenauer',
    packages=['snudd'],
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)
