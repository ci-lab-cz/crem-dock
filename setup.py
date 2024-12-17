import setuptools
from os import path
import cremdock

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name="cremdock",
    version=cremdock.__version__,
    author="Guzel Minibaeva, Pavel Polishchuk",
    author_email="pavel_polishchuk@ukr.net",
    description="CReM-dock: molecule generation and decoration guided by molecular docking",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DrrDom/cremdock",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry"
    ],
    python_requires='>=3.9',
    extras_require={
        'rdkit': ['rdkit>=2017.09'],
        'crem': ['crem>=0.2.7'],
        'easydock': ['easydock>=0.3.1'],
    },
    entry_points={'console_scripts':
                      ['cremdock = cremdock.cremdock:entry_point']},
    scripts=[]
)
