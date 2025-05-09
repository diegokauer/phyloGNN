from setuptools import setup, find_packages

setup(
    name="phylo_gnn",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Add your dependencies here, e.g.,
        'torch',
        'torch_geometric',
        'dotenv',
        'pandas',
        'ete3',
        'networkx',
        'numpy',
        'composition_stats'
    ],
    # entry_points={
    #     'console_scripts': [
    #         'mirai-chile-predict = mirai_chile.script:main',
    #     ]
    # }
)