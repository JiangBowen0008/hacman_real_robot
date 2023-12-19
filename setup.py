from setuptools import setup, find_packages

# # Read the contents of requirements.txt
# with open('requirements.txt') as f:
#     required_packages = f.read().splitlines()

# # Additional dependencies
# dependency_links = [
#     'https://download.pytorch.org/whl/cu113',
#     'https://data.pyg.org/whl/torch-1.11.0+cu113.html'
# ]

setup(
    name='HACManRealEnv',
    version='0.1dev',
    packages=find_packages(),
    # install_requires=required_packages + extra_packages,
    # dependency_links=dependency_links,
    license='MIT License',
    long_description='Learning Hybrid Actor-Critic Maps for 6D Non-Prehensile Manipulation',
)