from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    readme = fh.read()

requirements = {"install": ["ffmpeg", "natsort", "numpy", "tqdm", "torch", "torchvision",
                            "tensorboard", "coloredlogs", "petname", "pandas", "ray", "setproctitle",
                            "multiprocess"]}

install_requires = requirements["install"]

setup(
    # Metadata
    name="minerl",
    author="Your name",
    version="0.0.0",
    author_email="",
    url="https://path.to.repo",
    description="Description of your project.",
    long_description=readme,
    long_description_content_type="text/markdown",
    # Package info
    packages=find_packages(),
    zip_safe=True,
    install_requires=install_requires,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ]
)
