from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    readme = fh.read()

requirements = {"install": ["ffmpeg", "natsort", "numpy", "tqdm", "torch", "torchvision",
                            "tensorboard", "coloredlogs", "petname", "pandas", "ray", "setproctitle",
                            "multiprocess", "python-box"]}

install_requires = requirements["install"]

setup(
    # Metadata
    name="torch-lighter",
    author="Marius-Constantin Dinu",
    version="0.2.19",
    author_email="dinu.marius-constantin@hotmail.com",
    url="https://github.com/Xpitfire/lighter",
    scripts=['bin/lighter-init'],
    description="Lightweight extension for torch to speed-up prototyping.",
    long_description=readme,
    long_description_content_type="text/markdown",
    # Package info
    packages=find_packages(),
    zip_safe=True,
    install_requires=install_requires,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent"
    ]
)
