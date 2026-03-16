from setuptools import setup, find_packages

setup(
    name="hyplan",
    use_scm_version=True,
    setup_requires=["setuptools-scm"],
    description="Planning software for airborne remote sensing science campaigns",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Ryan Pavlick",
    author_email="ryan.p.pavlick@nasa.gov",
    url="https://github.com/ryanpavlick/hyplan",
    packages=find_packages(),
    install_requires=open("requirements.txt").read().splitlines(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License 2.0",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
