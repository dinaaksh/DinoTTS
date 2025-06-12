import setuptools
__version__ = "0.0.0"

REPO_NAME = "vc"
AUTHOR_USER_NAME = "Dinaaksh Aulakh"
SRC_REPO = "core"
AUTHOR_EMAIL = "dinaaksh.aulakh@wittybrains.com"


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="package for voice generation",
    long_description="",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "."},
    packages=setuptools.find_packages(where="."),
    install_requires=[
        "coqui-tts==0.26.2",
        "torch",
        "together",
        "mlflow",
        "pyYAML",
        "python-box"
    ],
    include_package_data=True,
    python_requires=">=3.11",
)