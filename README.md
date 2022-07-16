# vsc
The source code of paper: [Verb Sense Clustering using Contextualized Word Representations for Semantic Frame Induction](https://aclanthology.org/2021.findings-acl.381/), accepted to [ACL-IJCNLP Findings 2021](https://2021.aclweb.org/).

## Installation

```sh
# Before installation, upgrade pip and setuptools.
$ pip install -U pip setuptools

# Install other dependencies.
$ pip install -r requirements.txt

# Install the vsc package.
$ pip install .
# Or if you want to install it in editable mode:
$ pip install -e .
```

## Usage

Before you start, you need to download the annotated data, [FrameNet](https://framenet.icsi.berkeley.edu/fndrupal/framenet_request_data) and [PropBank](https://github.com/propbank/propbank-release/tree/master/data/ontonotes).
Note the file name if you use the source code directly.
[FrameNet](https://www.nltk.org/howto/framenet.html) and [PropBank](https://www.nltk.org/howto/propbank.html) can also be downloaded from the [NLTK](https://www.nltk.org/index.html) library, but they differ from the code we used and require careful preprocessing.

