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

**All scripts to run the source codes are in `script/`.**
**The file names of the scripts are `(directory name)_(file name).sh`, respectively.**

Before you start, you need to download the annotated data, [FrameNet](https://framenet.icsi.berkeley.edu/fndrupal/framenet_request_data) and [PropBank](https://github.com/propbank/propbank-release/tree/master/data/ontonotes).
Note the file name if you use the source code directly.
[FrameNet](https://www.nltk.org/howto/framenet.html) and [PropBank](https://www.nltk.org/howto/propbank.html) can also be downloaded from the [NLTK](https://www.nltk.org/index.html) library, but they differ from the code we used and require careful preprocessing.

### 1. Preprocessing (`preprocessing/`)

You extract examples, Lexical Units, frames, etc. from XML files for FrameNet (`extract_exemplars_framenet.py`) and PropBank (`extract_exemplars_propbank.py`).
See `script/preprocessing_extract_exemplars_*.sh` when running these.

In addition, frame-to-frame relationship data used in the experiment is extracted from the XML file in FrameNet (`make_relation_list.py`).

### 2. Experiment on Frame Distinction (`experiment_frame_distinction/`)

First, you need to make datasets for this experiment (`make_dataset.py`).
Next, the contextualized wordembeddings of the target verbs are obtained (`get_embeddings.py`). 
The use of GPUs is recommended here.
Then, frame distinction can be performed by clustering on the basis of the embeddings (`verb_sense_clustering.py`).

You can aggregate results by focusing on FrameNet frame-to-frame relationships (`aggregate_relations.py`).
You can also visualize the contextualized word embedding of the target verb in two dimensions (`visualize_embeddings.py`).

### 3. Experiment on Frame Number Estimation (`experiment_frame_number_estimation/`)

First, you need to make datasets for this experiment (`make_dataset.py`).
Next, the contextualized word embeddings of the target verbs are obtained (`get_embeddings.py`). 
The use of GPUs is recommended here.
Then, frame number estimation can be performed by clustering on the basis of the embeddings (`verb_sense_clustering.py`).

## Citation

Please cite our paper if this source code is helpful in your work.
```bibtex
@inproceedings{yamada-etal-2021-verb,
    title = "Verb Sense Clustering using Contextualized Word Representations for Semantic Frame Induction",
    author = "Yamada, Kosuke  and
      Sasano, Ryohei  and
      Takeda, Koichi",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    year = "2021",
    url = "https://aclanthology.org/2021.findings-acl.381",
    pages = "4353--4362",
}
```








