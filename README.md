# Pre-trained Language Model Fine-tuning on Non-textual Data Understanding

<p align="center">
  <img src="./ㅁssets/Thesis_structure.png" width="850" alt="Project Structure: Fine-tuning Pre-trained Language Models on Non-textual Data">
</p>

<p align="center">
  <em>Overview of fine-tuning pre-trained language models on text and non-textual sequence data.</em>
</p>

---

## About This Project

This repository contains the implementation code for my undergraduate thesis:

> **Pre-trained Language Model Fine-tuning on Non-textual Data Understanding**

This project investigates whether pre-trained language models, originally developed for natural language processing, can be fine-tuned and transferred to non-textual sequential data such as **protein sequences**, **DNA sequences**, and **music symbolic data**.

The main idea is that non-textual data can also be represented as ordered token sequences. By applying pre-trained language models to these domains, this project analyzes how much knowledge or structural representation learned from text can be transferred to other types of sequential data.

---

## Research Background

This project was developed with reference to the prior work:

> **Is BERT a Cross-Disciplinary Knowledge Learner? A Surprising Finding of Pre-trained Models’ Transferability**
> Wei-Tsung Kao and Hung-yi Lee, Findings of EMNLP 2021

The prior study investigates whether pre-trained language models such as BERT can transfer their learned representations to general token sequence classification tasks, including amino acid, DNA, and music data.

This repository refers to the research direction and publicly available implementation resources provided by the paper, while organizing and adapting the experiments for my undergraduate thesis project.

* Paper: https://aclanthology.org/2021.findings-emnlp.189/
* ACL Anthology: Findings of EMNLP 2021
* DOI: https://doi.org/10.18653/v1/2021.findings-emnlp.189

```bibtex
@inproceedings{kao-lee-2021-bert-cross,
    title = "Is {BERT} a Cross-Disciplinary Knowledge Learner? A Surprising Finding of Pre-trained Models' Transferability",
    author = "Kao, Wei-Tsung and Lee, Hung-yi",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    year = "2021",
    publisher = "Association for Computational Linguistics",
    pages = "2195--2208",
    doi = "10.18653/v1/2021.findings-emnlp.189"
}
```

---

## Project Overview

Pre-trained language models such as BERT have shown strong performance in natural language understanding tasks. However, sequential data is not limited to human language. Biological sequences and music can also be represented as ordered token sequences.

This project focuses on the following research question:

> Can pre-trained language models trained on text data be effectively fine-tuned for non-textual sequence understanding?

The repository includes experiments on:

* **GLUE**: Natural language understanding benchmark
* **Protein Classification**: Protein sequence classification tasks
* **DNA Classification**: DNA sequence classification tasks
* **Music Data**: Symbolic music sequence data based on MAESTRO MIDI data

---

## Key Ideas

The main ideas of this project are:

* Fine-tuning pre-trained language models on non-textual sequential data
* Comparing pre-trained models with models trained from scratch
* Testing whether text-based pre-training provides useful representations for other token sequence domains
* Applying token shifting strategies to avoid unused token problems
* Evaluating model transferability across text, biological sequence, and music domains

---

## Repository Structure

```text
transferability/
├── assets/
│   └── thesis_structure.png
├── DNA/
│   ├── finetune.py
│   ├── evaluate.py
│   ├── preprocess.py
│   ├── preprocess_splice.py
│   ├── run_finetune.sh
│   └── run_evaluate.sh
├── Music/
│   ├── finetune.py
│   ├── evaluate.py
│   ├── preprocess.py
│   ├── run_finetune.sh
│   └── run_evaluate.sh
├── Protein/
│   ├── finetune.py
│   ├── evaluate.py
│   ├── save_feature.py
│   ├── run_finetune.sh
│   └── run_evaluate.sh
├── Dockerfile
├── requirements.txt
├── explanation.md
├── LICENSE
└── README.md
```

---

## Environment

This project is designed to be run in a Docker environment.

The Docker image is based on:

```dockerfile
FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime
```

The Dockerfile installs the required system packages, copies `requirements.txt`, and installs the Python dependencies inside the container.

### Dockerfile

```dockerfile
FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    libsndfile1 \
    fluidsynth \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

CMD ["bash"]
```

### Build Docker Image

Run the following command from the project root directory:

```bash
docker build -t transferability .
```

### Run Docker Container

To run the container with GPU support:

```bash
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  transferability
```

For Windows PowerShell, use:

```powershell
docker run --gpus all -it --rm `
  -v ${PWD}:/workspace `
  transferability
```

For Windows Command Prompt, use:

```cmd
docker run --gpus all -it --rm ^
  -v %cd%:/workspace ^
  transferability
```

After entering the container, you can run the experiment scripts from each directory:

```bash
cd Protein
bash run_finetune.sh
bash run_evaluate.sh
```

or:

```bash
cd DNA
bash run_finetune.sh
bash run_evaluate.sh
```

or:

```bash
cd Music
bash run_finetune.sh
bash run_evaluate.sh
```

### Notes

The project directory is mounted to `/workspace` inside the container:

```text
local project directory → /workspace
```

This means that code changes made on the host machine are immediately reflected inside the Docker container.

If GPU is not available, remove the `--gpus all` option:

```bash
docker run -it --rm \
  -v $(pwd):/workspace \
  transferability
```

## Data Preparation

### 1. GLUE

For GLUE, this project uses the dataset download script from the Hugging Face Transformers examples.

Please download the GLUE dataset using `download_glue_data.py` under the `utils` directory:

```bash
python utils/download_glue_data.py --data_dir /path/to/glue --tasks all
```

Then set the `GLUE_DIR` environment variable before running the experiment:

```bash
export GLUE_DIR=/path/to/glue
```

---

### 2. Protein Classification

Move to the `Protein` directory:

```bash
cd Protein
```

Prepare the protein classification dataset as follows:

1. Download the data from the PLUS project site:
   http://ailab.snu.ac.kr/PLUS/

2. Unzip the `.fa` files into the `data` directory.

3. Run `save_feature.py` to preprocess input features:

```bash
python save_feature.py \
  --task TASK_NAME \
  --model bert-base-uncased \
  --savedir preprocess_input/TASK_NAME/ \
  --split train
```

Main arguments:

```text
--task      Downstream task name
--model     Pre-trained model name, default: bert-base-uncased
--savedir   Directory to save preprocessed features
--split     Train/dev/test split to preprocess
```

---

### 3. DNA Classification

Move to the `DNA` directory:

```bash
cd DNA
```

Prepare the DNA classification dataset as follows:

1. Clone the Hilbert-CNN repository:

```bash
git clone https://github.com/Doulrs/Hilbert-CNN
```

2. Run preprocessing scripts:

```bash
python preprocess.py
python preprocess_splice.py
```

The input arguments are similar to those used in the protein preprocessing script.

---

### 4. Music

Move to the `Music` directory:

```bash
cd Music
```

Prepare the music dataset as follows:

1. Download the **MAESTRO-v1** dataset from:
   https://magenta.tensorflow.org/datasets/maestro#download

2. Download both MIDI files and metadata.

3. Unzip the dataset under the `Music` directory.

4. Run preprocessing:

```bash
python preprocess.py
```

The preprocessed data will be saved under:

```text
data/maestro-v1/
```

---

## Fine-tuning and Evaluation

### 1. GLUE

This project uses a modified version of the Hugging Face Transformers v3.1.0 library.
Please use the modified source code included in this project instead of the original Transformers package.

Additional arguments were added to `run_glue.py`:

```text
--pretrain_ckpt   Checkpoint used for experiments with different pre-training stages
--scratch         Train the model from scratch
--rand_embed      Use randomly initialized embeddings for ablation study
--shift           Constant c for the "shift c" setting
--random_shift    Use the random shift setting
```

Run fine-tuning and evaluation using the provided shell script:

```bash
source run_hf_glue.sh
```

> Use `source` instead of `sh`, because the script includes environment-related commands.

---

### 2. Protein Classification

The protein classification code is modified from the code provided by the PLUS project.

To fine-tune and evaluate the model:

```bash
cd Protein

bash run_finetune.sh
bash run_evaluate.sh
```

You can also run the scripts directly:

```bash
python finetune.py
python evaluate.py
```

For details about input arguments, refer to the help message:

```bash
python finetune.py --help
```

Important reproduction note:

```text
Use --shift or --shift_table with the shift table files under assign_token/
to avoid using unused tokens.
```

This setting is important for reproducing the reported performance.

Another implementation note:

```text
drop_last=True is used in the dataloader to prevent errors when the last batch
contains only one sample. It can be changed to False if the issue does not occur.
```

---

### 3. DNA Classification

To fine-tune and evaluate DNA classification models:

```bash
cd DNA

bash run_finetune.sh
bash run_evaluate.sh
```

Or run the Python scripts directly:

```bash
python finetune.py
python evaluate.py
```

The input arguments are similar to those used in the protein classification experiments.

---

### 4. Music

To fine-tune and evaluate the music experiments:

```bash
cd Music

bash run_finetune.sh
bash run_evaluate.sh
```

Or run:

```bash
python finetune.py
python evaluate.py
```

Important reproduction note:

```text
Use --shift or the shift_table file under the data directory
to avoid using unused tokens.
```

---

## Logging

All training and evaluation results are logged using **tensorboardX**.

Logged values include:

* Training loss
* Development loss
* Test loss
* Evaluation metrics

Evaluation results are logged at the time step corresponding to the fine-tuning steps.

To view logs with TensorBoard:

```bash
tensorboard --logdir log
```

---

## Notes on Reproducibility

Some experiments require specific preprocessing steps and token assignment strategies. In particular, the `--shift` and `--shift_table` options are important when applying pre-trained language models to non-textual data.

These options are used to avoid relying on unused tokens and to reproduce the transferability experiments more reliably.

---

## Thesis Context

This repository was created as part of my undergraduate thesis:

> **Pre-trained Language Model Fine-tuning on Non-textual Data Understanding**

The code is intended for research reproduction, experiment tracking, and further analysis of how language-model-based architectures can be adapted to structured non-textual sequential data.

---

## License

This project is licensed under the MIT License.
