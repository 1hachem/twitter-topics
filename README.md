

# Twitter Topic Classification
This repository contains the code for a project that aims to classify tweets into different topics using embeddings obtained from pre-trained language models. The project includes data preprocessing, embedding extraction, model training, and evaluation.

## Getting started

### Dependencies

- Python 3.10.*
- PyTorch
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn

To install the dependencies, run:

```bash
pip install .
pip install -r requirements.txt
```

## Data
The raw data used in this project is a pickled file containing tweets with their associated topics.

## Preprocessing
The preprocessing step includes removing irrelevant features from the data and one-hot encoding the topics. The processed data is then embedded using pre-trained language models.

## Models
The model used for this task is a feed-forward neural network. The input to the model is the embedding, and the output is a binary vector indicating the presence or absence of each topic.

## Evaluation
The model is evaluated using precision, recall, and F1-score metrics. The evaluation results are reported for the validation and test sets.

## File Structure
The project has the following file structure:

```bash
├── notebooks
│   ├── main.ipynb
├── scripts
│   ├── fine_tuning.sh
│   ├── inference.sh
│   └── prepare_dataset.sh
├── src
│   ├── __init__.py
│   ├── evaluation.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
├── tests
│   └── test.py
├── Makefile
├── README.md
├── requirements.txt
├── .env
└── setup.py
```

## Usage
Clone the repository.

```bash
git clone https://github.com/1hachem/twitter-topics.git
Download the raw data file and place it in the data/raw directory.
```
Add your `OPENAI_API_KEY` to `.env`

```
OPENAI_API_KEY=sk- ...
```

Run `notebooks/main.ipynb`

### Fine-tuning 

## License
This project is licensed under the MIT License - see the LICENSE file for details.