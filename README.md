

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
We run multiple experiments with different models :

### **Divide-and-conquer (one vs rest) feed-forward neural networks on top of ada-002**

We train multiple neural-networks each with the task to binary classify a specific label. 

Input : `ada-002 embeddings`, `dimension=1536`

Hidden layer : `dimension=16`

Activations : `middle : ReLU`, `final : Sigmoid` 

Loss used : `BCELoss`

Optimizer : `Adams`, `lr=3e-4`

> Due to the low number of sample for certain labels their classifiers under-perform, potential solution would be to balance the number of samples for each classifier individually (not train on the whole data).

### **Few-shot classification using GPT-3**
### **Fine-tuning GPT-3**
### **Divide-and-conquer with decision trees**

## Evaluation
The model is evaluated using precision, recall, and F1-score metrics. The evaluation results are reported for the validation and test sets.

## File Structure
The project has the following file structure:

```bash
├── data
│   ├── processed
│   └── raw
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
After saving the training data as a json file, run :
```bash
openai tools fine_tunes.prepare_data -f /path/to/json
```
example json file :

```json
[{"prompt":"tweet here", "completion":"topics here"}]
```
start fine-tuning your model

```bash
openai api fine_tunes.create -t -m ada --n_epochs 1
```

visualize your fine-tuning using wandb :

```bash
openai sync wandb
```


## License
This project is licensed under the MIT License - see the LICENSE file for details.