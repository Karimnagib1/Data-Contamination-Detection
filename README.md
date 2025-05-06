
# Data Contamination Detection in Language Models

This repository accompanies our project report on detecting training data contamination in language models. We evaluate two main approaches: (1) an uncertainty-based method using Shifting Attention to Relevance (SAR), and (2) a similarity-based classifier trained on surface and semantic metrics.

---

## 🔧 Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
````

Login to Weights & Biases (WandB) for experiment tracking:

```bash
wandb login
# Paste your access token when prompted
```

---

## 🧪 Running Experiments

### 1. Supervised Fine-Tuning (SFT)

To train the models on contaminated subset of Mintaka dataset:

```bash
source train-sft.sh
```
Note: You can specify the base model in the bash script

---

### 2. Uncertainty-Based Detection (SAR)

Edit the following variables inside `sar.py`:

```python
model_name = "The model you want to get the scores for"
output_path = "Path to output JSON file"
dataset_name = "mintaka"  # or "wikimia"
contaminated = True       # Set to False for uncontaminated samples
```

Run:

```bash
python sar.py
```

---

### 3. Analyzing SAR Scores

To compare scores for contaminated vs. uncontaminated samples, update these variables in `analyze_uq.py`:

```python
contaminated_scores = "path/to/contaminated_scores.json"
uncontaminated_scores = "path/to/uncontaminated_scores.json"
```

Then run:

```bash
python analyze_uq.py
```

---

### 4. Classifier-Based Detection (Linear SVM)

To train SVM classifiers using metric-based similarity features:

1. Edit the following variables in `classify.py`:

```python
heatmap_path = "results/heatmap.png"
f1_scores_path = "results/f1_scores.csv"
```

2. To switch to length-controlled Mintaka generations, uncomment the corresponding lines in the script and comment out the uncontrolled ones.

Then run:

```bash
python classify.py
```
---

## 📚 References

```bibtex
@article{sen2022mintaka,
  title={Mintaka: A complex, natural, and multilingual dataset for end-to-end question answering},
  author={Sen, Priyanka and Aji, Alham Fikri and Saffari, Amir},
  journal={arXiv preprint arXiv:2210.01613},
  year={2022}
}

@article{duan2023shifting,
  title={Shifting attention to relevance: Towards the uncertainty estimation of large language models},
  author={Duan, Jinhao and Cheng, Hao and Wang, Shiqi and Zavalny, Alex and Wang, Chenan and Xu, Renjing and Kailkhura, Bhavya and Xu, Kaidi},
  year={2023}
}

```


<!-- 
## 📝 Notes

* SAR is an uncertainty-based method and requires access to model logits.
* The similarity-based classifier is fully black-box and uses metrics like BLEU, ROUGE, BERTScore, and NLI similarity.
* Length-controlled generation can improve in-domain accuracy but may reduce generalization.
* Min-\$k%\$ probability is also implemented as a white-box baseline.
 -->
