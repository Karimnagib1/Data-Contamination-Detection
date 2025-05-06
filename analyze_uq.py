import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score
import json
from cliffs_delta import cliffs_delta

# ==== 1. Define the paths to contaminated and uncontaminated scores' json files
contaminated_scores = "uq_results/uncontrolled/uq_scores_mintaka_contaminated_gpt_neo_both.json"
uncontaminated_scores = "uq_results/uncontrolled/uq_scores_mintaka_uncontaminated_gpt_neo_both.json"

# Load contaminated scores from JSON
with open(contaminated_scores, "r") as file:
    c = json.load(file)
    c = np.array(c)

# Load uncontaminated scores from JSON
with open(uncontaminated_scores, "r") as file:
    u = json.load(file)
    u = np.array(u)

# ==== 2. Paired statistical tests ====
wilcoxon_stat, wilcoxon_p = stats.wilcoxon(c, u)
ttest_stat, ttest_p = stats.ttest_rel(c, u)

# ==== 3. Effect sizes ====
d = c - u
cohen_dz = d.mean() / d.std(ddof=1)
cliffs_delta, res = cliffs_delta(u, c)


# ==== 4. Overlap coefficient ====
x = np.linspace(min(u.min(), c.min()), max(u.max(), c.max()), 1000)
pdf_u = stats.gaussian_kde(u)(x)
pdf_c = stats.gaussian_kde(c)(x)
ovl = np.trapz(np.minimum(pdf_u, pdf_c), x)

# ==== 5. ROC-AUC and best threshold ====
scores = np.concatenate([u, c])
labels = np.concatenate([np.zeros_like(u), np.ones_like(c)])  # 0 = clean, 1 = contaminated
fpr, tpr, thresholds = roc_curve(labels, scores)
auc = roc_auc_score(labels, scores)

youden_index = np.argmax(tpr - fpr)
best_threshold = thresholds[youden_index]
predictions = (scores >= best_threshold).astype(int)
cm = confusion_matrix(labels, predictions)
accuracy = accuracy_score(labels, predictions)

mean_u, std_u = u.mean(), u.std()
mean_c, std_c = c.mean(), c.std()


print("\n==== Distribution Summary ====")
print(f"Uncontaminated model: mean = {mean_u:.4f}, std = {std_u:.4f}")
print(f"Contaminated model:   mean = {mean_c:.4f}, std = {std_c:.4f}")

print("\n==== Statistical Tests ====")
print(f"Wilcoxon signed-rank p-value: {wilcoxon_p:.4f}")
print(f"Paired t-test p-value:         {ttest_p:.4f}")

print("\n==== Effect Sizes ====")
print(f"Cohen's dz (paired):           {cohen_dz:.4f}")
print(f"Cliff's delta:                 {cliffs_delta:.4f}")
print(f"Cliff's res:                   {res}")

print("\n==== Distribution Overlap ====")
print(f"Overlap Coefficient (OVL):     {ovl:.4f}  (1 = total overlap)")

print("\n==== ROC Analysis ====")
print(f"ROC AUC:                       {auc:.4f}")
print(f"Best threshold (Youden's J):   {best_threshold:.4f}")
print(f"Confusion matrix:\n{cm}")
print(f"Accuracy at threshold:         {accuracy:.4f}")

# # ==== 7. Optional: plot distributions ====
# plt.figure(figsize=(7,4))
# plt.title("Uncertainty Score Distributions")
# plt.xlabel("Uncertainty score")
# plt.ylabel("Density")
# plt.plot(x, pdf_u, label="Uncontaminated", color='blue')
# plt.plot(x, pdf_c, label="Contaminated", color='red')
# plt.fill_between(x, np.minimum(pdf_u, pdf_c), alpha=0.3, color='gray', label=f"Overlap = {ovl:.2f}")
# plt.axvline(best_threshold, color='green', linestyle='--', label=f"Threshold = {best_threshold:.2f}")
# plt.legend()
# plt.tight_layout()
# plt.show()
