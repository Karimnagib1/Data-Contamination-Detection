import pandas as pd

# # Similarity scores based on controlled mintaka generations

# gpt2_mintaka_solo = pd.read_parquet('controlled_metric_scores/generated_texts_gpt2_large_mintaka_solo_scores.parquet')
# gpt2_mintaka = pd.read_parquet('controlled_metric_scores/generated_texts_gpt2_large_mintaka_scores.parquet')
# gpt_neo_mintaka = pd.read_parquet('controlled_metric_scores/generated_texts_gptneo_mintaka_scores.parquet')
# gpt_neo_mintaka_solo = pd.read_parquet('controlled_metric_scores/generated_texts_gptneo_mintaka_solo_scores.parquet')


# Similarity scores based on uncontrolled mintaka generations
gpt2_mintaka_solo = pd.read_parquet('metric_scores/generated_texts_gpt2_large_mintaka_solo_scores.parquet')
gpt2_mintaka = pd.read_parquet('metric_scores/generated_texts_gpt2_large_mintaka_scores.parquet')
gpt_neo_mintaka = pd.read_parquet('metric_scores/generated_texts_gptneo_mintaka_scores.parquet')
gpt_neo_mintaka_solo = pd.read_parquet('metric_scores/generated_texts_gptneo_mintaka_solo_scores.parquet')

# Similarity scores based on wikimia generations
gpt2_wikimia_solo = pd.read_parquet('metric_scores/generated_texts_gpt2_large_wikimia_solo_scores.parquet')
gpt2_wikimia = pd.read_parquet('metric_scores/generated_texts_gpt2_large_wikimia_scores.parquet')
gpt_neo_wikimia = pd.read_parquet('metric_scores/generated_texts_gptneo_wikimia_scores.parquet')
gpt_neo_wikimia_solo = pd.read_parquet('metric_scores/generated_texts_gptneo_wikimia_solo_scores.parquet')


heatmap_path = "f1_score_results/linear_SVM_heatmap_controlled.png"
f1_scores_path = "f1_score_results/linear_SVM_f1_scores_controlled.csv"

dataframes = {
    "gpt2_mintaka": gpt2_mintaka_solo,
    "gpt2_mintaka_both": gpt2_mintaka,
    "gpt_neo_mintaka_both": gpt_neo_mintaka,
    "gpt_neo_mintaka": gpt_neo_mintaka_solo,
    "gpt2_wikimia": gpt2_wikimia_solo,
    "gpt2_wikimia_both": gpt2_wikimia,
    "gpt_neo_wikimia_both": gpt_neo_wikimia,
    "gpt_neo_wikimia": gpt_neo_wikimia_solo
}


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def train_logistic_regression_with_report(df):
    # Define features and target
    features = [
        'bertscore_precision', 'bertscore_recall', 'bertscore_f1',
        'rouge1', 'rouge2', 'rouge3', 'rougeL', 'bleu', 'nli_score'
    ]
    target = 'use_for_contamination'

    # Split into train/test with stratification
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, stratify=y, random_state=42,
    )

    # Create a pipeline with standard scaler and logistic regression
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    )

    # Train the model
    model.fit(X_train, y_train)

    # Predict and print classification report
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Return the trained model
    return model

lr_gpt2_mintaka_solo = train_logistic_regression_with_report(gpt2_mintaka_solo)

lr_gpt2_mintaka_solo[1].coef_


features = [
        'bertscore_precision', 'bertscore_recall', 'bertscore_f1',
        'rouge1', 'rouge2', 'rouge3', 'rougeL', 'bleu', 'nli_score'
    ]
target = 'use_for_contamination'
pred = lr_gpt2_mintaka_solo.predict(gpt_neo_mintaka[features])
print(classification_report(gpt_neo_mintaka[target], pred))


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def train_logistic_regression_with_report(df, features=None, target_column=None):
    # Default feature list
    if features is None:
        features = [
            'bertscore_precision', 'bertscore_recall', 'bertscore_f1',
            'rouge1', 'rouge2', 'rouge3', 'rougeL', 'bleu', 'nli_score'
        ]

    # Auto-detect target column if not specified
    if 'use_for_contamination' in df.columns:
        target_column = 'use_for_contamination'
    elif 'label' in df.columns:
        target_column = 'label'
    else:
        raise ValueError("Target column not found. Provide a target column or use a DataFrame with 'use_for_contamination' or 'label'.")

    # Prepare data
    X = df[features]
    y = df[target_column]

    # Split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, stratify=y, random_state=42
    )

    # Train model
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=42, max_iter=1000)
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print(f"Classification Report (target: {target_column}):")
    print(classification_report(y_test, y_pred))

    return model

def train_svm_with_report(df, features=None, target_column=None):
    if features is None:
        features = [
            'bertscore_precision', 'bertscore_recall', 'bertscore_f1',
            'rouge1', 'rouge2', 'rouge3', 'rougeL', 'bleu', 'nli_score'
        ]

    if not target_column:
        if 'use_for_contamination' in df.columns:
            target_column = 'use_for_contamination'
        elif 'label' in df.columns:
            target_column = 'label'
        else:
            raise ValueError("Target column not found.")

    X = df[features]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = make_pipeline(
        StandardScaler(),
        # SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
        # linear SVM
        SVC(kernel='linear', class_weight='balanced', probability=True, random_state=42)
        # SVC(kernel='poly', class_weight='balanced', probability=True, random_state=42)

    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Classification Report (target: {target_column}):")
    print(classification_report(y_test, y_pred))

    return model


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

def cross_predict_f1_heatmap(df_dict, features=None, target_column=None):
    datasets = list(df_dict.keys())
    f1_scores = pd.DataFrame(index=datasets, columns=datasets)

    for train_name in datasets:
        print(f"\nTraining on: {train_name}")
        # model = train_logistic_regression_with_report(df_dict[train_name], features, target_column)
        model = train_svm_with_report(df_dict[train_name])

        for test_name in datasets:
            df_test = df_dict[test_name]

            # Determine target
            # tgt_col = target_column or ('use_for_contamination' if 'use_for_contamination' in df_test.columns else 'label')

            if features is None:
                features = [
                    'bertscore_precision', 'bertscore_recall', 'bertscore_f1',
                    'rouge1', 'rouge2', 'rouge3', 'rougeL', 'bleu', 'nli_score'
                ]

            if 'use_for_contamination' in df_test.columns:
                target_column = 'use_for_contamination'
            elif 'label' in df_test.columns:
                target_column = 'label'
            else:
                raise ValueError("Target column not found.")


            X = df_test[features]
            y = df_test[target_column]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )

            # Predict
            y_pred = model.predict(X_test)

            # Compute macro f1 score
            f1 = f1_score(y_test, y_pred, average='macro')
            f1_scores.loc[train_name, test_name] = f1

    # Convert to float
    f1_scores = f1_scores.astype(float)

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(f1_scores, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Macro F1 Scores: Train (rows) vs Test (columns)")
    plt.xlabel("Test Setup")
    plt.ylabel("Train Setup")
    plt.tight_layout()
    plt.show()

    # Save the heatmap
    plt.savefig(heatmap_path, dpi=300)
    print(f"Heatmap saved to {heatmap_path}")
    # Save the F1 scores DataFrame
    f1_scores.to_csv(f1_scores_path)
    return f1_scores

res = cross_predict_f1_heatmap(dataframes)