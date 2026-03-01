from src.preprocessing import (
    load_data,
    create_binary_columns,
    create_text_features,
)
from src.models import train_baseline_model
from src.fairness_metrics import (
    demographic_parity_difference,
    equal_opportunity_difference,
)
from src.evaluation import evaluate_performance
from src.models import train_svm_model
from src.traditional_models import train_word2vec_model
from src.traditional_models import train_glove_model
from src.transformer_models import train_bert_model

def main():
    # Load data
    df = load_data("data/adult.csv")

    # Preprocess
    df = create_binary_columns(df)
    df = create_text_features(df)

    # Train baseline model
    model, X_test, y_test, y_pred = train_baseline_model(df)

    # Performance evaluation
    performance = evaluate_performance(y_test, y_pred)
    print("\n=== Baseline Performance ===")
    for k, v in performance.items():
        print(f"{k}: {v:.4f}")

    # Fairness evaluation
    A_gender = df.loc[X_test.index, "gender_binary"]
    A_age = df.loc[X_test.index, "age_old_binary"]

    gender_dpd = demographic_parity_difference(y_pred, A_gender)
    gender_eod = equal_opportunity_difference(y_test, y_pred, A_gender)

    age_dpd = demographic_parity_difference(y_pred, A_age)
    age_eod = equal_opportunity_difference(y_test, y_pred, A_age)

    print("\n=== Baseline Fairness ===")
    print(f"Gender DPD: {gender_dpd:.4f}")
    print(f"Gender EOD: {gender_eod:.4f}")
    print(f"Age DPD: {age_dpd:.4f}")
    print(f"Age EOD: {age_eod:.4f}")

    #------------------------Objective 2------------------#
    # =============================
    # SVM MODEL
    # =============================

    svm_model, svm_X_test, svm_y_test, svm_y_pred = train_svm_model(df)

    svm_perf = evaluate_performance(svm_y_test, svm_y_pred)

    print("\n=== SVM Performance ===")
    for k, v in svm_perf.items():
        print(f"{k}: {v:.4f}")

    svm_A_gender = df.loc[svm_X_test.index, "gender_binary"]
    svm_A_age = df.loc[svm_X_test.index, "age_old_binary"]

    print("\n=== SVM Fairness ===")
    print("Gender DPD:", round(demographic_parity_difference(svm_y_pred, svm_A_gender), 4))
    print("Gender EOD:", round(equal_opportunity_difference(svm_y_test, svm_y_pred, svm_A_gender), 4))
    print("Age DPD:", round(demographic_parity_difference(svm_y_pred, svm_A_age), 4))
    print("Age EOD:", round(equal_opportunity_difference(svm_y_test, svm_y_pred, svm_A_age), 4))

    # =============================
    # Word2Vec + Logistic Regression
    # =============================

    w2v_model, w2v_test_idx, w2v_y_test, w2v_y_pred = train_word2vec_model(df)

    w2v_perf = evaluate_performance(w2v_y_test, w2v_y_pred)

    print("\n=== Word2Vec + LR Performance ===")
    for k, v in w2v_perf.items():
        print(f"{k}: {v:.4f}")

    w2v_A_gender = df.loc[w2v_test_idx, "gender_binary"]
    w2v_A_age = df.loc[w2v_test_idx, "age_old_binary"]

    print("\n=== Word2Vec + LR Fairness ===")
    print("Gender DPD:", round(demographic_parity_difference(w2v_y_pred, w2v_A_gender), 4))
    print("Gender EOD:", round(equal_opportunity_difference(w2v_y_test, w2v_y_pred, w2v_A_gender), 4))
    print("Age DPD:", round(demographic_parity_difference(w2v_y_pred, w2v_A_age), 4))
    print("Age EOD:", round(equal_opportunity_difference(w2v_y_test, w2v_y_pred, w2v_A_age), 4))

    # =============================
    # GloVe + Logistic Regression
    # =============================

    glove_model, glove_test_idx, glove_y_test, glove_y_pred = train_glove_model(df)

    glove_perf = evaluate_performance(glove_y_test, glove_y_pred)

    print("\n=== GloVe + LR Performance ===")
    for k, v in glove_perf.items():
        print(f"{k}: {v:.4f}")

    glove_A_gender = df.loc[glove_test_idx, "gender_binary"]
    glove_A_age = df.loc[glove_test_idx, "age_old_binary"]

    print("\n=== GloVe + LR Fairness ===")
    print("Gender DPD:", round(demographic_parity_difference(glove_y_pred, glove_A_gender), 4))
    print("Gender EOD:", round(equal_opportunity_difference(glove_y_test, glove_y_pred, glove_A_gender), 4))
    print("Age DPD:", round(demographic_parity_difference(glove_y_pred, glove_A_age), 4))
    print("Age EOD:", round(equal_opportunity_difference(glove_y_test, glove_y_pred, glove_A_age), 4))

    # =============================
    # BERT Transformer Model
    # =============================

    bert_model, bert_test_idx, bert_y_test, bert_y_pred = train_bert_model(
        df,
        epochs=2,
        batch_size=8
    )

    bert_perf = evaluate_performance(bert_y_test, bert_y_pred)

    print("\n=== BERT Performance ===")
    for k, v in bert_perf.items():
        print(f"{k}: {v:.4f}")

    bert_A_gender = df.loc[bert_test_idx, "gender_binary"]
    bert_A_age = df.loc[bert_test_idx, "age_old_binary"]

    print("\n=== BERT Fairness ===")
    print("Gender DPD:", round(demographic_parity_difference(bert_y_pred, bert_A_gender), 4))
    print("Gender EOD:", round(equal_opportunity_difference(bert_y_test, bert_y_pred, bert_A_gender), 4))
    print("Age DPD:", round(demographic_parity_difference(bert_y_pred, bert_A_age), 4))
    print("Age EOD:", round(equal_opportunity_difference(bert_y_test, bert_y_pred, bert_A_age), 4))

if __name__ == "__main__":
    main()
