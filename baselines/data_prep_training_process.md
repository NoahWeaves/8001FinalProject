# End-to-end data loading & preprocessing (and why it’s leak-free)

Below is a walkthrough of what the pipeline does to the data, from raw CSVs to train/test files, with notes on how each choice prevents data leakage. A sample is referenced to make the abstract steps concrete.

---

## 1) Configuration & logging

* The run fixes core knobs: `MAX_SAMPLES=10,000`, `MIN_CLASS_SAMPLES=4,000`, and **per-file sampling** (not global), then creates a timestamped output directory and sets up a logger. This ensures deterministic behavior and reproducible artifacts.  

---

## 2) Load each CSV and normalize columns

* All CSVs under `data/train/*.csv` are enumerated and read. Column names are stripped of stray whitespace, and a helper column **`Scenario`** is added from the filename (used later only for metadata/analysis, not as a model feature).  

* A dataset-specific cleanup removes the `WebDDoS` rows from `UDPLag.csv` up front (as shown in your log), so downstream sampling/metrics don’t get confounded by an extra class. 

**Data-leakage stance:** at this point we’re only reading files and doing lossless normalization. No targets/labels are used to transform features.

---

## 3) Per-file stratified downsampling (capacity control, class balance)

* Because original files are huge and extremely imbalanced (e.g., `DrDoS_LDAP` ~99.9%), each file is **independently** reduced via `sample_data_stratified` to at most `MAX_SAMPLES` while guaranteeing a **minimum per-class** count of `MIN_CLASS_SAMPLES`. The sampling plan is computed **per class** using the file’s own class distribution and a global seed.   

* The sampler:

  * prints the pre-sampling class histogram,
  * decides per-class quotas,
  * samples **within each class** with a fixed `random_state`,
  * concatenates and shuffles the result.   

Log entries showing “Class distribution before sampling / Sampling plan / Class distribution after sampling” for each file (e.g., DrDoS_LDAP, DrDoS_MSSQL, etc.) are produced by this exact function and the per-file branch of `load_data`. The **per-file** strategy is why the log repeatedly reports “Sampling strategy: 10,000 samples per file” and then prints a plan for that file.  

**Leakage stance:** sampling only *selects rows*; it does not compute statistics that will later be reused for transformation. It also works class-by-class using only the label of that row, which is fine (we must know labels to enforce stratification). No feature statistics are learned here.

---

## 4) Concatenate sampled files and drop identifier columns

* After all files are sampled, they’re concatenated into one DataFrame. 

* Columns that are **identifiers or protocol-level artifacts** (which risk target leakage or trivial memorization) are **dropped**:
  `["Unnamed: 0", "Flow ID", "Source IP", "Destination IP", "SimillarHTTP", "Inbound"]`. This is intentionally conservative—e.g., IP addresses can encode the label almost directly in a closed capture. 

* Infinities are replaced by NaNs (which will later be imputed). 

**Leakage stance:** removing direct identifiers and quasi-identifiers is a primary guardrail against target leakage and data snooping (e.g., the model memorizing that “this IP pair is always attack X”).

---

## 5) Timestamp parsing and **group** construction (for leak-free splitting)

* The `Timestamp` column is parsed into datetimes, then **coarsened to minute resolution** to build a **`groups`** vector. Each row belongs to the minute in which it occurred. 

* Features/target separation:

  * **X** = all features **excluding** `Label`, `Timestamp`, and `Scenario`.
  * **y** = `Label`.
    That means time and scenario metadata never enter the model as features. 

**Leakage stance:** timestamps are used **only** to define groups (splitting units), not as features; the `Scenario` tag is also excluded from X. This prevents the model from keying on capture windows or scenario IDs.

---

## 6) Label encoding (saved for reproducibility)

* Labels are integer-encoded once (to a vocabulary mapping like `{'BENIGN':0, 'DrDoS_LDAP':1, ...}`), and the fitted encoder is saved to disk. This affects *only* `y`; X is untouched. 

**Leakage stance:** using the whole label vocabulary to build a **name→index** mapping does not leak feature information or numeric statistics. It simply ensures consistent label indices across train/test/eval artifacts.

---

## 7) **Group-based train/test split** (temporal blocks, no overlap)

* With groups available, the code selects **~20% of the unique minute-groups** as a hold-out set. Every row whose timestamp falls in those minutes goes to **test**; the rest go to **train**. This enforces **strict disjointness in time** between train and test. 

* The log lines
  `Train set: 53,403 samples / Test set: 8,777 samples`
  correspond to this group-wise split over your 62,180 sampled rows.

**Leakage stance (critical):** samples from the same minute never appear in both splits. That blocks classic temporal leakage (look-ahead) and correlation leakage (near-duplicate flows split across folds).

---

## 8) Pipeline: impute → scale → univariate MI selection → model (CV-safe)

* The model is wrapped in a scikit-learn **`Pipeline`** that does:

  1. **Imputation** (`median`) to fill NaNs,
  2. **Scaling** (`StandardScaler` by default),
  3. **Feature selection** with **`SelectKBest(mutual_info_classif)`**, where **k** is tuned,
  4. The **classifier** (MLP CPU or GPU via skorch).  

* Because these steps live **inside** the Pipeline, and the Pipeline is fitted **inside cross-validation**, all statistics (medians, scaling parameters, MI scores, and the selected feature set) are **learned only on the training fold** and then applied to its validation fold. There’s no peeking at validation/test data when computing preprocessing parameters.

**Leakage stance:** this is the standard, recommended pattern to avoid preprocessing leakage. No global imputer/scaler/selector is fit on the entire dataset.

---

## 9) Cross-validation & tuning honor the **group structure**

* Hyperparameter search uses either **GroupKFold** when groups are present, or stratified K-fold otherwise. In your run, groups exist, so the CV iterator is **GroupKFold**—ensuring CV folds don’t mix minutes either. The `groups` array is passed to `fit()` so the split logic is respected during tuning.   

**Leakage stance:** even during tuning, validation folds are temporally disjoint from training folds. Preprocessing remains fold-local (via Pipeline), so CV metrics aren’t inflated.

---

## 10) Class imbalance handled **from training labels only**

* After the split, class weights are computed from **`y_train`** frequencies (not from the whole dataset) and (if using the GPU MLP) passed to the loss as `criterion__weight`.  

**Leakage stance:** weighting only uses training labels, avoiding any subtle signal from test distribution.

---

## 11) Persisted artifacts

* The pipeline saves: the label encoder, CV results and the tuned estimator, and the hold-out predictions/metrics/plots. Your separate data-prep phase also saved `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`, and `groups_*.csv` along with `metadata.json` and a run log under a timestamped directory (as shown in your log). The saving/plotting functions operate **after** the split and contain no transforms that could feed back into training.   

---

## 12) LLM fine-tuning on data_processed (separate script)

- Inputs: data_processed/*_train.csv for training, data_processed/*_test.csv for final evaluation. These splits are pre-defined; no CV is used for the LLM track.
- Feature policy: drop identifiers (`Unnamed: 0`, `Flow ID`, `Source IP`, `Destination IP`, `SimillarHTTP`, `Inbound`) and exclude `Timestamp`/`Scenario` from features. Replace ±inf with NaN.
- Text serialization: remaining feature columns are serialized into a compact text prompt (e.g., `key=value` lines). Labels are mapped via a `label2id` built only from training labels.
- Models: Llama-3.2-1B and Llama-3.2-3B with a sequence-classification head, trained via Transformers Trainer + PEFT (LoRA/QLoRA). Tokenizer pad token is set to EOS if missing.
- Evaluation: run inference on the test split and compute the same metrics as baseline (Accuracy, F1_macro/weighted, Precision_macro/weighted, Recall_macro/weighted), plus confusion matrix and classification report. Artifacts are saved to runs/llm/classif_YYYYMMDD_HHMMSS/{model_name}/.

Leakage stance for LLM path:
1) Train/test are fixed; the label map is derived from training labels only.
2) Identifiers and time/scenario fields are excluded from features before text serialization.
3) Test is used strictly for final inference/metrics; no test-derived statistics are used during training.

---

## Why there’s **no data leakage** (checklist)

1. **Temporal separation:** Train/test and CV folds are split by **minute-groups**, so no example shares a time window across splits.  

2. **No timestamp/scenario features:** `Timestamp` and `Scenario` are **dropped** from X; they’re used only for grouping/metadata. 

3. **Identifiers removed:** `Flow ID`, IPs, etc., are excluded from X to prevent trivial memorization or implicit label leakage. 

4. **Preprocessing fit only on training folds:** Imputation, scaling, and MI-based feature selection live **inside** an sklearn Pipeline and are fit **within** CV folds. No fold ever sees validation/test statistics during transform.  

5. **Hyperparameter tuning respects groups:** GroupKFold is used for CV when groups exist, passed correctly to `.fit(...)`.  

6. **Class weights from training only:** Derived from `y_train`, not from all labels. 

7. **Label encoding is safe:** The encoder maps class names → integers; it does not use feature statistics and does not alter features. Using the full label set to build a codebook is standard and does not inflate performance. 

---

## Mapping the log to the steps

* “Sampling strategy: 10,000 samples per file … Class distribution before/after sampling”: **§3 Per-file stratified downsampling**.   
* “Loaded … UDPLag.csv (WebDDoS removed)”: **§2 Load & normalize**. 
* “Data loaded: 62,180 samples, 80 features”: **§5 Timestamp & features/target**. 
* “Label mapping: {…}; Number of classes: 7”: **§6 Label encoding**. 
* “Performing group-based train/test split… Train: 53,403; Test: 8,777”: **§7 Group split**. 
* “Files saved: X_train.csv, … groups_train.csv, … label_encoder.joblib, metadata.json”: **§11 Persisted artifacts** (plus your preparer script). 

---

## Summary 

We load each traffic CSV, strip column names, purge `WebDDoS` from UDPLag, and attach a non-feature `Scenario` tag. Each file is then independently stratified-sampled to at most 10k rows with a minimum of 4k per class to control compute while preserving within-file label balance. After concatenation, strong identifiers (`Flow ID`, IPs, etc.) are dropped; infinities become NaNs. We parse `Timestamp` and form minute-level **groups**, then define **X** by removing `Label`, `Timestamp`, and `Scenario`. Labels are integer-encoded and saved. We perform a **group-based holdout split** (≈20% of minutes entirely in test) to ensure temporal disjointness. Training uses an sklearn **Pipeline**—median imputation → standardization → `SelectKBest` (mutual information) → classifier—so all preprocessing is fit **within CV folds** using **GroupKFold**, preventing information bleed. Class weights are computed from `y_train` only. Artifacts (label encoder, CV results, tuned model, holdout predictions/metrics) are saved to a timestamped directory. This design prevents data leakage by (i) time-blocking train/test and CV folds, (ii) excluding timestamps/scenario/identifiers from features, and (iii) fitting all preprocessing strictly on training partitions.     

---
