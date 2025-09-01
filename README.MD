# **Deep Learning Assignment 2: Automatic Image Captioning** (Spring 2025)

## Team Members ( TEAM ID 13)

| Name                     | Roll Number | Email                          |
|--------------------------|-------------|--------------------------------|
| Sai Deepak Reddy Mara    | 22CS10066   | saideepakmara@gmail.com        | 
| Byreddi Sri Chaitanya    | 22CS10018   | rockstarman957@gmail.com       |
| Angalakuduru Tejasree Sai| 22CS10009   | tejasree.angalakuduru@gmil.com |   


## Overview

Key tasks include:
1.  Building and training a custom ViT-GPT2 model for image captioning.
2.  Benchmarking this custom model against a pre-trained SmolVLM.
3.  Evaluating the robustness of both models on occluded images.
4.  Creating a BERT-based classifier to identify which model generated a given caption.

## Files Included

- `team_id_13_a.ipynb`
- `team_id_13_b.ipynb`
- `team_id_13_c.ipynb`
- `team_id_report.pdf`

## PART A

This part focuses on **Automatic Image Captioning** using:
- **Zero-shot captioning with SmolVLM**
- **A custom encoder-decoder model** (ViT encoder + transformer decoder)
- **Evaluation** using BLEU, ROUGE-L, and METEOR

## Notes

* **Training Note:** 
Implemented custom early stopping for the ViT-GPT2 model
    * The training stops if validation loss doesn't improve for 3 consecutive epochs, and the model with the best validation loss is saved.
* **Execution Note:** 
    * **GPU Constraints:** Running SmolVLM inference, custom model training, and custom model evaluation within a single session exceeded GPU memory limits.
    * **Training Workaround:** The custom model was trained by restarting session after running the initial SmolVLM evaluation to ensure sufficient GPU memory.
    * **METEOR Evaluation:** Calculating the METEOR score in Kaggle caused errors. The final evaluation results, including METEOR were obtained by running in Colab.

## Part B

* Implemented random patch masking (10%, 50%, 80% occlusion). 
* Evaluated both SmolVLM and the custom model on these perturbed images. 
* Generated dataset for part C `final_raw_results.csv`.

## Part C

* Built a dataset from Part B results. 
* Trained a BERT-based classifier (`bert-base-uncased`) to distinguish between SmolVLM and custom model captions.

* **Training Note:** 
   * The classifier was trained for a fixed number of epochs. 
   * Plots comparing training and validation loss were used after training to analyze performance and found that 5 epoch was optimal.

## Dataset

* PART A Dataset: [Dataset Link](https://drive.google.com/file/d/1-4zt018qT1M85m1X0v95C9-a6_6YelIQ/view?usp=drive_link)
* Generated data fron part B for part C: [Dataset(csv) Link](https://drive.google.com/file/d/11KSaN3uj2--eOkvLfMRtFDxxCioPHnVD/view?usp=drive_link)


## Models Used

* **SmolVLM:** (`HuggingFaceTB/SmolVLM-Instruct`)
* **Custom Captioning:** ViT Encoder (`WinKawaks/vit-small-patch16-224`) + GPT-2 Decoder (`gpt2`)
    * *Trained custom model for our dataset:* [Model Link](https://drive.google.com/file/d/1IS_jKcD0ginPrwo-OY14DutAsRF7cuMt/view?usp=drive_link)
* **Classifier:** BERT-base-uncased (`google-bert/bert-base-uncased`)

## How to Run

### 1.  **Setup:**
Install required Python libraries. You can use the following command:
   ```bash
   pip install torch transformers evaluate rouge_score nltk pandas Pillow gdown matplotlib scikit-learn tqdm
   ```
---
### 2. **Datasets and Model**

- Notebooks include `gdown` commands to download:
  - The dataset (zip format)
  - The custom model (`vit_gpt2_caption_model.pth`)
  - Evaluation results (`final_raw_results.csv`)

**Note:** If you have already downloaded these files and placed them correctly, you can skip or comment out the `gdown` cells.  
Ensure the dataset is extracted in `custom_captions_dataset/` directory.

---
### 3.  **Execution Order:**

Run the notebooks in the following order:
##### **Notebook:** `team_id_13_a.ipynb`

- Run all cells in sequence
- Restart the runtime between the following stages(if needed):
  1. SmolVLM evaluation  
  2. Custom model training  
- This step saves the model as: `vit_gpt2_caption_model.pth`

**Notebook:** `team_id_13_b.ipynb`

- Run after completing Part A
- Uses the model saved in Part A
- Saves intermediate results to `part_b_raw_results_csv/`
- Merges all results into `final_raw_results.csv`

**Notebook:** `team_id_13_c.ipynb`

- Run after Part B is complete
- Requires `final_raw_results.csv` from Part B
- Trains and evaluates a caption classifier
- Saves the final classifier as: `bert_caption_classifier_final.pth`
---
### 4.  **Hardware** 
A GPU environment (like Google Colab T4 or Kaggle GPU) is required for efficient model training and evaluation.