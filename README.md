# PromptCASL

**PromptCASL** is a prompt-driven, context-aware framework that leverages biomedical large language models (LLMs) for cell line-specific synthetic lethality (SL) prediction.

## Overview

![PromptCASL Overview](./PromptCASL_pipeline.png)

## File Structure

```
PromptCASL
├── data
│   ├── multi_omics/
│   │   ├── multi_omics_emb_3_256.pkl
│   │   └── cell_line_omics_data.json
│   └── text_template_genes/
|   └── datasets/
├── src
│   ├── data_loader.py              
│   ├── data_prepare.py             
│   ├── eval.py                     
│   ├── main.py                     
│   ├── model.py                    
│   ├── preprocess_datasets.py     
│   └── train_model.py              
├── env.yml                         
├── PromptCASL_pipeline.png         
└── README.md                       
```

## Requirements

To set up the environment for PromptCASL, use the provided `env.yml` file. A CUDA-enabled GPU is strongly recommended for training and evaluation.

```bash
conda env create -f env.yml
conda activate promptcasl
```

## Training and Evaluation


To train and evaluate the PromptCASL framework, run the following command with appropriate hyperparameters and dataset paths:

```bash
python -m src.main --llm_lr 1e-5 --geneagg_lr 1e-4 --pred_lr 1e-4 --weight_decay 1e-5 --batch_size 24 --num_epochs 50 --early_stop_patience 6 --lr_patience 3 --device 1 --cell_line mixed --pred_in 4608 --omics_data 0 --omics_encoder 1
```

- `--llm_lr`, `--geneagg_lr`, and `--pred_lr` control learning rates for different model components  
- `--omics_data` and `--omics_encoder` toggle omics usage and encoder mode
- `--cell_line` specifies the training scenario (e.g., `mixed`, `mctl`, or a specific cell line)  
- All paths and hyperparameters can be customized via command-line arguments or modified directly in `src/main.py`.

After training, evaluation metrics such as AUC and AUPR will be saved automatically in the `./save/` directory for each run. Note that the datasets used in this project are not included in this repository due to size limitations.


