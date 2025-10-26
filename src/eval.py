import torch
import torch.nn as nn
import os
from transformers import AutoModel, AutoTokenizer
from .data_loader import load_data_SCL, load_data_MCL
from .train_model import eval_model, validate_model, load_omics_emb_dict, load_omics_dict
from .model import LLMEncoder, GeneAttentionAggregator, Predictor, OmicsEncoder
import logging
import re
from types import SimpleNamespace
import warnings
import pandas as pd
warnings.filterwarnings("ignore")

def parse_training_log(log_path):
    args_dict = {}
    with open(log_path, "r") as f:
        lines = f.readlines()
    start = None
    for i, line in enumerate(lines):
        if "Command-line arguments:" in line:
            start = i + 1
            break
    if start is None:
        raise ValueError("Could not find command-line arguments in log file.")
    for line in lines[start:]:
        match = re.match(r".*INFO -\s+(\w+):\s+(.+)", line)
        if match:
            key, value = match.groups()
            if value.isdigit():
                value = int(value)
            else:
                try:
                    value = float(value)
                except ValueError:
                    if value.lower() == "true":
                        value = True
                    elif value.lower() == "false":
                        value = False
            args_dict[key] = value
        else:
            break
    return args_dict

def load_models_from_log(model_dir, bioLLM, scenario, fold_idx, device, single_model=False):
    log_file = None
    for f in os.listdir(model_dir):
        if f.endswith("training.log"):
            log_file = os.path.join(model_dir, f)
            break
    if log_file is None:
        raise FileNotFoundError("No training.log found in model directory.")
    args_dict = parse_training_log(log_file)
    args = SimpleNamespace(**args_dict)
    if single_model:
        filename = f"model_{scenario}.pt"
    else:
        filename = f"model_{scenario}_F{fold_idx}.pt"
    model_path = os.path.join(model_dir, filename)
    checkpoint = torch.load(model_path, map_location="cpu")
    llm_encoder = LLMEncoder(bioLLM).to(device)
    gene_agg = GeneAttentionAggregator(args.geneagg_in, args.geneagg_hidden, args.geneagg_ff).to(device)
    predictor = Predictor(args.pred_in, args.pred_hidden).to(device)
    llm_encoder.load_state_dict(checkpoint["llm_encoder"])
    gene_agg.load_state_dict(checkpoint["gene_agg"])
    predictor.load_state_dict(checkpoint["predictor"])
    llm_encoder.eval()
    gene_agg.eval()
    predictor.eval()
    
    if "ge_encoder" in checkpoint and "cnv_encoder" in checkpoint and "mu_encoder" in checkpoint:
        ge_encoder = OmicsEncoder(input_dim=942, output_dim=256).to(device)
        cnv_encoder = OmicsEncoder(input_dim=942, output_dim=256).to(device)
        mu_encoder = OmicsEncoder(input_dim=942, output_dim=256).to(device)

        ge_encoder.load_state_dict(checkpoint["ge_encoder"])
        cnv_encoder.load_state_dict(checkpoint["cnv_encoder"])
        mu_encoder.load_state_dict(checkpoint["mu_encoder"])

        ge_encoder.eval()
        cnv_encoder.eval()
        mu_encoder.eval()
        
        return llm_encoder, gene_agg, predictor, ge_encoder, cnv_encoder, mu_encoder, args
    else:
         return llm_encoder, gene_agg, predictor, None, None, None, args


def evaluate_model(cell_lines: list, model_cell_line: str, scenario, model_ids: list, base_model_dir: str, result_root: str, device, bioLLM):
    for model_id in model_ids:
        print("eval for model: ", scenario, model_id)
        result_dir = os.path.join(result_root, scenario)
        os.makedirs(result_dir, exist_ok=True)
        result_path = os.path.join(result_dir, f"{model_id}.csv")
        # cell_lines = ["mixed"]
        for cell_line in cell_lines: # only need eval once for mixed
            print(f"\nEvaluating for {cell_line}\n{'='*50}")
            model_dir = os.path.join(base_model_dir, model_cell_line, model_id)
            model_files = [f for f in os.listdir(model_dir) if f.endswith(".pt") and "F" in f]
            single_model_filename = f"model_{scenario}.pt"
            single_model_path = os.path.join(model_dir, single_model_filename)
            results = []
            
            if not model_files and os.path.exists(single_model_path): ## MCTL
                print(scenario)
                print(f"Found single model file: {single_model_filename}")
                fold_idx = 1
                llm_encoder, gene_agg, predictor, ge_encoder, cnv_encoder, mu_encoder, args = load_models_from_log(model_dir, bioLLM, scenario, fold_idx, device, single_model=True)
                # cell_line_set = ["22RV1", "GI1", "HS936T"]
                # eval_loader, _ = load_data_MCL(args.dataset_path, cell_line_set, batch_size=args.batch_size, negpos=1, fold_idx=fold_idx-1)
                eval_loader, _ = load_data_SCL(args.dataset_path, cell_line, batch_size=args.batch_size, negpos=1, fold_idx=fold_idx-1)
                    
                if hasattr(args, "omics_data") and args.omics_data:
                    omics_emb_dict = load_omics_emb_dict()
                else:
                    omics_emb_dict = None
                if hasattr(args, "omics_encoder") and args.omics_encoder:
                    omics_dict = load_omics_dict()
                else:
                    omics_dict = None
                val_metrics = eval_model(eval_loader, llm_encoder, gene_agg, predictor, device, omics_emb_dict=omics_emb_dict, omics_dict=omics_dict, ge_encoder=ge_encoder,cnv_encoder=cnv_encoder,mu_encoder=mu_encoder, fold=fold_idx)
                
                results.append({
                    "Cell Line": cell_line,
                    "Fold": fold_idx,
                    "Val Loss": val_metrics['loss'],
                    "Acc": val_metrics['accuracy'],
                    "AUC": val_metrics['roc_auc'],
                    "AUPR": val_metrics['aupr'],
                    "F1": val_metrics['f1'],
                    "BACC": val_metrics['bacc'],
                    "Recall": val_metrics['recall']
                })
                df = pd.DataFrame(results)
                # Add an empty row for visual separation
                empty_row = {col: '' for col in df.columns}
                df = pd.concat([df, pd.DataFrame([empty_row])], ignore_index=True)
                
            else: ## MCL
                print(scenario)
                for model_file in model_files:
                    match = re.search(r'F(\d+)', model_file)
                    if not match:
                        continue
                    fold_idx = int(match.group(1))
                    llm_encoder, gene_agg, predictor, ge_encoder, cnv_encoder, mu_encoder, args = load_models_from_log(model_dir, bioLLM, scenario, fold_idx, device, single_model=False)
                    cell_line_set = ["A549", "PK1", "IPC298", "MEWO"]
                    eval_loader, _ = load_data_MCL(args.dataset_path, cell_line_set, batch_size=args.batch_size, negpos=1, fold_idx=fold_idx-1)
                    # specific_cell_line = "MEWO"
                    # eval_loader, _ = load_data_SCL(args.dataset_path, specific_cell_line, batch_size=args.batch_size, negpos=1, fold_idx=fold_idx-1)
                    
                    
                    if hasattr(args, "omics_data") and args.omics_data:
                        omics_emb_dict = load_omics_emb_dict()
                    else:
                        omics_emb_dict = None
                    if hasattr(args, "omics_encoder") and args.omics_encoder:
                        omics_dict = load_omics_dict()
                    else:
                        omics_dict = None
                    val_metrics = eval_model(eval_loader, llm_encoder, gene_agg, predictor, device, omics_emb_dict=omics_emb_dict, omics_dict=omics_dict, ge_encoder=ge_encoder,cnv_encoder=cnv_encoder,mu_encoder=mu_encoder, fold=fold_idx)
        
                    results.append({
                        "Cell Line": cell_line,
                        "Fold": fold_idx,
                        "Val Loss": val_metrics['loss'],
                        "Acc": val_metrics['accuracy'],
                        "AUC": val_metrics['roc_auc'],
                        "AUPR": val_metrics['aupr'],
                        "F1": val_metrics['f1'],
                        "BACC": val_metrics['bacc'],
                        "Recall": val_metrics['recall']
                    })
                df = pd.DataFrame(results)
                # Compute mean and std for each metric (excluding 'Cell Line' and 'Fold')
                if not df.empty and all(col in df.columns for col in ["Cell Line", "Fold"]):
                    mean_row = df.drop(columns=["Cell Line", "Fold"]).mean()
                    std_row = df.drop(columns=["Cell Line", "Fold"]).std()
                    mean_row["Cell Line"] = cell_line
                    mean_row["Fold"] = "mean"
                    std_row["Cell Line"] = cell_line
                    std_row["Fold"] = "std"
                    df = pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)
            print(df)
            if os.path.exists(result_path):
                print(f"Result file exists, appending: {result_path}")
                df.to_csv(result_path, mode='a', header=False, index=False)
            else:
                df.to_csv(result_path, index=False)
    
##########################


# device = torch.device("cuda:1")
# print(device)
# # print(torch.cuda.is_available()) # If False and AssertionError: Torch not compiled with CUDA enabled: conda activate only one env
# # # biobert_model_name = "../data/biobert-base-cased-v1.2"
# biobert_model_name = "../data/BiomedBERT"
# tokenizer = AutoTokenizer.from_pretrained(biobert_model_name)
# biobert = AutoModel.from_pretrained(biobert_model_name)
# biobert.to(device)
# ## eval 
# # eval_cell_lines = ["A549", "PK1", "IPC298", "MEWO"]
# eval_cell_lines = ["22RV1", "GI1", "HS936T"]
# model_cell_line = "mctl" # mixed_saved
# scenario = "MCTL"
# # eval_cell_lines = args.target_cell_lines
# # model_cell_line = "mctl"
# base_model_dir = f"../save/{scenario}"
# result_root = "../results/case_study"
# model_ids=[
#     # '2025-07-21_10-36-43_cuda1',
#     # '2025-07-21_13-43-34_cuda1' # freeze_llm
#     # "2025-07-22_10-47-30_cuda1" # mask gene omics
#     "2025-07-04_03-21-08_cuda1" 
# ]
# evaluate_model(eval_cell_lines, model_cell_line, scenario, model_ids, base_model_dir, result_root, device, biobert)


########  python -m src.eval 

