import torch
import yaml
import pickle
import pandas as pd
import logging
import numpy as np
import argparse
from src.train_model import train_model
from datetime import datetime
from transformers import AutoModel, AutoTokenizer
from .data_loader import load_data_SCL, load_data_MCL, load_data_MCTL
from .eval import evaluate_on_folds
import os
os.environ["WANDB_MODE"] = "offline"
import wandb
import warnings
warnings.filterwarnings("ignore")


def load_config(file_path):
    """Load configuration from a YAML file."""
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def setup_logger(name, log_file_path, level=logging.INFO):
    """Set up a logger that writes to a file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # File handler for logging to a file
    file_handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    # Add the file handler to the logger
    logger.addHandler(file_handler)
    return logger

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Training configuration for PromptCASL.")
    # Dataset and model paths
    parser.add_argument("--dataset_path", type=str, default="../data/datasets/cell_line_biomedbert_kg50shared", help="Path to the dataset file.")
    parser.add_argument("--biobert_path", type=str, default="../data/biobert-base-cased-v1.2", help="Path to the pretrained BioBERT model.")
    parser.add_argument("--save_folder", type=str, default="../save", help="Path to the save train results.")
    parser.add_argument("--cell_line", type=str, choices=["mixed", "mctl","PK1", "IPC298", "MEWO", "A549", "22RV1", "GI1", "HS936T"], default="mixed", help="Specify the cell line to train on (or 'all' to use all datasets combined).")
    parser.add_argument("--scenario", type=str, choices=["SCL", "MCL", "MCTL"], default="SCL", help="Specify the scenario to train on. MCL: Multi Cell line Learning, SCL: Single Cell line Learning, MCTL: Multi Cell line Transfer Learning")
    parser.add_argument("--cell_line_set", nargs="+", default=["A549", "PK1", "IPC298", "MEWO"], help="Source cell lines for MCL/MCTL scenarios.")
    parser.add_argument("--target_cell_lines", nargs="+", default=["22RV1", "GI1", "HS936T"], help="Target (Test)) cell lines for MCTL scenarios.")
    parser.add_argument("--negpos", type=int, default=1, help="Negative to positive sample ratio (e.g., 1, 5, 10).")
    # Hyperparameters for model and optimizer
    parser.add_argument("--llm_lr", type=float, default=1e-5, help="Learning rate for LLM encoder.")
    parser.add_argument("--omics_lr", type=float, default=1e-4, help="Learning rate for Omics encoder.")
    parser.add_argument("--geneagg_lr", type=float, default=1e-4, help="Learning rate for gene attention aggregator.")
    parser.add_argument("--pred_lr", type=float, default=1e-4, help="Learning rate for predictor.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs for training.")
    parser.add_argument("--early_stop_patience", type=int, default=30, help="Early stop patience.")
    parser.add_argument("--lr_factor", type=float, default=0.5, help="Learning rate reducing factor.")
    parser.add_argument("--lr_patience", type=int, default=5, help="Patience for learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--device", type=int, default=0, help="Select the CUDA device index.")
    parser.add_argument("--omics_data", type=int, default=0, help="Whether to contain cell line omics embeddings.")
    parser.add_argument("--omics_encoder", type=int, default=0, help="Whether to use encoder to process cell line omics data.")
    # Model architecture
    parser.add_argument("--geneagg_in", type=int, default=3072, help="Input dim for gene attention aggregator.")
    parser.add_argument("--geneagg_hidden", type=int, default=768, help="Hidden dim for gene attention aggregator.")
    parser.add_argument("--geneagg_ff", type=int, default=1024, help="FF dim for gene attention aggregator.")
    parser.add_argument("--pred_in", type=int, default=4608, help="Input dim for predictor (concatenated features).")
    parser.add_argument("--pred_hidden", type=int, default=1024, help="Hidden dim for predictor.")
    parser.add_argument("--freeze_llm", action="store_true", help="If set, freeze the LLM encoder (default: False, i.e., trainable)")
    return parser.parse_args()



def create_save_folder(device, save_folder, scenario, cell_line):
    """Create a folder named save/{timestamp} in the parent directory of src/main.py."""
    current_time = datetime.now().strftime(f"%Y-%m-%d_%H-%M-%S_cuda{device}")
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_folder_path = os.path.join(save_folder, scenario, cell_line, current_time)
    os.makedirs(save_folder_path, exist_ok=True)
    return save_folder_path, current_time

def summarize_cv_results(final_results_path):
    df = pd.read_csv(final_results_path).dropna(how='all')
    if df.empty:
        print("No results found.")
        return
    # Compute mean and std
    mean_vals = df.mean()
    std_vals = df.std()

    # Append to file
    with open(final_results_path, "a") as f:
        f.write("\n")
        f.write(",".join([f"{v:.4f}" for v in mean_vals]) + "\n")
        f.write(",".join([f"{v:.4f}" for v in std_vals]) + "\n")

    # Prepare wandb log
    log_dict = {
        "mean_acc": mean_vals["Val Accuracy"],
        "mean_f1": mean_vals["Val F1"],
        "mean_auc": mean_vals["Val ROC-AUC"],
        "mean_aupr": mean_vals["Val AUPR"],
        "mean_bacc": mean_vals["Val BACC"],
        "mean_recall": mean_vals["Val Recall"],
        "std_acc": std_vals["Val Accuracy"],
        "std_f1": std_vals["Val F1"],
        "std_auc": std_vals["Val ROC-AUC"],
        "std_aupr": std_vals["Val AUPR"],
        "std_bacc": std_vals["Val BACC"],
        "std_recall": std_vals["Val Recall"],
        "folds": len(df)
    }

    wandb.log(log_dict)
    print("Logged summary metrics to wandb.")
    
def infer_scenario_from_cell_line(cell_line):
    if cell_line == "mixed":
        return "MCL"
    elif cell_line in {"mctl", "22RV1", "GI1", "HS936T"}:
    # elif cell_line == "mctl":
        return "MCTL"
    elif cell_line in {"A549", "PK1", "IPC298", "MEWO"}:
        return "SCL"
    else:
        raise ValueError(f"Unknown cell line: {cell_line}")


def main():
    args = parse_arguments()
    args.scenario = infer_scenario_from_cell_line(args.cell_line)
    
    # Create save folder
    save_folder, current_time = create_save_folder(args.device, args.save_folder, args.scenario, args.cell_line)
    print(f"Results will be saved in: {save_folder}")
    # Set up logging
    log_file_path = os.path.join(save_folder, "training.log")
    logger = setup_logger("training_logger", log_file_path)
    logger.info("Starting training process.")
    logger.info("Command-line arguments:")
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")
    # Set device
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    # Set up wandb
    wandb.init(
        project="PromptCASL",
        config=vars(args)
    )
    # Data loading and training
    if args.scenario in ["SCL", "MCL"]:
        for fold_idx in range(5):
            logger.info(f"Running fold {fold_idx + 1}")
            print(args.scenario, args.cell_line, f"Running fold {fold_idx + 1}")
            args.fold_idx = fold_idx
            # Load BioBERT
            biobert = AutoModel.from_pretrained(args.biobert_path).to(device)
            if args.scenario == "SCL":
                val_loader, train_loader = load_data_SCL(args.dataset_path, args.cell_line, args.batch_size, args.negpos, fold_idx)
            elif args.scenario == "MCL":
                val_loader, train_loader = load_data_MCL(args.dataset_path, args.cell_line_set, args.batch_size, args.negpos, fold_idx)
            train_model(
                train_loader=train_loader,
                val_loader=val_loader,
                bioLLM=biobert,
                args=args,
                save_path=save_folder,
                logger=logger,
                omics_data=args.omics_data,
                omics_encoder=args.omics_encoder
            )
            logger.info(f"Fold {fold_idx + 1} training completed successfully.")
        summarize_cv_results(os.path.join(save_folder, "final_results.csv"))
    elif args.scenario == "MCTL":
        args.fold_idx = None
        # Load BioBERT
        biobert = AutoModel.from_pretrained(args.biobert_path).to(device)
        val_loader, train_loader = load_data_MCTL(args.dataset_path, args.target_cell_lines, args.cell_line_set, args.negpos, args.batch_size)
        train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            bioLLM=biobert,
            args=args,
            save_path=save_folder,
            logger=logger,
            omics_data=args.omics_data,
            omics_encoder=args.omics_encoder
        )
    logger.info("Training completed successfully.")
    
    
    ### eval 
    if args.scenario == "MCL":
        eval_cell_lines = args.cell_line_set
        model_cell_line = "mixed"
    elif args.scenario == "MCTL":
        eval_cell_lines = args.target_cell_lines
        model_cell_line = "mctl"
    base_model_dir = f"../save/{args.scenario}"
    result_root = "../results"
    model_ids=[os.path.basename(save_folder)]
    evaluate_on_folds(eval_cell_lines, model_cell_line, args.scenario, model_ids, base_model_dir, result_root, device, biobert)





if __name__ == "__main__":
    main()




