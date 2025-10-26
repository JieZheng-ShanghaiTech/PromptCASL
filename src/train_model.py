import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.optim import Adam
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import os
import pandas as pd
import wandb
import pickle
import json
from .model import EarlyStopping, LLMEncoder, GeneAttentionAggregator, Predictor, OmicsEncoder
from sklearn.metrics import auc, roc_auc_score, f1_score, balanced_accuracy_score,accuracy_score, precision_recall_curve, recall_score
from torch.optim.lr_scheduler import ReduceLROnPlateau


def log_print(string, logger):
    print(string)
    logger.info(string)

def load_omics_emb_dict(pkl_path="../data/multi_omics/multi_omics_emb_3_256.pkl"):
    with open(pkl_path, "rb") as f:
        records_list = pickle.load(f)
    # Convert list to dict keyed by cell_line_name
    records = {rec["cell_line_name"]: rec for rec in records_list}
    print("Load cell line omics embeddings from AE.")
    return records

def get_omics_emb_for_cell_line(records, cell_line_name):
    rec = records.get(cell_line_name)
    if rec is None:
        raise ValueError(f"Cell line {cell_line_name} not found in omics embeddings.")
    omics_emb = np.concatenate([rec["ge_emb"], rec["cnv_emb"], rec["mu_emb"]], axis=0)  # shape (768,)
    return omics_emb

def load_omics_dict(path="../data/multi_omics/cell_line_omics_data.json"):
    with open(path, "r") as f:
        omics_emb_dict = json.load(f)
    return omics_emb_dict

def get_omics_for_cell_line(omics_dict, cell_line_name):
    """
    Returns a dictionary with GE, CNV, MU numpy vectors of shape (942,)
    """
    if cell_line_name not in omics_dict:
        raise ValueError(f"Cell line '{cell_line_name}' not found in omics_dict")
    data = omics_dict[cell_line_name]
    return {
        "GE": np.array(data["GE"], dtype=np.float32),
        "CNV": np.array(data["CNV"], dtype=np.float32),
        "MU": np.array(data["MU"], dtype=np.float32),
    }


def train_model(train_loader, val_loader, bioLLM, args, save_path, logger=None, omics_data=False, omics_encoder=False):
    device = args.device if hasattr(args, 'device') else 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if omics_data: # load cell line omics embs
        omics_emb_dict = load_omics_emb_dict()
    else:
        omics_emb_dict = None
        
    # 1. Instantiate models
    if omics_encoder:
        omics_dict = load_omics_dict()
        ge_encoder = OmicsEncoder(input_dim=942, output_dim=256).to(device)
        cnv_encoder = OmicsEncoder(input_dim=942, output_dim=256).to(device)
        mu_encoder = OmicsEncoder(input_dim=942, output_dim=256).to(device)
    else:
        omics_dict = None

    llm_encoder = LLMEncoder(bioLLM).to(device)
    
    if getattr(args, "freeze_llm", True):
        log_print("Instantiation: freeze_llm is set to be True", logger)
        for param in llm_encoder.parameters():
            param.requires_grad = False
          
    gene_agg = GeneAttentionAggregator(args.geneagg_in, args.geneagg_hidden, args.geneagg_ff).to(device)
    predictor = Predictor(args.pred_in, args.pred_hidden).to(device)

    # 2. Optimizer & Scheduler
    param_groups = []
    if not getattr(args, "freeze_llm", False):
        param_groups.append({'params': llm_encoder.parameters(), 'lr': args.llm_lr})
    if omics_encoder:
        param_groups.append({'params': ge_encoder.parameters(), 'lr': args.omics_lr})
        param_groups.append({'params': cnv_encoder.parameters(), 'lr': args.omics_lr})
        param_groups.append({'params': mu_encoder.parameters(), 'lr': args.omics_lr})
    param_groups.append({'params': gene_agg.parameters(), 'lr': args.geneagg_lr})
    param_groups.append({'params': predictor.parameters(), 'lr': args.pred_lr})

    optimizer = Adam(param_groups, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=args.lr_patience, factor=args.lr_factor)

    # 3. Early stopping
    early_stop = EarlyStopping(patience=args.early_stop_patience, verbose=True, save_model=True)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.num_epochs):
        # if epoch == 1: # for debug
        #     break
        if not getattr(args, "freeze_llm", False):
            llm_encoder.train()
        # else: do not set llm_encoder to train mode if frozen
        gene_agg.train()
        predictor.train()
        
        total_loss, total_correct, total_samples = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
        for batch in pbar:
            embeddings, labels, gene_names = batch
            
            if omics_emb_dict:
                omics_emb_list = []
                gene_pair_list, cell_line_list = gene_names
                for cell_line in cell_line_list:
                    omics_emb = get_omics_emb_for_cell_line(omics_emb_dict, cell_line)
                    omics_emb_list.append(omics_emb)
                omics_emb_batch = torch.tensor(np.stack(omics_emb_list), dtype=torch.float32).to(device)
            
            if omics_dict:
                ge_list, cnv_list, mu_list = [], [], []
                gene_pair_list, cell_line_list = gene_names
                for cell_line in cell_line_list:
                    omics_emb = get_omics_for_cell_line(omics_dict, cell_line)
                    ge_list.append(omics_emb["GE"])
                    cnv_list.append(omics_emb["CNV"])
                    mu_list.append(omics_emb["MU"])

                ge_tensor = torch.tensor(np.stack(ge_list), dtype=torch.float32).to(device)
                cnv_tensor = torch.tensor(np.stack(cnv_list), dtype=torch.float32).to(device)
                mu_tensor = torch.tensor(np.stack(mu_list), dtype=torch.float32).to(device)

                ge_encoded = ge_encoder(ge_tensor)
                cnv_encoded = cnv_encoder(cnv_tensor)
                mu_encoded = mu_encoder(mu_tensor)

                omics_encoded = torch.cat([ge_encoded, cnv_encoded, mu_encoded], dim=-1)  # shape [B, 768]

                 
            (token_emb, attention_mask, geneA_genept, geneA_kg_emb, geneB_genept, geneB_kg_emb, geneA_neighbors_emb, geneB_neighbors_emb) = [x.to(device) for x in embeddings]
            labels = labels.to(device)

            # print("geneA_genept shape:", geneA_genept.shape) # [20, 3072]
            ## Replace genePT embeddings with random tensors for ablation study
            # torch.manual_seed(42)
            # geneA_genept = torch.randn_like(geneA_genept) # normally distributed random vectors
            # geneB_genept = torch.randn_like(geneB_genept)

            # mask_indices = list(range(60, 160)) # mask gene omics description
            # # mask_indices = list(range(0, 60)) # mask cell line sentence 
            # pad_mask = (attention_mask == 0)  # [B, 512]
            # if pad_mask.any():
            #     # Use the first found padding token from any batch as replacement
            #     pad_emb = token_emb[pad_mask][0]  # [768]
            # else:
            #     raise ValueError("No padding token found in batch. Can't extract padding embedding.")
            # # Broadcast to [B, num_mask, 768]
            # for idx in mask_indices:
            #     token_emb[:, idx, :] = pad_emb 


            # Step 1: LLM Encoding
            sentence_emb = llm_encoder(token_emb, attention_mask)
            # sentence_emb = None
            
            # Step 2: Gene Attention Aggregation
            expressed_geneA, geneA_agg_emb = gene_agg(geneA_genept, geneA_neighbors_emb, geneB_neighbors_emb, geneA_kg_emb, geneB_kg_emb)
            expressed_geneB, geneB_agg_emb = gene_agg(geneB_genept, geneB_neighbors_emb, geneA_neighbors_emb, geneB_kg_emb, geneA_kg_emb)
            
            ## Step 3: Concatenate features     
            if omics_emb_dict: # omics_data (AE emb)
                features = torch.cat([ # 4608
                    omics_emb_batch, sentence_emb, expressed_geneA, geneA_agg_emb, expressed_geneB, geneB_agg_emb
                ], dim=-1)
                # ## without sentence_emb
                # features = torch.cat([ # 4608 - 768 = 3840
                #     omics_emb_batch, expressed_geneA, geneA_agg_emb, expressed_geneB, geneB_agg_emb
                # ], dim=-1)
                # ## without omics_emb_batch
                # features = torch.cat([ # 4608 - 768 = 3840
                #     omics_emb_batch, expressed_geneA, geneA_agg_emb, expressed_geneB, geneB_agg_emb
                # ], dim=-1)
            elif omics_dict: # omics_encoder
                features = torch.cat([
                    omics_encoded, sentence_emb, expressed_geneA, geneA_agg_emb, expressed_geneB, geneB_agg_emb
                ], dim=-1)
                # features = torch.cat([ # 3840
                #     omics_encoded, expressed_geneA, geneA_agg_emb, expressed_geneB, geneB_agg_emb
                # ], dim=-1)
            else:
                features = torch.cat([ # 3072
                    sentence_emb, expressed_geneA, geneA_agg_emb, expressed_geneB, geneB_agg_emb
                ], dim=-1)
            
            # Step 4: Prediction
            logits = predictor(features)
            # Step 5: Loss and Backprop
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Step 6: Metrics
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            pbar.set_postfix({'loss': loss.item()})
        train_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        
        # Validation
        # omics_emb_dict = omics_emb_dict if omics_emb_dict is not None else None
        if omics_dict: # omics_encoder
            val_metrics = validate_model(val_loader, llm_encoder, gene_agg, predictor, device, omics_emb_dict=omics_emb_dict, omics_dict=omics_dict, ge_encoder=ge_encoder,cnv_encoder=cnv_encoder,mu_encoder=mu_encoder)
        else:
            val_metrics = validate_model(val_loader, llm_encoder, gene_agg, predictor, device, omics_emb_dict=omics_emb_dict, omics_dict=omics_dict)
        
        log_print(f"Epoch {epoch + 1}/{args.num_epochs} - Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}, BACC: {val_metrics['bacc']:.4f}, AUC: {val_metrics['roc_auc']:.4f}, AUPR: {val_metrics['aupr']:.4f}, Recall: {val_metrics['recall']:.4f}", logger)
        
        # Early stopping and checkpoint
        if omics_encoder:
            early_stop(
                val_metrics['accuracy'], val_metrics['f1'], val_metrics['roc_auc'], val_metrics['aupr'], val_metrics['bacc'], val_metrics['recall'],
                [llm_encoder, gene_agg, predictor, ge_encoder, cnv_encoder, mu_encoder], save_path, args, logger
            )
        else: 
            early_stop(
                val_metrics['accuracy'], val_metrics['f1'], val_metrics['roc_auc'], val_metrics['aupr'], val_metrics['bacc'], val_metrics['recall'],
                [llm_encoder, gene_agg, predictor], save_path, args, logger
            )
        scheduler.step(val_metrics['loss'])
        if early_stop.early_stop:
            log_print(f"Early stopping at epoch {epoch+1}", logger)
            save_final_results(early_stop.best_metrics, save_path, fold_idx=args.fold_idx)
            break
    # If not early stopped, save final results after all epochs
    if not early_stop.early_stop and args.fold_idx is not None:
        save_final_results(early_stop.best_metrics, save_path, fold_idx=args.fold_idx)


def validate_model(data_loader, llm_encoder, gene_agg, predictor, device, omics_emb_dict=None, omics_dict=None, ge_encoder=None, cnv_encoder=None, mu_encoder=None):
    llm_encoder.eval()
    gene_agg.eval()
    predictor.eval()
    if ge_encoder:
        ge_encoder.eval()
        cnv_encoder.eval()
        mu_encoder.eval()
    
    with torch.no_grad():
        criterion = nn.CrossEntropyLoss()
        total_loss, total_correct, total_samples = 0, 0, 0
        all_outputs = []

        for batch in data_loader:
            embeddings, labels, gene_names = batch
            
            if omics_emb_dict:
                omics_emb_list = []
                gene_pair_list, cell_line_list = gene_names
                for cell_line in cell_line_list:
                    omics_emb = get_omics_emb_for_cell_line(omics_emb_dict, cell_line)
                    omics_emb_list.append(omics_emb)
                omics_emb_batch = torch.tensor(np.stack(omics_emb_list), dtype=torch.float32).to(device)
            
            if omics_dict:
                ge_list, cnv_list, mu_list = [], [], []
                gene_pair_list, cell_line_list = gene_names
                for cell_line in cell_line_list:
                    omics_emb = get_omics_for_cell_line(omics_dict, cell_line)
                    ge_list.append(omics_emb["GE"])
                    cnv_list.append(omics_emb["CNV"])
                    mu_list.append(omics_emb["MU"])
                ge_tensor = torch.tensor(np.stack(ge_list), dtype=torch.float32).to(device)
                cnv_tensor = torch.tensor(np.stack(cnv_list), dtype=torch.float32).to(device)
                mu_tensor = torch.tensor(np.stack(mu_list), dtype=torch.float32).to(device)
                ge_encoded = ge_encoder(ge_tensor)
                cnv_encoded = cnv_encoder(cnv_tensor)
                mu_encoded = mu_encoder(mu_tensor)

                omics_encoded = torch.cat([ge_encoded, cnv_encoded, mu_encoded], dim=-1)  # shape [B, 768]
                
            # Unpack embeddings
            (token_emb, attention_mask, geneA_genept, geneA_kg_emb, geneB_genept, geneB_kg_emb, geneA_neighbors_emb, geneB_neighbors_emb) = [x.to(device) for x in embeddings]
            labels = labels.to(device)
            
            ## Replace genePT embeddings with random tensors for ablation study
            # torch.manual_seed(42)
            # geneA_genept = torch.randn_like(geneA_genept) # normally distributed random vectors
            # geneB_genept = torch.randn_like(geneB_genept)
            
            
            # mask_indices = list(range(60, 160)) # mask gene omics description
            # # mask_indices = list(range(0, 60)) # mask cell line sentence
            # pad_mask = (attention_mask == 0)  # [B, 512]
            # if pad_mask.any():
            #     # Use the first found padding token from any batch as replacement
            #     pad_emb = token_emb[pad_mask][0]  # [768]
            # else:
            #     raise ValueError("No padding token found in batch. Can't extract padding embedding.")
            # # Broadcast to [B, num_mask, 768]
            # for idx in mask_indices:
            #     token_emb[:, idx, :] = pad_emb 
                
                
            # Step 1: LLM Encoding
            sentence_emb = llm_encoder(token_emb, attention_mask)
            # sentence_emb = None

            # Step 2: Gene Attention Aggregation
            expressed_geneA, geneA_agg_emb = gene_agg(geneA_genept, geneA_neighbors_emb, geneB_neighbors_emb, geneA_kg_emb, geneB_kg_emb)
            expressed_geneB, geneB_agg_emb = gene_agg(geneB_genept, geneB_neighbors_emb, geneA_neighbors_emb, geneB_kg_emb, geneA_kg_emb)
            
            ## Step 3: Concatenate features     
            if omics_emb_dict: # omics_data (AE emb)
                features = torch.cat([ # 4608
                    omics_emb_batch, sentence_emb, expressed_geneA, geneA_agg_emb, expressed_geneB, geneB_agg_emb
                ], dim=-1)
                # ## without sentence_emb
                # features = torch.cat([ # 4608 - 768 = 3840
                #     omics_emb_batch, expressed_geneA, geneA_agg_emb, expressed_geneB, geneB_agg_emb
                # ], dim=-1)
                # ## without omics_emb_batch
                # features = torch.cat([ # 4608 - 768 = 3840
                #     omics_emb_batch, expressed_geneA, geneA_agg_emb, expressed_geneB, geneB_agg_emb
                # ], dim=-1)
            elif omics_dict: # omics_encoder
                features = torch.cat([
                    omics_encoded, sentence_emb, expressed_geneA, geneA_agg_emb, expressed_geneB, geneB_agg_emb
                ], dim=-1)
                # features = torch.cat([ # 3840
                #     omics_encoded, expressed_geneA, geneA_agg_emb, expressed_geneB, geneB_agg_emb
                # ], dim=-1)
            else:
                features = torch.cat([ # 3072
                    sentence_emb, expressed_geneA, geneA_agg_emb, expressed_geneB, geneB_agg_emb
                ], dim=-1)
            
            # Step 4: Prediction
            logits = predictor(features)

            # Step 5: Loss and Backprop
            loss = criterion(logits, labels)

            # Step 6: Collect for metrics
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

            # For binary classification, get probabilities and predictions
            if logits.shape[1] == 1 or len(logits.shape) == 1:
                # Already sigmoid output or single logit
                probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
            else:
                # Multi-class, use softmax
                probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

            if 'all_labels' not in locals():
                all_labels = []
                all_probs = []
            all_labels.extend(labels.detach().cpu().numpy())
            all_probs.extend(probs)

            all_outputs.append(logits.cpu())

        avg_loss = total_loss / total_samples

        # Use macro F1 by finding best threshold
        all_labels_np = np.array(all_labels)
        all_probs_np = np.array(all_probs)
        best_f1 = 0
        best_threshold = 0.5
        for threshold in [x * 0.05 for x in range(20)]:
            threshold_preds = (all_probs_np > threshold).astype(int)
            f1 = f1_score(all_labels_np, threshold_preds, average='macro')
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        # Use best threshold for other metrics
        final_preds = (all_probs_np > best_threshold).astype(int)

        accuracy = accuracy_score(all_labels_np, final_preds)
        bacc = balanced_accuracy_score(all_labels_np, final_preds)
        roc_auc = roc_auc_score(all_labels_np, all_probs_np)
        precision, recall, _ = precision_recall_curve(all_labels_np, all_probs_np)
        aupr = auc(recall, precision)
        recall_scr = recall_score(all_labels_np, final_preds, average='macro')

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'bacc': bacc,
            'roc_auc': roc_auc,
            'aupr': aupr,
            'f1': best_f1,
            'recall': recall_scr
        }


def eval_model(data_loader, llm_encoder, gene_agg, predictor, device, omics_emb_dict=None, omics_dict=None, ge_encoder=None, cnv_encoder=None, mu_encoder=None, fold=None):
    llm_encoder.eval()
    gene_agg.eval()
    predictor.eval()
    if ge_encoder:
        ge_encoder.eval()
        cnv_encoder.eval()
        mu_encoder.eval()
    
    with torch.no_grad():
        criterion = nn.CrossEntropyLoss()
        total_loss, total_correct, total_samples = 0, 0, 0
        all_outputs = []

        ##########################################
        att_sum = None
        batch_count = 0
        embedding_records = []
        prediction_records = []
        ##########################################
        
        for batch in tqdm(data_loader, desc=f"Evaluating Fold {fold}"):
            embeddings, labels, gene_names = batch
            
            if omics_emb_dict:
                omics_emb_list = []
                gene_pair_list, cell_line_list = gene_names
                for cell_line in cell_line_list:
                    omics_emb = get_omics_emb_for_cell_line(omics_emb_dict, cell_line)
                    omics_emb_list.append(omics_emb)
                omics_emb_batch = torch.tensor(np.stack(omics_emb_list), dtype=torch.float32).to(device)
            
            if omics_dict:
                ge_list, cnv_list, mu_list = [], [], []
                gene_pair_list, cell_line_list = gene_names
                for cell_line in cell_line_list:
                    omics_emb = get_omics_for_cell_line(omics_dict, cell_line)
                    ge_list.append(omics_emb["GE"])
                    cnv_list.append(omics_emb["CNV"])
                    mu_list.append(omics_emb["MU"])
                ge_tensor = torch.tensor(np.stack(ge_list), dtype=torch.float32).to(device)
                cnv_tensor = torch.tensor(np.stack(cnv_list), dtype=torch.float32).to(device)
                mu_tensor = torch.tensor(np.stack(mu_list), dtype=torch.float32).to(device)
                ge_encoded = ge_encoder(ge_tensor)
                cnv_encoded = cnv_encoder(cnv_tensor)
                mu_encoded = mu_encoder(mu_tensor)

                omics_encoded = torch.cat([ge_encoded, cnv_encoded, mu_encoded], dim=-1)  # shape [B, 768]
                
            # Unpack embeddings
            (token_emb, attention_mask, geneA_genept, geneA_kg_emb, geneB_genept, geneB_kg_emb, geneA_neighbors_emb, geneB_neighbors_emb) = [x.to(device) for x in embeddings]
            labels = labels.to(device)
            
            ########### Replace genePT embeddings with random tensors for ablation study ###
            # torch.manual_seed(42)
            # geneA_genept = torch.randn_like(geneA_genept) # normally distributed random vectors
            # geneB_genept = torch.randn_like(geneB_genept)
            #################################################################################

            ########### mask part of the tokens ######################################
            # # mask_indices = list(range(60, 160)) # mask gene omics description
            # mask_indices = list(range(0, 60)) # mask cell line sentence
            # pad_mask = (attention_mask == 0)  # [B, 512]
            # if pad_mask.any():
            #     # Use the first found padding token from any batch as replacement
            #     pad_emb = token_emb[pad_mask][0]  # [768]
            # else:
            #     raise ValueError("No padding token found in batch. Can't extract padding embedding.")
            # # Broadcast to [B, num_mask, 768]
            # for idx in mask_indices:
            #     token_emb[:, idx, :] = pad_emb 
            ###########################################################################

            # Step 1: LLM Encoding
            sentence_emb, att_scores = llm_encoder(token_emb, attention_mask) ############ attention aggregation
            # sentence_emb = llm_encoder(token_emb, attention_mask)
            
            ########## attention aggregation ####################
            # print(len(att_scores)) # 12 layers # att_scores[0]: torch.Size([24, 12, 512, 512])
            # Average over heads: [batch, heads, 512, 512] → [batch, 512, 512]
            att_scores_avg_heads = [layer.mean(dim=1) for layer in att_scores]
            # Stack to [12, batch, 512, 512]
            att_stack = torch.stack(att_scores_avg_heads)  # [12, B, 512, 512]
            # Average over batch (dim=1): → [12, 512, 512]
            att_avg_batch = att_stack.mean(dim=1)
            # Accumulate
            if att_sum is None:
                att_sum = att_avg_batch
            else:
                att_sum += att_avg_batch
            batch_count += 1
            ######################################################
            
            # Step 2: Gene Attention Aggregation
            expressed_geneA, geneA_agg_emb = gene_agg(geneA_genept, geneA_neighbors_emb, geneB_neighbors_emb, geneA_kg_emb, geneB_kg_emb)
            expressed_geneB, geneB_agg_emb = gene_agg(geneB_genept, geneB_neighbors_emb, geneA_neighbors_emb, geneB_kg_emb, geneA_kg_emb)
            
            ## Step 3: Concatenate features     
            if omics_emb_dict:
                features = torch.cat([ # 4608
                    omics_emb_batch, sentence_emb, expressed_geneA, geneA_agg_emb, expressed_geneB, geneB_agg_emb
                ], dim=-1)
            elif omics_dict:
                features = torch.cat([
                    omics_encoded, sentence_emb, expressed_geneA, geneA_agg_emb, expressed_geneB, geneB_agg_emb
                ], dim=-1)
            else:
                features = torch.cat([ # 3072
                    expressed_geneA, geneA_agg_emb, expressed_geneB, geneB_agg_emb
                ], dim=-1)
            
            # Step 4: Prediction
            logits = predictor(features)

            # Step 5: Loss and Backprop
            loss = criterion(logits, labels)

            # Step 6: Collect for metrics
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

            # For binary classification, get probabilities and predictions
            if logits.shape[1] == 1 or len(logits.shape) == 1:
                # Already sigmoid output or single logit
                probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
            else:
                # Multi-class, use softmax
                probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

            if 'all_labels' not in locals():
                all_labels = []
                all_probs = []
            all_labels.extend(labels.detach().cpu().numpy())
            all_probs.extend(probs)

            all_outputs.append(logits.cpu())
            
            
            ####### get prediction and emb for pairs #####################
            for i in range(labels.size(0)):
                gp, cl = gene_names[0][i], gene_names[1][i]
                
                # Save prediction result
                label_i = labels[i].item()
                prob_i = float(probs[i])
                prediction_records.append({
                    "gene_pair": gp,
                    "cell_line": cl,
                    "label": label_i,
                    "pred_score": prob_i
                })

                # Save all embedding parts as arrays
                record = {
                    "gene_pair": gp,
                    "cell_line": cl
                }
                if 'omics_encoded' in locals():
                    record["omics_encoded"] = omics_encoded[i].cpu().numpy()
                record["sentence_emb"] = sentence_emb[i].cpu().numpy()
                record["expressed_geneA"] = expressed_geneA[i].cpu().numpy()
                record["geneA_agg_emb"] = geneA_agg_emb[i].cpu().numpy()
                record["expressed_geneB"] = expressed_geneB[i].cpu().numpy()
                record["geneB_agg_emb"] = geneB_agg_emb[i].cpu().numpy()
                embedding_records.append(record)
            #############################################################
        
        ####### save prediction and emb for pairs #####################
        # case_name = "mctl_complete"
        # output_dir = f"../results/case_study/materials/{case_name}"
        # os.makedirs(output_dir, exist_ok=True)
        # ## Save predictions (CSV)
        # pred_df = pd.DataFrame(prediction_records)
        # pred_df.to_csv(os.path.join(output_dir, f"predictions_F{fold}.csv"), index=False)
        # ## Save embeddings (Pickle)
        # with open(os.path.join(output_dir, f"gene_pair_embeddings_F{fold}.pkl"), "wb") as f:
        #     pickle.dump(embedding_records, f)
        # ########## attention save ##################################
        # att_final = att_sum / batch_count  # shape: [12, 512, 512]
        # torch.save(att_final.cpu(), f"{output_dir}/all_att_scores_F{fold}.pt")
        #############################################################
        
        avg_loss = total_loss / total_samples

        # Use macro F1 by finding best threshold
        all_labels_np = np.array(all_labels)
        all_probs_np = np.array(all_probs)
        best_f1 = 0
        best_threshold = 0.5
        for threshold in [x * 0.05 for x in range(20)]:
            threshold_preds = (all_probs_np > threshold).astype(int)
            f1 = f1_score(all_labels_np, threshold_preds, average='macro')
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        # Use best threshold for other metrics
        final_preds = (all_probs_np > best_threshold).astype(int)

        accuracy = accuracy_score(all_labels_np, final_preds)
        bacc = balanced_accuracy_score(all_labels_np, final_preds)
        roc_auc = roc_auc_score(all_labels_np, all_probs_np)
        precision, recall, _ = precision_recall_curve(all_labels_np, all_probs_np)
        aupr = auc(recall, precision)
        recall_scr = recall_score(all_labels_np, final_preds, average='macro')

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'bacc': bacc,
            'roc_auc': roc_auc,
            'aupr': aupr,
            'f1': best_f1,
            'recall': recall_scr
        }
        
        
def save_final_results(best_metrics, save_path, fold_idx=None):
    # Prepare the data for this run as a dictionary
    data = {
        'Fold': [fold_idx if fold_idx is not None else 'N/A'],
        'Val Accuracy': [best_metrics['accuracy']],
        'Val F1': [best_metrics['f1']],
        'Val ROC-AUC': [best_metrics['roc_auc']],
        'Val AUPR': [best_metrics['aupr']],
        'Val BACC': [best_metrics['bacc']],
        'Val Recall': [best_metrics['recall']],
    }

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(data)
    result_file = os.path.join(save_path, 'final_results.csv')
    # Write to CSV (append mode, with header only if the file doesn't exist)
    if os.path.exists(result_file):
        df.to_csv(result_file, mode='a', header=False, index=False)
    else:
        df.to_csv(result_file, mode='w', header=True, index=False)
        
        
        