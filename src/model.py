import torch
import torch.nn as nn
import os

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, save_model=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.last_best = 0.0
        self.best_metrics = {
            'accuracy': 0,
            'f1': 0,
            'roc_auc': 0,
            'aupr': 0,
            'bacc': 0,
            'recall':0,
        }
        self.save_model = save_model

    def __call__(self, val_accuracy, val_f1, val_roc_auc, aupr, val_bacc, val_recall, model, model_path, args, logger=None):
        score = val_roc_auc

        if self.best_score is None:
            self.best_score = score
            self.best_metrics['accuracy'] = val_accuracy
            self.best_metrics['f1'] = val_f1
            self.best_metrics['roc_auc'] = val_roc_auc
            self.best_metrics['aupr'] = aupr
            self.best_metrics['bacc'] = val_bacc
            self.best_metrics['recall'] = val_recall
            self.save_checkpoint(score, val_accuracy, val_f1, val_roc_auc, aupr, val_bacc, val_recall, model, model_path, args, logger)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_metrics['accuracy'] = val_accuracy
            self.best_metrics['f1'] = val_f1
            self.best_metrics['roc_auc'] = val_roc_auc
            self.best_metrics['aupr'] = aupr
            self.best_metrics['bacc'] = val_bacc
            self.best_metrics['recall'] = val_recall
            self.save_checkpoint(score, val_accuracy, val_f1, val_roc_auc, aupr, val_bacc, val_recall, model, model_path, args, logger)
            self.counter = 0
        return self.best_metrics

    def save_checkpoint(self, score, val_accuracy, val_f1, val_roc_auc, aupr, val_bacc, val_recall, model, model_path, args, logger=None):
        if self.verbose:
            msg = (f'Validation AUC improved ({self.last_best:.6f} → {score:.6f}). '
                   f'Acc: {val_accuracy:.4f}, F1: {val_f1:.4f}, BACC: {val_bacc:.4f}, '
                   f'AUC: {val_roc_auc:.4f}, AUPR: {aupr:.4f}, Recall: {val_recall:.4f}')
            print(msg)
            if logger:
                logger.info(msg)
        # Save models
        os.makedirs(model_path, exist_ok=True)
        model_dict = {
            "llm_encoder": model[0].state_dict(),
            "gene_agg": model[1].state_dict(),
            "predictor": model[2].state_dict(),
        }
        if len(model) == 6: # [llm_encoder, gene_agg, predictor, ge_encoder, cnv_encoder, mu_encoder] 
            model_dict.update({
                "ge_encoder": model[3].state_dict(),
                "cnv_encoder": model[4].state_dict(),
                "mu_encoder": model[5].state_dict(),
            })

        # Compose filename
        if hasattr(args, "fold_idx") and args.fold_idx is not None and hasattr(args, "scenario"):
            filename = f"model_{args.scenario}_F{args.fold_idx + 1}.pt"
        elif hasattr(args, "scenario"):
            filename = f"model_{args.scenario}.pt"
        else:
            filename = "model.pt"
        torch.save(model_dict, os.path.join(model_path, filename))
        self.last_best = score


#########

class GeneAttentionAggregator(nn.Module):
    def __init__(self, input_dim=3072, hidden_dim=768, ff_dim=1024):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, geneA_emb, geneA_neighbors, geneB_neighbors, geneA_kg_emb, geneB_kg_emb):
        # geneA_emb: [B, hidden]
        # geneA_neighbors: [B, N, hidden]
        # geneB_neighbors: [B, N, hidden]
        # geneA_kg_emb: [B, hidden]
        # geneB_kg_emb: [B, hidden]

        compressed_geneA = self.proj(geneA_emb)  # [B, hidden]
        q = compressed_geneA.unsqueeze(1)  # [B, 1, hidden]
        geneA_kg_emb_expanded = geneA_kg_emb.unsqueeze(1)  # [B, 1, hidden]
        geneB_kg_emb_expanded = geneB_kg_emb.unsqueeze(1)  # [B, 1, hidden]
        geneA_neighbors = torch.cat([geneA_neighbors, geneA_kg_emb_expanded], dim=1)  # [B, N+1, hidden]
        geneB_neighbors = torch.cat([geneB_neighbors, geneB_kg_emb_expanded], dim=1)  # [B, N+1, hidden]
        k_v = torch.cat([geneA_neighbors, geneB_neighbors], dim=1)  # [B, 2*(N+1), hidden]

        attn_out, _ = self.attn(q, k_v, k_v)  # [B, 1, hidden]
        attn_out = attn_out.squeeze(1)  # [B, hidden]

        x = self.norm1(attn_out + compressed_geneA)
        ff_out = self.ff(x)
        out = self.norm2(x + ff_out)
        return compressed_geneA, out  # [B, hidden], [B, hidden]


class Predictor(nn.Module):
    def __init__(self, input_dim=768 * 6, hidden_dim=1024, output_dim=2, dropout_rate=0.1):
        super(Predictor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, features):
        return self.mlp(features)


class LLMEncoder(nn.Module):
    def __init__(self, model, model_type="auto"):
        super().__init__()
        self.model = model
        self.model_type = model_type.lower()
        # For auto-detection if not specified
        if model_type == "auto":
            if hasattr(model, "encoder"):
                self.model_type = "bert"
            elif hasattr(model, "layers"):
                self.model_type = "qwen"
            else:
                raise ValueError("Unknown model type. Please specify model_type explicitly.")

    def forward(self, inputs, attention_mask=None):
        """
        For BERT: 
          inputs = token embeddings ([batch, seq_len, hidden_dim])
        For Qwen3: 
          inputs = input_ids ([batch, seq_len])
        """
        if self.model_type == "bert":
            # print("token_emb shape:", token_emb.shape) # [batch_size, seq_len, hidden_size]
            # print("attention_mask shape:", attention_mask.shape) # [batch_size, seq_len]

            # attention_mask: [batch_size, seq_len]，0/1
            extended_attention_mask = attention_mask[:, None, None, :]  # [batch, 1, 1, seq_len]
            extended_attention_mask = extended_attention_mask.to(dtype=inputs.dtype)  # float
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0  # 1->0, 0->-10000
            
            # Get encoder outputs
            encoder_outputs = self.model.encoder(inputs, attention_mask=extended_attention_mask, output_attentions=True)  # Shape: [batch_size, seq_len, hidden_size]
            
            # print(type(encoder_outputs))
            # print(encoder_outputs.keys()) # ['last_hidden_state', 'attentions']
            # print(encoder_outputs.last_hidden_state.shape)
            # print(encoder_outputs.attentions[0].shape) # torch.Size([24, 12, 512, 512])

            encoder_output = encoder_outputs[0]
            attention_scores = encoder_outputs[-1]
            # Extract CLS token (first token) output
            cls_token_output = encoder_output[:, 0, :]  # Shape: [batch_size, hidden_size]
            # return cls_token_output, attention_scores
            return cls_token_output
        
        elif self.model_type == "qwen":
            # Ensure inputs (input_ids) are integers and squeeze middle dim if needed
            if not torch.is_floating_point(inputs):
                input_ids = inputs
            else:
                input_ids = inputs.long()
            # Squeeze the middle dimension if present (e.g., [B, 1, S] -> [B, S])
            if input_ids.dim() == 3 and input_ids.shape[1] == 1:
                input_ids = input_ids.squeeze(1)
            input_ids = input_ids.to(next(self.model.parameters()).device)
            outputs = self.model(input_ids)
            cls_token_output = outputs[0][:, 0, :]
            return cls_token_output
        
        else:
            raise ValueError("Unsupported model type: " + str(self.model_type))

class OmicsEncoder(nn.Module):
    def __init__(self, input_dim=942, output_dim=256, dropout_rate=0.1):
        super(OmicsEncoder, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.dropout(self.activation(self.linear(x)))

