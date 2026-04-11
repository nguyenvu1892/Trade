import torch
from src.model.transformer import XAUTransformer
from src.training.dataset_loader import H5Dataset
from torch.utils.data import DataLoader

def run_eval():
    print("Evaluating checkpoints/best_model_cme_sniper_v2.pt ...")
    model = XAUTransformer(n_features=15, window_size=256, d_model=256, n_heads=8, n_layers=6)
    
    checkpoint = torch.load("checkpoints/best_model_cme_sniper_v2.pt", map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    
    dataset = H5Dataset("data/processed/XAUUSD_M5_w256.h5")
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    confidences = {"BUY": [], "SELL": []}
    
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            logits, _ = model(x)
            probs = torch.softmax(logits, dim=-1)
            
            # Action 1 = BUY, 2 = SELL
            for j in range(len(probs)):
                p = probs[j]
                buy_prob = p[1].item()
                sell_prob = p[2].item()
                
                confidences["BUY"].append(buy_prob)
                confidences["SELL"].append(sell_prob)
                    
            if i > 50: # Only scan a few batches for quick feedback
                break

    print("\n--- RESULTS ---")
    print(f"Total test samples scanned: {64 * 50}")
    
    for action in ["BUY", "SELL"]:
        c_list = confidences[action]
        print(f"{action} Signals Details:")
        if len(c_list) > 0:
            print(f"  Max Confidence: {max(c_list):.4f}")
            print(f"  Avg Confidence: {sum(c_list)/len(c_list):.4f}")
            print(f"  Min Confidence: {min(c_list):.4f}")

if __name__ == "__main__":
    run_eval()
