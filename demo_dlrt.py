import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_friedman1
import pandas as pd

# This import comes from the RobustDLRT repo itself — adjust the path
# if you placed your script outside the repo root.
from src.low_rank_layers.layer_utils import transform_to_low_rank


# ---------------------------------------------------------------------
# 1. Dataset — replace this with your real data loading logic.
#    Right now it's synthetic: random X of shape [B, 384] and a target
#    Y of shape [B, 1] (e.g. some noisy linear/nonlinear function of X).
# ---------------------------------------------------------------------



class RegressionDatasetFriedman1(Dataset):
    def __init__(self, num_samples=10000, in_dim=384):
        
        X_output, Y_output = make_friedman1(n_samples=num_samples, n_features=in_dim, noise=0.0, random_state=42)
        
        self.X = torch.tensor(X_output, dtype=torch.float32).contiguous()
        # toy target function — replace with your actual labels
        # true_w = torch.randn(in_dim, 1)
        self.Y = torch.tensor(Y_output, dtype=torch.float32).unsqueeze(1).contiguous()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class RegressionDatasetCT(Dataset):
    def __init__(self, num_samples=10000, input_filename="~/PHANTOM/small-scale-study-main/slice_localization_data.csv"):
        df = pd.read_csv(input_filename)
        X_output = df.iloc[:, 1:-1].values
        Y_output = df.iloc[:, -1].values
        
        print(f"Loaded {X_output.shape[0]} samples with {X_output.shape[1]} features from {input_filename}")
        
        # Randomly sample num_samples from the data
        total_samples = X_output.shape[0]
        indices = torch.randperm(total_samples)[:num_samples].numpy()
        X_output = X_output[indices]
        Y_output = Y_output[indices]
        
        self.X = torch.tensor(X_output, dtype=torch.float32).contiguous()
        self.Y = torch.tensor(Y_output, dtype=torch.float32).unsqueeze(1).contiguous()


    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# train_dataset = RegressionDatasetFriedman1(num_samples=10000, in_dim=384)
# val_dataset = RegressionDatasetFriedman1(num_samples=2000, in_dim=384)

train_dataset = RegressionDatasetCT(num_samples=53500, input_filename="~/PHANTOM/small-scale-study-main/slice_localization_data.csv")
val_dataset = RegressionDatasetCT(num_samples=53500, input_filename="~/PHANTOM/small-scale-study-main/slice_localization_data.csv")


train_loader = DataLoader(
    train_dataset, batch_size=128, shuffle=True, num_workers=0
)
val_loader = DataLoader(
    val_dataset, batch_size=128, shuffle=False, num_workers=0
)

P = 8
nP = 16384//P

# ---------------------------------------------------------------------
# 2. Model — your FFN
# ---------------------------------------------------------------------
class FFNRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(384, nP)
        self.fc2 = nn.Linear(nP, nP)
        self.fc3 = nn.Linear(nP, nP)
        self.fc4 = nn.Linear(nP, nP)
        self.fc5 = nn.Linear(nP, 1)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.act(self.fc4(x))
        return self.fc5(x)


device = "cuda" if torch.cuda.is_available() else "cpu"
Model = FFNRegressor().to(device)


# ---------------------------------------------------------------------
# 3. Convert to low-rank (DLRT) layers
#    Skip this block entirely if you just want a standard dense FFN.
# ---------------------------------------------------------------------
USE_DLRT = True

print("First", Model)

if USE_DLRT:
    Model, lr_layers = transform_to_low_rank(
        Model, max_rank=200, init_rank=50, tol=0.1
    )
    print(f"Number of low-rank layers: {len(lr_layers)}")
    print("After DLRT conversion", Model)
    print("Current ranks:", [layer.r for layer in lr_layers])
else:
    lr_layers = []

print(Model)


optimizer = torch.optim.AdamW(Model.parameters(), lr=5e-4)
loss_fn = nn.MSELoss()

num_epochs = 20
num_local_iter = 10            # how often to augment/truncate (DLRT hyperparam)
robustness_beta = 0.0          # set > 0 (e.g. 0.075) to enable spectral regularization
print_per_batches = 20

for epoch in range(num_epochs):
    Model.train()
    running_loss = 0.0

    for batch_index, (X_train, Y_train) in enumerate(train_loader):
        X_train, Y_train = X_train.to(device), Y_train.to(device)

        optimizer.zero_grad()
        output = Model(X_train)                 # shape [B, 1]
        loss = loss_fn(output, Y_train)

        if USE_DLRT and robustness_beta > 0:
            for layer in lr_layers:
                loss = loss + layer.robustness_regularization(beta=robustness_beta)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(Model.parameters(), max_norm=1.0)

        if USE_DLRT:
            # ---- DLRT augment / train / truncate cycle ----
            if batch_index % num_local_iter == 0:
                for layer in lr_layers:
                    layer.augment(optimizer)
            else:
                for layer in lr_layers:
                    layer.set_basis_grad_zero()
                optimizer.step()

            if batch_index % num_local_iter == num_local_iter - 1:
                for layer in lr_layers:
                    layer.truncate(optimizer)
            # -------------------------------------------------
        else:
            optimizer.step()

        running_loss += loss.item()

        if (batch_index + 1) % print_per_batches == 0:
            avg_loss = running_loss / print_per_batches
            print(f"Epoch {epoch+1}/{num_epochs} "
                  f"Batch {batch_index+1}/{len(train_loader)} "
                  f"Loss: {avg_loss:.4f}")
            running_loss = 0.0

    # ---- validation ----
    Model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_val, Y_val in val_loader:
            X_val, Y_val = X_val.to(device), Y_val.to(device)
            val_loss += loss_fn(Model(X_val), Y_val).item()
    val_loss /= len(val_loader)
    print(f"[VAL] Epoch {epoch+1}: MSE = {val_loss:.4f}")

    if USE_DLRT:
        print("Current ranks:", [layer.r for layer in lr_layers])


os.makedirs("./checkpoints", exist_ok=True)
torch.save(Model.state_dict(), "./checkpoints/ffn_regressor.pth")