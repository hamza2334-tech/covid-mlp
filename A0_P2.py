import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

df = pd.read_csv(
    "/data/Provisional_COVID-19_death_counts__rates__and_percent_of_total_deaths__by_jurisdiction_of_residence.csv"
)

df_clean = df.drop(
    columns=["data_as_of", "data_period_start", "data_period_end", "footnote"]
)

df_clean = df_clean.dropna(subset=["COVID_deaths"])

cols_to_impute = [
    "COVID_pct_of_total",
    "pct_change_wk",
    "pct_diff_wk",
    "crude_COVID_rate",
    "aa_COVID_rate",
    "crude_COVID_rate_ann",
    "aa_COVID_rate_ann",
]

for col in cols_to_impute:
    df_clean[col] = df_clean.groupby("Jurisdiction_Residence")[col].transform(
        lambda x: x.fillna(x.median())
    )

df_encoded = pd.get_dummies(
    df_clean, columns=["Jurisdiction_Residence", "Group"], drop_first=True
)

scaler = MinMaxScaler()
num_cols = df_encoded.select_dtypes(include=["float64", "int64"]).columns
df_encoded[num_cols] = scaler.fit_transform(df_encoded[num_cols])


def classify_deaths(value):
    if value <= 100:
        return 0
    elif value <= 500:
        return 1
    elif value <= 1000:
        return 2
    else:
        return 3


df_encoded["COVID_class"] = df_clean["COVID_deaths"].apply(classify_deaths)

X = df_encoded.drop(columns=["COVID_deaths", "COVID_class"])
y = df_encoded["COVID_class"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

X_train = X_train.astype("float32")
X_val = X_val.astype("float32")
X_test = X_test.astype("float32")

X_train_tensor = torch.tensor(X_train.values)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_val_tensor = torch.tensor(X_val.values)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test.values)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

train_loader = DataLoader(
    TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True
)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=64)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=64)


class CovidNN(nn.Module):
    def __init__(self, input_size):
        super(CovidNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.out = nn.Linear(16, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.out(x)
        return x


model = CovidNN(X_train.shape[1])

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.015)

for epoch in range(40):
    model.train()
    correct, total = 0, 0
    for inputs, targets in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Train Accuracy = {100 * correct / total:.2f}%")

model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predictions = torch.max(outputs, 1)
    test_accuracy = (
        (predictions == y_test_tensor).sum().item() / y_test_tensor.size(0) * 100
    )

print(f"\n Final Test Accuracy: {test_accuracy:.2f}%")

# --- 7. Plot Confusion Matrix ---
cm = confusion_matrix(y_test_tensor.numpy(), predictions.numpy())
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2, 3])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Test Set")
plt.show()
