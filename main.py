import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Graines aléatoires
torch.manual_seed(42)
np.random.seed(42)

# Chargement des données
print("Chargement des données...")
df = pd.read_csv('ai4i2020.csv')

# Calcul des poids
nb_pannes = df['Machine failure'].value_counts()
poids_classe_0 = 1.0
poids_classe_1 = nb_pannes[0] / nb_pannes[1]

print(f"\nDistribution des classes:")
print(f"Pas de panne: {nb_pannes[0]} échantillons")
print(f"Panne: {nb_pannes[1]} échantillons")
print(f"Ratio: {nb_pannes[0]/nb_pannes[1]:.2f}")

# Préparation des données
colonnes_inutiles = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
df = df.drop(colonnes_inutiles, axis=1)
df['Type'] = df['Type'].map({'L': 0, 'M': 1, 'H': 2})

X = df.drop('Machine failure', axis=1)
y = df['Machine failure']

# Split des données
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Conversion en tenseurs PyTorch
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train.values)
X_val_tensor = torch.FloatTensor(X_val_scaled)
y_val_tensor = torch.FloatTensor(y_val.values)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test.values)

# Modèle
class ReseauNeurones(nn.Module):
    def __init__(self, nb_features):
        super(ReseauNeurones, self).__init__()
        self.couche1 = nn.Linear(nb_features, 16)
        self.couche2 = nn.Linear(16, 8)
        self.couche3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.relu(self.couche1(x))
        x = self.dropout(x)
        x = self.relu(self.couche2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.couche3(x))
        return x

# Fonction de perte personnalisée
class FonctePertePonderee(nn.Module):
    def __init__(self, poids_classe_0, poids_classe_1):
        super(FonctePertePonderee, self).__init__()
        self.poids_classe_0 = poids_classe_0
        self.poids_classe_1 = poids_classe_1
        
    def forward(self, output, target):
        weights = torch.where(target == 1, 
                            torch.tensor(self.poids_classe_1),
                            torch.tensor(self.poids_classe_0))
        loss = -(weights * (target * torch.log(output + 1e-10) + 
                          (1 - target) * torch.log(1 - output + 1e-10)))
        return torch.mean(loss)

# Initialisation
print("\nInitialisation du modèle...")
model = ReseauNeurones(X_train_scaled.shape[1])
epochs = 300
batch_size = 32
learning_rate = 0.001

criterion = FonctePertePonderee(poids_classe_0, poids_classe_1)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)

def calculer_f1(y_pred, y_true):
    predictions = (y_pred > 0.5).float()
    predictions = predictions.cpu().numpy()
    y_true = y_true.cpu().numpy()
    return f1_score(y_true, predictions)

print("\nDébut de l'entraînement...")
historique_train_loss = []
historique_val_loss = []
historique_train_f1 = []
historique_val_f1 = []
meilleur_f1 = 0
seuil_optimal = 0.5

for epoch in range(epochs):
    model.train()
    train_losses = []
    y_pred_epoch = []
    y_true_epoch = []
    
    # Entraînement
    for i in range(0, len(X_train_tensor), batch_size):
        batch_X = X_train_tensor[i:i+batch_size]
        batch_y = y_train_tensor[i:i+batch_size]
        
        optimizer.zero_grad()
        outputs = model(batch_X).squeeze()
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        y_pred_epoch.extend(outputs.detach())
        y_true_epoch.extend(batch_y)
    
    # Calcul des métriques d'entraînement
    y_pred_epoch = torch.tensor(y_pred_epoch)
    y_true_epoch = torch.tensor(y_true_epoch)
    train_f1 = calculer_f1(y_pred_epoch, y_true_epoch)
    train_loss = np.mean(train_losses)
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor).squeeze()
        val_loss = criterion(val_outputs, y_val_tensor)
        val_f1 = calculer_f1(val_outputs, y_val_tensor)
        
        # Recherche du meilleur seuil
        seuils = np.arange(0.2, 0.8, 0.05)
        for seuil in seuils:
            f1_courant = f1_score(y_val_tensor.numpy(), 
                                (val_outputs.numpy() > seuil).astype(float))
            if f1_courant > meilleur_f1:
                meilleur_f1 = f1_courant
                seuil_optimal = seuil
    
    historique_train_loss.append(train_loss)
    historique_val_loss.append(val_loss.item())
    historique_train_f1.append(train_f1)
    historique_val_f1.append(val_f1)
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}]:')
        print(f'Perte train: {train_loss:.4f}, F1 train: {train_f1:.4f}')
        print(f'Perte validation: {val_loss:.4f}, F1 validation: {val_f1:.4f}')
        print(f'Meilleur seuil: {seuil_optimal:.3f}, Meilleur F1: {meilleur_f1:.4f}\n')

# Évaluation finale
print("\nÉvaluation sur le jeu de test...")
model.eval()
with torch.no_grad():
    outputs_test = model(X_test_tensor).squeeze()
    predictions = (outputs_test > seuil_optimal).float().numpy()
    y_test_np = y_test.values
    
    rapport = classification_report(y_test_np, predictions)
    matrice_confusion = confusion_matrix(y_test_np, predictions)

print(f"\nSeuil optimal final: {seuil_optimal:.3f}")
print("\nRapport de classification:")
print(rapport)

# Visualisation
plt.figure(figsize=(15, 10))

# Courbes de perte
plt.subplot(2, 2, 1)
plt.plot(historique_train_loss, label='Train')
plt.plot(historique_val_loss, label='Validation')
plt.title('Courbes de perte')
plt.xlabel('Epoch')
plt.ylabel('Perte')
plt.legend()
plt.grid(True)

# Courbes F1-score
plt.subplot(2, 2, 2)
plt.plot(historique_train_f1, label='Train')
plt.plot(historique_val_f1, label='Validation')
plt.title('Courbes F1-score')
plt.xlabel('Epoch')
plt.ylabel('F1-score')
plt.legend()
plt.grid(True)

# Matrice de confusion
plt.subplot(2, 2, 3)
sns.heatmap(matrice_confusion, annot=True, fmt='d', cmap='Blues')
plt.title(f'Matrice de confusion (seuil: {seuil_optimal:.3f})')
plt.xlabel('Prédiction')
plt.ylabel('Réalité')

# Distribution des classes
plt.subplot(2, 2, 4)
labels = ['Pas de panne', 'Panne']
class_dist = [matrice_confusion[0].sum(), matrice_confusion[1].sum()]
plt.pie(class_dist, labels=labels, autopct='%1.1f%%')
plt.title('Distribution des classes')

plt.tight_layout()
plt.show()
