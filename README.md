# Projet Segmentation Sémantique - P8

Application de segmentation sémantique d'images urbaines avec plusieurs architectures de deep learning.

## 🚀 Déploiement

- **API Flask** : Heroku
- **Interface Streamlit** : Streamlit Cloud

📖 **Guide complet de déploiement** : Voir [DEPLOYMENT.md](DEPLOYMENT.md)

## 🏗️ Architecture

### Backend (API Flask)
- **Fichier** : `api.py`
- **Framework** : Flask + gunicorn
- **Fonctionnalités** :
  - Chargement dynamique des modèles
  - Endpoint `/predict` pour la segmentation
  - Endpoint `/models` pour lister les modèles disponibles
  - Cache des modèles en mémoire

### Frontend (Interface Streamlit)
- **Fichier** : `interface.py`
- **Framework** : Streamlit
- **Fonctionnalités** :
  - Upload multiple d'images
  - Navigation entre les images
  - Sélection du modèle de segmentation
  - Visualisation côte à côte (original / segmentation)

### Modèles disponibles

| Modèle | Taille | Paramètres | Déployé sur Heroku |
|--------|--------|------------|-------------------|
| UNet Mini | 7.4 MB | 0.9M | ✅ |
| HRNet-FPN | 43 MB | 11.2M | ✅ |
| DeepLabV3+ | 103 MB | 26.7M | ❌ (trop gros) |
| UNet-VGG16 | 306 MB | 26.7M | ❌ (trop gros) |

## 🛠️ Installation locale

### Prérequis
- Python 3.11.9
- pip

### Installation
```bash
# Cloner le repo
git clone https://github.com/kikduck/P8-OP-IA.git
cd P8-OP-IA

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

### Lancer l'API (terminal 1)
```bash
python api.py
```
L'API sera accessible sur `http://localhost:5000`

### Lancer l'interface (terminal 2)
```bash
streamlit run interface.py
```
L'interface sera accessible sur `http://localhost:8501`

## 📁 Structure du projet

```
P8-OP-IA/
├── api.py                    # API Flask
├── interface.py              # Interface Streamlit
├── model.py                  # Définitions des architectures
├── Procfile                  # Configuration Heroku
├── requirements.txt          # Dépendances Python
├── runtime.txt              # Version Python pour Heroku
├── .slugignore              # Fichiers ignorés par Heroku
├── DEPLOYMENT.md            # Guide de déploiement détaillé
├── train_models/
│   ├── unet_mini_best.pt    # Modèle UNet léger
│   ├── hrnet_fpn_best.pt    # Modèle HRNet
│   ├── deeplabv3plus_best.pt # (local uniquement)
│   └── unet_vgg16_best.pt    # (local uniquement)
└── DATA/                     # Données d'entraînement (gitignored)
```

## 🔧 Configuration

### Variables d'environnement (API)
- `PORT` : Port de l'API (auto sur Heroku)
- `DEFAULT_MODEL` : Modèle par défaut (`unet_mini` ou `hrnet`)
- `FLASK_DEBUG` : Mode debug (`True` ou `False`)

### Variables d'environnement (Interface)
- `API_URL` : URL de l'API Flask (ex: `https://mon-api.herokuapp.com`)

## 📊 Endpoints de l'API

### `GET /`
Informations sur l'API et le modèle chargé

### `GET /health`
Health check

### `GET /models`
Liste des modèles disponibles avec leurs métriques (mIoU)

### `POST /load_model`
Charger un modèle spécifique
```json
{
  "model_name": "hrnet"
}
```

### `POST /predict`
Prédire la segmentation d'une image
- **Input** : Image (multipart/form-data)
- **Output** : Image segmentée (PNG)

## 🎨 Classes de segmentation

| Classe | Couleur | Description |
|--------|---------|-------------|
| 0 | 🟪 Violet foncé | Route |
| 1 | 🟣 Rose | Trottoir |
| 2 | ⬛ Gris foncé | Bâtiment |
| 3 | 🟡 Jaune | Panneau de signalisation |
| 4 | 🟢 Vert olive | Végétation |
| 5 | 💚 Vert clair | Terrain |
| 6 | 🔴 Rouge | Personne |
| 7 | ⬛ Noir | Ignore |

## 📝 Licence

Projet académique - OpenClassrooms Formation IA Engineer

## 👤 Auteur

**kikduck**
- GitHub: [@kikduck](https://github.com/kikduck)
