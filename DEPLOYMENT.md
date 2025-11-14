# Guide de déploiement

## Vue d'ensemble
- **API Flask** → Déployée sur Heroku
- **Interface Streamlit** → Déployée sur Streamlit Cloud

---

## 1. Déploiement de l'API Flask sur Heroku (via UI)

### Étape 1 : Préparer votre compte Heroku
1. Créer un compte sur [https://heroku.com](https://heroku.com) (gratuit)
2. Se connecter au Dashboard Heroku

### Étape 2 : Créer une nouvelle application
1. Cliquer sur **"New"** → **"Create new app"**
2. Choisir un nom (ex: `mon-api-segmentation`)
3. Sélectionner la région (Europe ou US)
4. Cliquer sur **"Create app"**

### Étape 3 : Connecter votre repo GitHub
1. Dans l'onglet **"Deploy"** de votre app
2. Section **"Deployment method"** → Sélectionner **"GitHub"**
3. Cliquer sur **"Connect to GitHub"**
4. Autoriser Heroku à accéder à votre compte GitHub
5. Rechercher votre repo : `P8-OP-IA` (ou le nom de votre repo)
6. Cliquer sur **"Connect"**

### Étape 4 : Configurer le déploiement automatique (optionnel)
1. Section **"Automatic deploys"**
2. Sélectionner la branche `main`
3. Cliquer sur **"Enable Automatic Deploys"**
   - ✅ L'app se redéploiera automatiquement à chaque push sur `main`

### Étape 5 : Déployer manuellement
1. Section **"Manual deploy"**
2. Sélectionner la branche `main`
3. Cliquer sur **"Deploy Branch"**
4. Attendre la fin du build (2-5 minutes)

### Étape 6 : Configurer les variables d'environnement (optionnel)
1. Aller dans l'onglet **"Settings"**
2. Section **"Config Vars"** → Cliquer sur **"Reveal Config Vars"**
3. Ajouter (optionnel) :
   - `DEFAULT_MODEL` = `hrnet` (modèle par défaut)
   - `FLASK_DEBUG` = `False`

### Étape 7 : Vérifier que l'API fonctionne
1. Cliquer sur **"Open app"** (en haut à droite)
2. Vous devriez voir un JSON avec :
   ```json
   {
     "message": "API Flask - Segmentation d'images",
     "status": "running",
     ...
   }
   ```
3. Notez l'URL de votre API (ex: `https://mon-api-segmentation.herokuapp.com`)

### Étape 8 : Tester les endpoints
- **Health check** : `https://votre-app.herokuapp.com/health`
- **Liste des modèles** : `https://votre-app.herokuapp.com/models`

---

## 2. Déploiement de l'interface Streamlit sur Streamlit Cloud

### Étape 1 : Créer un compte Streamlit Cloud
1. Aller sur [https://streamlit.io/cloud](https://streamlit.io/cloud)
2. Se connecter avec votre compte GitHub

### Étape 2 : Déployer l'application
1. Cliquer sur **"New app"**
2. Sélectionner votre repo GitHub : `kikduck/P8-OP-IA`
3. **Branch** : `main`
4. **Main file path** : `interface.py`
5. Cliquer sur **"Advanced settings"**

### Étape 3 : Configurer les variables d'environnement
Dans **"Secrets"**, ajouter :
```toml
API_URL = "https://votre-app-heroku.herokuapp.com"
```
⚠️ **IMPORTANT** : Remplacez par l'URL réelle de votre API Heroku !

### Étape 4 : Déployer
1. Cliquer sur **"Deploy!"**
2. Attendre 2-3 minutes
3. L'interface sera accessible sur une URL comme : `https://votre-app.streamlit.app`

### Étape 5 : Vérifier
1. L'interface devrait se charger
2. La liste des modèles devrait apparaître dans la sidebar
3. Vous pouvez uploader des images et faire des prédictions

---

## 3. Vérifier que tout fonctionne ensemble

### Test complet :
1. ✅ API Heroku accessible : `https://votre-api.herokuapp.com/health`
2. ✅ Interface Streamlit chargée
3. ✅ Liste des modèles visible dans la sidebar
4. ✅ Upload d'image fonctionne
5. ✅ Prédiction fonctionne

---

## Structure des fichiers

```
.
├── api.py                  # API Flask
├── interface.py            # Interface Streamlit
├── model.py               # Définitions des modèles
├── Procfile               # Configuration Heroku (API)
├── requirements.txt       # Dépendances Python
├── runtime.txt            # Version Python
├── .slugignore           # Fichiers à ignorer sur Heroku
└── train_models/
    ├── unet_mini_best.pt     # Modèle léger (inclus)
    └── hrnet_fpn_best.pt     # Modèle HRNet (inclus)
```

## Notes importantes

1. **Taille des modèles**: Seuls `unet_mini_best.pt` et `hrnet_fpn_best.pt` sont déployés sur Heroku (les autres sont trop gros)

2. **Limite de slug Heroku**: Maximum 500MB. Si vous dépassez, utilisez Git LFS ou stockage externe (S3, Google Cloud Storage)

3. **Timeout**: Les requêtes Heroku ont un timeout de 30s. Pour les gros modèles, augmentez le timeout dans gunicorn

4. **Mémoire**: Heroku offre 512MB sur le plan gratuit. Surveillez l'utilisation mémoire avec:
   ```bash
   heroku ps:scale web=1:standard-1x  # Pour plus de mémoire
   ```

5. **Variables d'environnement**:
   - `PORT`: Automatiquement défini par Heroku
   - `DEFAULT_MODEL`: Modèle à charger au démarrage (défaut: hrnet)
   - `API_URL`: URL de l'API (pour l'interface Streamlit)

## Tester localement

### API Flask
```bash
python api.py
# Accessible sur http://localhost:5000
```

### Interface Streamlit
```bash
streamlit run interface.py
# Accessible sur http://localhost:8501
```

## Dépannage

### Erreur de mémoire
- Utiliser un dyno avec plus de RAM
- Charger les modèles à la demande au lieu du démarrage

### Timeout
- Augmenter le timeout dans le Procfile: `--timeout 300`
- Utiliser des workers asynchrones: `--worker-class gevent`

### Build échoue
```bash
# Voir les logs de build
heroku logs --tail

# Nettoyer le cache
heroku repo:purge_cache -a votre-app
```

## URL de l'application

- API: https://votre-app.herokuapp.com
- Health check: https://votre-app.herokuapp.com/health
- Liste des modèles: https://votre-app.herokuapp.com/models
