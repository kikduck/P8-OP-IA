# Guide de déploiement sur Heroku

## Prérequis
- Compte Heroku
- Heroku CLI installé
- Git configuré

## Déploiement de l'API Flask

### 1. Installation de Heroku CLI
```bash
# Windows
https://devcenter.heroku.com/articles/heroku-cli

# Vérifier l'installation
heroku --version
```

### 2. Connexion à Heroku
```bash
heroku login
```

### 3. Créer une application Heroku
```bash
# Créer l'application
heroku create votre-nom-app-segmentation

# Ou si vous avez déjà une app
heroku git:remote -a votre-nom-app-segmentation
```

### 4. Configurer les variables d'environnement (optionnel)
```bash
heroku config:set DEFAULT_MODEL=hrnet
heroku config:set FLASK_DEBUG=False
```

### 5. Déployer sur Heroku
```bash
git push heroku main
```

### 6. Vérifier les logs
```bash
heroku logs --tail
```

### 7. Ouvrir l'application
```bash
heroku open
```

## Déploiement de l'interface Streamlit

L'interface Streamlit doit être déployée séparément car Heroku ne supporte qu'un seul processus web.

### Option 1: Streamlit Cloud (recommandé)
1. Aller sur https://streamlit.io/cloud
2. Connecter votre repo GitHub
3. Sélectionner `interface.py` comme fichier principal
4. Ajouter les variables d'environnement:
   - `API_URL`: URL de votre API Heroku (ex: `https://votre-app.herokuapp.com`)
5. Déployer

### Option 2: Heroku séparé pour Streamlit
Créer un nouveau `Procfile.streamlit`:
```
web: streamlit run interface.py --server.port=$PORT --server.address=0.0.0.0
```

Puis créer une nouvelle app Heroku pour l'interface.

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
