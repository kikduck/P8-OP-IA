# Optimisation mémoire pour Heroku

## Problème initial
Heroku Free tier = 512MB RAM maximum
Notre app utilisait **720MB** → Erreur R14 (Memory quota exceeded)

## Solutions implémentées

### 1. Lazy Loading des modèles ✅
**Avant** : Le modèle est chargé au démarrage de l'application
**Après** : Le modèle est chargé uniquement lors de la première requête `/predict`

**Gain estimé** : ~150MB au démarrage

```python
# Détection automatique de Heroku via HEROKU_SLUG_COMMIT
if os.environ.get("HEROKU_SLUG_COMMIT"):
    logger.info("Running on Heroku - lazy loading enabled")
else:
    load_model(current_model_name)  # En local seulement
```

### 2. Cache désactivé sur Heroku ✅
**Avant** : Tous les modèles chargés restent en mémoire (cache)
**Après** : Sur Heroku, le cache est vidé avant de charger un nouveau modèle

**Gain** : Permet de ne garder qu'un seul modèle en RAM à la fois

```python
if os.environ.get("HEROKU_SLUG_COMMIT") and models_cache:
    models_cache.clear()
    gc.collect()
```

### 3. Gunicorn optimisé ✅
**Configuration** :
- `--workers 1` : Un seul worker (économie RAM)
- `--threads 2` : 2 threads par worker (pour gérer plusieurs requêtes)
- `--max-requests 100` : Redémarre le worker après 100 requêtes (nettoie la mémoire)
- `--timeout 120` : Timeout de 2 minutes pour les requêtes longues

### 4. PyTorch CPU-only ✅
**Avant** : PyTorch avec CUDA (~2GB)
**Après** : PyTorch CPU-only (~500MB)

**Gain** : ~1.5GB

### 5. Dépendances minimales ✅
Suppression de :
- `matplotlib` (non utilisé en production)
- `albumentations` (non utilisé en production)
- Jupyter notebooks

## Résultat attendu

| Composant | Mémoire |
|-----------|---------|
| Python + Flask + gunicorn | ~100MB |
| PyTorch CPU | ~150MB |
| UNet Mini (modèle) | ~30MB |
| Dépendances | ~100MB |
| **Total estimé** | **~380MB** ✅ |

**Marge de sécurité** : ~130MB (25% de la limite)

## Monitoring sur Heroku

### Vérifier l'utilisation mémoire
```bash
heroku logs --tail | grep "mem="
```

### Logs typiques après optimisation
```
Process running mem=380M(74.2%)  ✅ OK
```

### Si toujours R14
```
Process running mem=550M(107.4%) ❌ Dépassement
```

## Solutions supplémentaires si nécessaire

### Option 1 : Utiliser uniquement UNet Mini
Dans Heroku Config Vars :
```
DEFAULT_MODEL=unet_mini
```
UNet Mini est le plus léger (7.4MB vs 43MB pour HRNet)

### Option 2 : Passer au plan Hobby ($7/mois)
- RAM : 512MB → 1GB
- Pas de sleep automatique
- Meilleur pour production

### Option 3 : Hébergement alternatif
- **Render.com** : 512MB gratuit (meilleure gestion mémoire)
- **Railway.app** : 512MB gratuit + $5 de crédit
- **Fly.io** : 256MB gratuit mais optimisé

## Variables d'environnement Heroku

| Variable | Valeur | Description |
|----------|--------|-------------|
| `DEFAULT_MODEL` | `unet_mini` | Modèle par défaut (le plus léger) |
| `FLASK_DEBUG` | `False` | Désactiver le mode debug |
| `PYTHONUNBUFFERED` | `1` | Logs en temps réel |

## Notes importantes

1. **Premier chargement lent** : La première requête `/predict` sera plus lente (2-3s) car le modèle doit être chargé en mémoire

2. **Changement de modèle** : Sur Heroku, changer de modèle via `/load_model` videra le cache pour économiser la RAM

3. **Performance** : Les prédictions restent rapides une fois le modèle chargé (100-200ms par image)

4. **Sleep Heroku** : L'app s'endort après 30min d'inactivité (free tier). Au réveil, le modèle devra être rechargé.
