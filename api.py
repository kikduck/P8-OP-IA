from flask import Flask, request, jsonify, send_file
import torch
from pathlib import Path
import logging
import cv2
import numpy as np
from PIL import Image
import io
import os
from torchvision import transforms
from model import UNetMini, build_hrnet, build_deeplabv3plus, UNetVGG16

# Palette de couleurs pour les classes :
PALETTE_8 = [
    [128, 64, 128],   # 0: road
    [244, 35, 232],   # 1: sidewalk
    [70, 70, 70],     # 2: building
    [220, 220, 0],    # 3: traffic sign
    [107, 142, 35],   # 4: vegetation
    [152, 251, 152],  # 5: terrain
    [220, 20, 60],    # 6: person
    [0, 0, 0]         # 7: ignore
]

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialisation de Flask
app = Flask(__name__)

# Configuration des modèles disponibles
AVAILABLE_MODELS = {
    "unet_mini": {
        "name": "UNet Mini",
        "description": "UNet personnalisé léger (0.9M params)",
        "path": Path("train_models/unet_mini_best.pt"),
        "available": True
    },
    "hrnet": {
        "name": "HRNet-FPN",
        "description": "HRNet avec Feature Pyramid Network (11.2M params)",
        "path": Path("train_models/hrnet_fpn_best.pt"),
        "available": True
    },
    "deeplabv3plus": {
        "name": "DeepLabV3+",
        "description": "DeepLabV3+ avec encodeur ResNet50 (26.7M params)",
        "path": Path("train_models/deeplabv3plus_best.pt"),
        "available": False  # Not on Heroku (file too large)
    },
    "vgg16": {
        "name": "UNet-VGG16",
        "description": "UNet avec encodeur VGG16 (26.7M params)",
        "path": Path("train_models/unet_vgg16_best.pt"),
        "available": False  # Not on Heroku (file too large)
    }
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Variables globales pour les modèles
current_model_name = os.environ.get("DEFAULT_MODEL", "hrnet")  # Modèle par défaut
model = None
models_cache = {}  # Cache pour garder les modèles chargés en mémoire
models_info_cache = {}  # Cache pour les infos des modèles (mIoU, etc.)


def init_models_info_cache():
    """Initialise le cache des informations des modèles au démarrage"""
    global models_info_cache

    for key, info in AVAILABLE_MODELS.items():
        if info["available"] and info["path"].exists():
            try:
                checkpoint = torch.load(info["path"], map_location="cpu", weights_only=False)
                if isinstance(checkpoint, dict) and 'miou' in checkpoint:
                    models_info_cache[key] = {
                        "miou": float(checkpoint['miou'])
                    }
                    logger.info(f"Cache info pour {key}: mIoU={checkpoint['miou']:.4f}")
            except Exception as e:
                logger.warning(f"Impossible de charger les infos pour {key}: {e}")


def load_model(model_name):
    """Charge le modèle PyTorch (avec cache)"""
    global model, current_model_name, models_cache

    try:
        # Vérifier si le modèle est disponible
        if model_name not in AVAILABLE_MODELS:
            logger.error(f"Modèle inconnu: {model_name}")
            return False

        if not AVAILABLE_MODELS[model_name]["available"]:
            logger.error(f"Modèle {model_name} pas encore implémenté")
            return False

        # Si le modèle est déjà en cache, l'utiliser
        if model_name in models_cache:
            logger.info(f"Utilisation du modèle '{model_name}' depuis le cache")
            model = models_cache[model_name]
            current_model_name = model_name
            return True

        model_path = AVAILABLE_MODELS[model_name]["path"]
        logger.info(f"Chargement du modèle '{model_name}' depuis {model_path}...")

        # Créer une instance du modèle selon model_name
        if model_name == "unet_mini":
            new_model = UNetMini(num_classes=7, base_ch=32)
        elif model_name == "hrnet":
            new_model = build_hrnet(num_classes=7, weights=None)
        elif model_name == "deeplabv3plus":
            new_model = build_deeplabv3plus(num_classes=7, weights=None)
        elif model_name == "vgg16":
            new_model = UNetVGG16(num_classes=7, pretrained=False)  # Poids chargés depuis checkpoint
        else:
            logger.error(f"Modèle {model_name} pas implémenté dans model.py")
            return False

        # Charger le checkpoint
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)

        # Extraire le state_dict
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
                miou = checkpoint.get('miou', 'N/A')
                logger.info(f"mIoU du modèle sauvegardé: {miou}")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            logger.error("Format de checkpoint non reconnu")
            return False

        # Charger les poids dans le modèle
        new_model.load_state_dict(state_dict)
        new_model.eval()
        new_model.to(DEVICE)

        # Sauvegarder dans le cache
        models_cache[model_name] = new_model
        model = new_model
        current_model_name = model_name

        logger.info(f"Modèle '{model_name}' chargé avec succès sur {DEVICE}")
        logger.info(f"Type de modèle: {type(model).__name__}")
        return True
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

@app.route("/", methods=["GET"])
def home():
    """Endpoint de test"""
    return jsonify({
        "message": "API Flask - Segmentation d'images",
        "status": "running",
        "current_model": current_model_name,
        "model_loaded": model is not None,
        "device": str(DEVICE),
        "models_in_cache": list(models_cache.keys())
    })


@app.route("/health", methods=["GET"])
def health():
    """Endpoint health"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })


@app.route("/models", methods=["GET"])
def get_models():
    """Retourne la liste des modèles disponibles (avec cache)"""
    models_list = []
    for key, info in AVAILABLE_MODELS.items():
        model_info = {
            "id": key,
            "name": info["name"],
            "description": info["description"],
            "available": info["available"],
            "loaded": key in models_cache,
            "current": key == current_model_name
        }
        # Utiliser le cache pour le mIoU au lieu de charger le checkpoint
        if key in models_info_cache and "miou" in models_info_cache[key]:
            model_info["miou"] = models_info_cache[key]["miou"]

        models_list.append(model_info)

    return jsonify({
        "models": models_list,
        "current_model": current_model_name
    })


@app.route("/load_model", methods=["POST"])
def load_model_endpoint():
    """Charge un modèle spécifique"""
    data = request.get_json()
    if not data or 'model_name' not in data:
        return jsonify({"error": "Paramètre 'model_name' requis"}), 400

    model_name = data['model_name']

    if load_model(model_name):
        return jsonify({
            "success": True,
            "message": f"Modèle '{model_name}' chargé avec succès",
            "current_model": current_model_name
        })
    else:
        return jsonify({
            "success": False,
            "error": f"Erreur lors du chargement du modèle '{model_name}'"
        }), 500


@app.route("/predict", methods=["POST"])
def predict():
    """Endpoint de prédiction"""
    if model is None:
        return jsonify({"error": "Modèle non chargé"}), 500
    
    if 'image' not in request.files:
        return jsonify({"error": "Aucune image fournie"}), 400
    
    file = request.files['image']
    
    try:
        # Lire l'image
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Prétraitement (identique à l'entraînement)
        h, w = img_rgb.shape[:2]
        img_resized = cv2.resize(img_rgb, (512, 256), interpolation=cv2.INTER_LINEAR)

        # Conversion en float32 [0,1]
        img_float = img_resized.astype(np.float32) / 255.0

        # Normalisation ImageNet (CRUCIAL pour la performance du modèle)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_normalized = (img_float - mean) / std

        # Conversion en tenseur PyTorch (HWC -> CHW)
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).float()
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
        
        # Prédiction
        with torch.no_grad():
            logits = model(img_tensor)
            pred_mask = torch.argmax(logits, dim=1)[0].cpu().numpy()
        
        # Coloriser le mask (7 classes prédites: 0-6)
        mask_color = np.zeros((256, 512, 3), dtype=np.uint8)
        for k in range(7):
            mask_color[pred_mask == k] = PALETTE_8[k]
        
        # Redimensionner à la taille originale
        mask_color = cv2.resize(mask_color, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Superposer (50% image, 50% mask)
        result = cv2.addWeighted(img_rgb, 0.5, mask_color, 0.5, 0)
        
        # Convertir en bytes pour renvoyer
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.png', result_bgr)
        
        return send_file(
            io.BytesIO(buffer),
            mimetype='image/png',
            as_attachment=False,
            download_name='prediction.png'
        )
        
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        return jsonify({"error": str(e)}), 500

# Initialiser le cache des informations des modèles au démarrage
logger.info("Initialisation du cache des modèles...")
init_models_info_cache()

# Charger le modèle par défaut au démarrage
logger.info(f"Chargement du modèle par défaut: {current_model_name}")
if not load_model(current_model_name):
    logger.warning(f"Impossible de charger le modèle {current_model_name}, l'API démarrera sans modèle pré-chargé")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    logger.info(f"Démarrage de l'API Flask sur le port {port}...")
    app.run(host="0.0.0.0", port=port, debug=debug)
