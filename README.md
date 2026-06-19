# 🐝 Ruche Dashboard — Guide complet

Système de surveillance de ruche avec comptage d'abeilles par vision IA.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  GitHub Pages (frontend/index.html)                     │
│  Dashboard HTML/CSS/JS — interroge le backend toutes    │
│  les N secondes + interface upload vidéo                │
└────────────────────┬────────────────────────────────────┘
                     │ HTTPS GET /api/status
                     │ HTTPS POST /api/video
                     ▼
┌─────────────────────────────────────────────────────────┐
│  Backend Python (FastAPI) — Render / Fly.io / VPS       │
│  ┌──────────────┐  ┌─────────────────────────────────┐  │
│  │  /api/pico   │  │  /api/video  →  YOLOv8n + SORT  │  │
│  │  /api/status │  │  Counting line IN / OUT          │  │
│  └──────────────┘  └─────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────┘
                     │ HTTPS POST /api/pico
                     ▼
┌─────────────────────────────────────────────────────────┐
│  Raspberry Pi Pico WH (MicroPython)                     │
│  DHT22 · ADC batterie · compteur IR (optionnel)         │
└─────────────────────────────────────────────────────────┘
```

---

## 1. Backend (Render — recommandé)

### Déploiement

1. Créer un repo GitHub, y pousser le dossier `backend/`
2. Sur [render.com](https://render.com) → **New Web Service** → connecter le repo
3. Le fichier `render.yaml` configure tout automatiquement
4. Copier l'URL générée (ex: `https://ruche-api-xxxx.onrender.com`)
5. Copier le **RUCHE_TOKEN** dans les variables d'environnement Render

> **RAM** : YOLOv8n nécessite ~600 MB. Utiliser le plan **Starter** ($7/mois).
> Le plan Free (512 MB) est insuffisant pour l'inférence YOLO.

### Variables d'environnement

| Variable        | Valeur                                      |
|-----------------|---------------------------------------------|
| `RUCHE_TOKEN`   | token secret (généré automatiquement)       |
| `ALLOWED_ORIGIN`| `https://TONUSER.github.io`                 |
| `LINE_POS`      | `0.5` (50% = milieu de l'image)             |
| `CONF_THRESH`   | `0.30` (seuil YOLO, baisser si peu détecté) |
| `MAX_VIDEO_MB`  | `200`                                       |

### Test local

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
# → http://localhost:8000
# → http://localhost:8000/docs  (Swagger UI)
```

---

## 2. Frontend (GitHub Pages)

1. Créer un repo GitHub (ex: `ruche-dashboard`)
2. Y pousser le fichier `frontend/index.html`
3. Dans les **Settings** du repo → **Pages** → Branch: `main` / `/(root)`
4. L'URL sera `https://TONUSER.github.io/ruche-dashboard/`
5. Ouvrir la page → cliquer ⚙ → entrer l'URL du backend

---

## 3. Pico WH (MicroPython)

### Flasher MicroPython

1. Télécharger le firmware : https://micropython.org/download/RPI_PICO_W/
2. Maintenir **BOOTSEL** enfoncé → brancher USB → relâcher
3. Copier le `.uf2` sur le lecteur `RPI-RP2`

### Installer les fichiers

Avec **Thonny** (Menu View → Files) :
- Copier `pico/main.py` → `/main.py` sur la Pico
- Copier `pico/config.json` → `/config.json` sur la Pico (adapter les valeurs)

### Câblage DHT22

```
DHT22 Pin 1 (VCC)  → 3.3V (Pin 36)
DHT22 Pin 2 (DATA) → GP15 (Pin 20) + résistance 10kΩ vers 3.3V
DHT22 Pin 4 (GND)  → GND (Pin 38)
```

### Câblage batterie (pont diviseur)

```
V_BAT ──┤R1 100kΩ├──┬── GP28 (A2, Pin 34)
                     │
                   R2 100kΩ
                     │
                    GND
```
Tension mesurée = V_BAT / 2 → max 1.65 V sur l'ADC (Vref = 3.3V).

---

## 4. Pipeline de comptage vidéo

### Comment ça marche

```
Vidéo  →  [YOLOv8n]  →  bboxes par frame
                ↓
         [SORT Tracker]  →  ID stable par abeille
                ↓
      [Crossing Line]  →  détecte traversée
                ↓
           bee_in / bee_out
```

**Ligne de comptage** : horizontale, positionnée à 50% de la hauteur par défaut.
- Abeille qui monte (bottom→top) = **ENTRÉE** (IN)
- Abeille qui descend (top→bottom) = **SORTIE** (OUT)

Adapter la position selon l'orientation de ta caméra.

### Fine-tuner le modèle (optionnel mais recommandé)

Le modèle par défaut (YOLOv8n COCO) filtre les petits objets mais n'est pas
spécialisé "abeille". Pour de meilleures performances :

```bash
# 1. Télécharger un dataset abeilles (Roboflow)
#    https://universe.roboflow.com/mel-bees/bee-detection-ijmyp
#    Format : YOLOv8, dézipper dans bee_dataset/

# 2. Lancer l'entraînement
cd backend
python train_custom.py --data bee_dataset/data.yaml --epochs 100

# 3. Le modèle est automatiquement copié dans models/best.pt
#    et utilisé par le backend au prochain démarrage
```

---

## 5. Routes API

| Méthode | Route                           | Description                              |
|---------|---------------------------------|------------------------------------------|
| GET     | `/`                             | Health check                             |
| POST    | `/api/pico`                     | Données capteurs Pico WH                 |
| GET     | `/api/status`                   | État courant (frontend)                  |
| POST    | `/api/video`                    | Upload vidéo → analyse IA                |
| GET     | `/api/video/{job_id}`           | Statut/résultat d'un job                 |
| GET     | `/api/video/{job_id}/annotated` | Télécharger la vidéo annotée             |
| POST    | `/api/frame`                    | Image base64 → détection instantanée     |

Documentation interactive : `https://TON-BACKEND.example.com/docs`

---

## 6. Structure des fichiers

```
ruche/
├── frontend/
│   └── index.html          # GitHub Pages — dashboard + upload vidéo
├── backend/
│   ├── main.py             # FastAPI — toutes les routes
│   ├── bee_counter.py      # YOLOv8 + SORT tracker + comptage
│   ├── tracker.py          # Implémentation SORT (Kalman + IoU)
│   ├── train_custom.py     # Script fine-tuning modèle abeilles
│   ├── requirements.txt
│   ├── render.yaml
│   └── models/             # (créé au runtime)
│       └── best.pt         # modèle fine-tuné (optionnel)
└── pico/
    ├── main.py             # Firmware MicroPython
    └── config.json         # Configuration Wi-Fi / backend
```
