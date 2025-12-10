# Ministral-3-3B Sudoku Fine-tuning avec GRPO

Ce projet implémente le fine-tuning d'un modèle Ministral-3-3B sur une tâche de résolution de Sudoku en utilisant le **Group Relative Policy Optimization (GRPO)** avec Unsloth, puis l'export au format GGUF pour une inférence optimisée.

## 📋 Table des matières

- [Vue d'ensemble](#vue-densemble)
- [Prérequis](#prérequis)
- [Architecture du projet](#architecture-du-projet)
- [Workflow complet](#workflow-complet)
- [Scripts principaux](#scripts-principaux)
- [Dépendances et workarounds](#dépendances-et-workarounds)
- [Déploiement](#déploiement)
- [Structure des dossiers](#structure-des-dossiers)

---

## 🎯 Vue d'ensemble

Ce projet démontre:
- **Fine-tuning GRPO** d'un modèle de langage sur une tâche de raisonnement (Sudoku)
- **Conversion au format GGUF** pour llama.cpp
- **Déploiement** avec vLLM ou llama.cpp
- **Workaround** pour les bugs d'export GGUF d'Unsloth

### Modèle de base
- **Modèle**: `unsloth/Ministral-3-3B-Instruct-2512`
- **Méthode**: GRPO (Group Relative Policy Optimization)
- **Tâche**: Génération de code Python pour résoudre des puzzles Sudoku

### Résultats
- Modèle fine-tuné capable de générer des stratégies Sudoku valides
- Export GGUF réussi (F16, Q8_0)
- Déploiement réussi sur vLLM et llama.cpp

---

## 🔧 Prérequis

### Environnement requis
- **GPU**: NVIDIA avec CUDA (testé sur DGX Spark GB10)
- **Python**: 3.10+
- **VRAM**: ~12GB minimum pour le fine-tuning

### Packages Python principaux
```bash
pip install unsloth torch transformers trl datasets
pip install python-dotenv huggingface-hub

# Pour l'export GGUF (installation manuelle requise)
pip install gguf>=0.17.0
pip install sentencepiece>=0.2.0
pip install protobuf>=6.0.0
```

### Configuration
Créer un fichier `.env` à la racine:
```bash
HF_TOKEN=your_huggingface_token_here
```

---

## 🏗️ Architecture du projet

```
┌─────────────────────────────────────────────────────────────┐
│  1. Fine-tuning GRPO (Unsloth)                              │
│     → Génère des adaptateurs LoRA                           │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  2. Vérification LoRA (optionnel)                           │
│     → Vérifie que les tensors ne sont pas vides            │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  3. Merge LoRA + Base Model (Unsloth)                       │
│     → Modèle 16bit complet                                  │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  4. Push vers HuggingFace Hub                               │
│     → Modèle partageable et téléchargeable                 │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  5. Export GGUF (llama.cpp) ⭐ WORKAROUND                  │
│     → Bypass Unsloth, utilise llama.cpp directement        │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  6. Push GGUF vers HuggingFace Hub                          │
│     → Fichiers GGUF pour llama.cpp                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Workflow complet

### Étape 1: Fine-tuning GRPO

```bash
python 1_ministral_3_rl_sudoku.py
```

**Ce script fait**:
- Charge le modèle Ministral-3-3B-Instruct
- Configure les adaptateurs LoRA (rank 32)
- Entraîne avec GRPO sur 1000 exemples de Sudoku
- Sauvegarde les adaptateurs dans `grpo_saved_lora/`

**Outputs**:
- `grpo_saved_lora/` - Adaptateurs LoRA
- `outputs/` - Checkpoints d'entraînement

**Durée estimée**: 1-3 heures selon le GPU

---

### Étape 2: Vérification LoRA (optionnel)

```bash
python 2_check_lora.py
```

**Ce script fait**:
- Vérifie que les tensors LoRA ne sont pas tous à zéro
- Affiche le pourcentage de zéros par layer

**Output**: Validation console uniquement

---

### Étape 3: Merge des adaptateurs

```bash
python 3_merge_for_vllm_v2.py
```

**Ce script fait**:
- Charge le modèle de base
- Applique la structure PEFT (même config que le training)
- Charge les poids des adaptateurs
- Merge avec `save_pretrained_merged("merged_16bit")`
- Sauvegarde le modèle complet

**Configuration**:
```python
BASE_MODEL = "unsloth/Ministral-3-3B-Instruct-2512"
LORA_ADAPTERS_PATH = "grpo_saved_lora"
OUTPUT_DIR = "ministral_3_sudoku_vllm"
```

**Output**: `ministral_3_sudoku_vllm/` (~6GB)

**Durée estimée**: 5-10 minutes

---

### Étape 4: Push vers HuggingFace

#### Option A: Script complet (recommandé)

```bash
python 4_save_to_hf_v2.py
```

**Ce script fait**:
- Charge le modèle mergé depuis le disque
- Push vers HuggingFace Hub
- Télécharge dans le cache local pour vLLM

**Configuration**:
```python
MERGED_MODEL_DIR = "/models/fine-tuned/ministral_3_sudoku_vllm"
HF_REPO_NAME = "applied-ai-subscr/ministral_3_sudoku_vllm"
```

#### Option B: Script simple

```bash
python 4-alt_push_manual.py
```

Version simplifiée qui upload directement le dossier sans téléchargement.

**Durée estimée**: 10-30 minutes selon la connexion

---

### Étape 5: Export GGUF ⭐

```bash
python 5_export_gguf_v2.py
```

**Ce script fait**:
- **Utilise directement** `llama.cpp/convert_hf_to_gguf.py`
- **Bypass Unsloth** (qui plante sur l'export GGUF)
- Génère plusieurs quantizations:
  - **F16**: 6.4 GB (qualité originale)
  - **Q8_0**: 3.5 GB (qualité excellente)

**Configuration**:
```python
MODEL_DIR = "/workspace/model"
OUTPUT_DIR = "/workspace/model_gguf"
LLAMA_CPP_CONVERTER = "/workspace/llama.cpp/convert_hf_to_gguf.py"
```

**Output**: `model_gguf/ministral-3-3b-sudoku-{f16,q8_0}.gguf`

**Durée estimée**: 10-20 minutes

---

### Étape 6: Push GGUF vers HuggingFace

```bash
python 6_push_gguf_to_hf.py
```

**Ce script fait**:
- Upload les fichiers GGUF vers HuggingFace Hub
- Génère un README.md pour le repo GGUF
- Affiche l'URL du modèle

**Configuration**:
```python
GGUF_DIR = "/workspace/model_gguf"
HF_REPO = "applied-ai-subscr/ministral_3_3B_sudoku_gguf"
```

**Durée estimée**: 10-30 minutes selon la connexion

---

## 📁 Scripts principaux

### `1_ministral_3_rl_sudoku.py`
**Rôle**: Fine-tuning GRPO principal

**Fonctionnalités clés**:
- Implémentation du jeu Sudoku (`SudokuGame` class)
- Génération de puzzles aléatoires
- Reward functions pour GRPO:
  - `function_works`: Vérifie que le code est exécutable
  - `no_cheating`: Pénalise les imports externes
  - `strategy_succeeds`: Récompense les stratégies qui résolvent le puzzle
- Trainer GRPO avec 200 steps

**Hyperparamètres**:
```python
max_seq_length = 4096
lora_rank = 32
learning_rate = 5e-5
per_device_train_batch_size = 1
num_generations = 4
max_steps = 200
```

---

### `3_merge_for_vllm_v2.py`
**Rôle**: Merge LoRA + Base model

**Important**: Utilise la méthode Unsloth pour garantir la compatibilité:
1. Charge base model avec `FastVisionModel.from_pretrained()`
2. Applique structure PEFT avec `FastVisionModel.get_peft_model()`
3. Charge les poids LoRA depuis safetensors
4. Merge avec `save_pretrained_merged("merged_16bit")`

---

### `5_export_gguf_v2.py` ⭐ **WORKAROUND**
**Rôle**: Conversion GGUF (contournement du bug Unsloth)

**Contexte du workaround**:
1. Unsloth télécharge automatiquement llama.cpp
2. La fonction `model.push_to_hub_gguf()` d'Unsloth **plante**
3. Solution: Utiliser directement `convert_hf_to_gguf.py` de llama.cpp

**Dépendances requises** (à installer manuellement):
```bash
pip install gguf sentencepiece protobuf
```

**Pourquoi ça marche**:
- Conversion directe depuis safetensors (pas de GPU nécessaire)
- Plus rapide que la méthode Unsloth
- Script officiel maintenu par llama.cpp
- Support natif de Ministral3 (ajouté en décembre 2024)

---

## 🔧 Dépendances et workarounds

### Problème: Export GGUF via Unsloth plante

**Symptôme**:
```python
# Dans ministral_3_rl_sudoku.py (lignes 628-633)
model.push_to_hub_gguf(...)  # ❌ Plante
```

**Cause**: Bug dans l'implémentation Unsloth de l'export GGUF

**Solution**: Utiliser directement llama.cpp

```python
# 5_export_gguf_v2.py
subprocess.run([
    "python", "/workspace/llama.cpp/convert_hf_to_gguf.py",
    MODEL_DIR,
    "--outfile", output_file,
    "--outtype", "q8_0",
])
```

### Packages installés manuellement

| Package | Version | Pourquoi |
|---------|---------|----------|
| `gguf` | 0.17.1 | Format GGUF (requis par convert_hf_to_gguf.py) |
| `sentencepiece` | 0.2.1 | Tokenizer Mistral/Ministral |
| `protobuf` | 6.32.0 | Sérialization des données |

### llama.cpp téléchargé par Unsloth

Quand vous exécutez `1_ministral_3_rl_sudoku.py`, Unsloth télécharge automatiquement llama.cpp dans `/workspace/llama.cpp/`. On bénéficie ensuite de ce téléchargement pour notre workaround.

---

## 🚀 Déploiement

### Option 1: vLLM (recommandé pour production)

```bash
docker run -d \
  --name vllm_ministral_sudoku \
  --gpus all \
  --ipc=host \
  -p 8003:8000 \
  -v /workspace/ministral_3_sudoku_vllm:/model \
  nvcr.io/nvidia/vllm:25.09-py3 \
  vllm serve /model \
    --tokenizer_mode mistral \
    --config_format mistral \
    --load_format mistral \
    --gpu-memory-utilization 0.9
```

**Test**:
```bash
curl http://localhost:8003/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/model",
    "prompt": "Create a Sudoku solving strategy...",
    "max_tokens": 512,
    "temperature": 0.7
  }'
```

---

### Option 2: llama.cpp (plus léger)

```bash
cd /workspace/llama.cpp

# Compiler (si nécessaire)
make -j$(nproc)

# Lancer le serveur
./llama-server \
  -m /workspace/model_gguf/ministral-3-3b-sudoku-q8_0.gguf \
  -c 4096 \
  -ngl 99 \
  --port 8080
```

**Test**:
```bash
curl http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Create a Sudoku solving strategy...",
    "n_predict": 512
  }'
```

---

## 📂 Structure des dossiers

```
/workspace/
├── 1_ministral_3_rl_sudoku.py      # Fine-tuning GRPO
├── 2_check_lora.py                 # Vérification LoRA
├── 3_merge_for_vllm_v2.py          # Merge LoRA + Base
├── 4_save_to_hf_v2.py              # Push vers HF (complet)
├── 4-alt_push_manual.py            # Push vers HF (simple)
├── 5_export_gguf_v2.py             # Export GGUF (workaround)
├── 6_push_gguf_to_hf.py            # Upload GGUF vers HF
│
├── grpo_saved_lora/                # Adaptateurs LoRA (output step 1)
├── ministral_3_sudoku_vllm/        # Modèle mergé (output step 3)
├── model_gguf/                     # Fichiers GGUF (output step 5)
│   ├── ministral-3-3b-sudoku-f16.gguf   (6.4 GB)
│   └── ministral-3-3b-sudoku-q8_0.gguf  (3.5 GB)
│
├── deprecated/                     # Anciens scripts (ne plus utiliser)
│   ├── gguf_format.py
│   ├── save_to_hf.py
│   ├── merge_for_vllm.py
│   ├── merge_ministral_sudoku.py
│   └── export_to_gguf.py
│
├── untested/                       # Scripts non testés
│   └── merge_and_quantize_nvfp4.py
│
├── llama.cpp/                      # Téléchargé par Unsloth
│   └── convert_hf_to_gguf.py       # Script utilisé pour workaround
│
├── .env                            # HF_TOKEN
└── README.md                       # Ce fichier
```

---

## 📊 Comparaison des formats

| Format | Taille | Qualité | Vitesse | Usage |
|--------|--------|---------|---------|-------|
| **16bit merged** | ~6 GB | 100% | Rapide | vLLM production |
| **GGUF F16** | 6.4 GB | 100% | Rapide | llama.cpp qualité max |
| **GGUF Q8_0** | 3.5 GB | 99% | Très rapide | llama.cpp recommandé |
| **GGUF Q4_K_M** | ~2 GB | 95% | Rapide | llama.cpp léger |

---

## 🤝 Contributions

Ce projet utilise:
- **Unsloth**: Fine-tuning et merge ([unsloth.ai](https://unsloth.ai))
- **llama.cpp**: Conversion GGUF ([ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp))
- **TRL**: GRPO Trainer ([huggingface/trl](https://github.com/huggingface/trl))
- **Transformers**: Modèles HuggingFace

---

## 📝 Notes importantes

### ⚠️ Ne pas utiliser les fichiers dans `deprecated/`
Ces scripts sont des versions antérieures qui:
- Utilisaient l'export GGUF d'Unsloth (qui plante)
- Avaient des chemins incorrects
- Sont remplacés par les versions v2

### ✅ Workflow recommandé minimal

Pour un workflow complet minimal:
```bash
# 1. Fine-tuning + training
python 1_ministral_3_rl_sudoku.py

# 2. Merge
python 3_merge_for_vllm_v2.py

# 3. Export GGUF
python 5_export_gguf_v2.py

# 4. Push vers HF (optionnel)
python 4_save_to_hf_v2.py
python 6_push_gguf_to_hf.py
```

### 🐛 Troubleshooting

**Erreur "gguf module not found"**:
```bash
pip install gguf sentencepiece protobuf
```

**Erreur lors de l'export GGUF**:
- Vérifier que `/workspace/llama.cpp/convert_hf_to_gguf.py` existe
- Vérifier que le modèle mergé existe dans `MODEL_DIR`

**CUDA out of memory**:
- Réduire `per_device_train_batch_size` dans step 1
- Utiliser `load_in_4bit=True` pour le training

---

## 📄 License

Ce projet est basé sur:
- Unsloth (Apache 2.0)
- llama.cpp (MIT)
- Ministral-3 (Apache 2.0)

---

## 🎉 Résultats

Modèles disponibles sur HuggingFace:
- **Modèle mergé 16bit**: `applied-ai-subscr/ministral_3_sudoku_vllm`
- **Fichiers GGUF**: `applied-ai-subscr/ministral_3_3B_sudoku_gguf`

Le modèle fine-tuné est capable de générer des stratégies Python valides pour résoudre des puzzles Sudoku avec un taux de réussite significativement amélioré par rapport au modèle de base.
