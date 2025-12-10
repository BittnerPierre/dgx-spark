import os
from huggingface_hub import HfApi

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise SystemExit("❌ HF_TOKEN manquant dans les variables d'environnement")

REPO_ID = "applied-ai-subscr/ministral_3_3B_sudoku_vllm"
LOCAL_DIR = "/workspace/model"

api = HfApi(token=HF_TOKEN)

# 1) Créer le repo (ou ne rien dire s'il existe déjà)
api.create_repo(
    repo_id=REPO_ID,
    private=False,
    exist_ok=True,
)

# 2) Uploader tout le dossier /workspace/model
api.upload_folder(
    repo_id=REPO_ID,
    folder_path=LOCAL_DIR,
    repo_type="model",
)

print(f"✅ Upload terminé sur https://huggingface.co/{REPO_ID}")

