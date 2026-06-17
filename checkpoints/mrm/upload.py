"""
Upload the locally-trained MRM checkpoint to an ANONYMOUS Hugging Face model repo
(for double-blind review).

Prerequisites
-------------
  pip install huggingface_hub
  # Log in with an ANONYMOUS account's write token (do NOT reuse an account
  # that identifies you; create a fresh one with a neutral username + email):
  huggingface-cli login

Anonymity checklist (read before running)
------------------------------------------
  * REPO_ID must use the anonymous account / a neutral repo name.
  * The checkpoint 55.pth was scanned and contains no identifying paths/usernames.
  * The model card (HF_MODEL_CARD.md) is anonymous; it is uploaded as README.md.
  * After acceptance (de-anonymized), either move to an identified repo or simply
    reveal authorship; the file (and its SHA256) stays the same.

Usage
-----
  1. Edit REPO_ID below.
  2. python upload.py
  3. Put the printed URL into CHECKPOINTS.md and the paper's repro statement.
"""
from pathlib import Path
from huggingface_hub import HfApi, create_repo

# --- EDIT THIS to your anonymous repo, e.g. "anon-2026-xyz/partial-spoof-mrm-audit"
REPO_ID = "ANON-USERNAME/partial-spoof-mrm-audit"

HERE = Path(__file__).resolve().parent
CKPT = HERE / "55.pth"
CARD = HERE / "HF_MODEL_CARD.md"
EXPECTED_SHA256 = "5b753752f7c25370c6abf973f69f58e100dad4b5d3ea035872335358a876fdd1"


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    assert CKPT.exists(), f"missing checkpoint: {CKPT}"
    got = _sha256(CKPT)
    assert got == EXPECTED_SHA256, f"SHA256 mismatch:\n  got {got}\n  exp {EXPECTED_SHA256}"
    if "ANON-USERNAME" in REPO_ID:
        raise SystemExit("Set REPO_ID to your anonymous Hugging Face repo first.")

    api = HfApi()
    create_repo(REPO_ID, repo_type="model", exist_ok=True, private=False)
    api.upload_file(path_or_fileobj=str(CARD), path_in_repo="README.md",
                    repo_id=REPO_ID, repo_type="model")
    api.upload_file(path_or_fileobj=str(CKPT), path_in_repo="55.pth",
                    repo_id=REPO_ID, repo_type="model")
    print(f"Done. Public (anonymous) URL: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
