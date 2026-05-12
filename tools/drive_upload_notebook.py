"""Drive API で notebook を upload + Colab share link 生成 (= seamless 連携).

orbit-wars/tools/drive_upload_for_colab.py をベースに generic 化。
Drive 上の同名 file を update (= 重複防止)、 私が再実行すれば Drive 内 file が
overwrite されて Colab で reload するだけで反映。

Usage:
    /home/yusuke_kaya/projects/kaggle/orbit-wars/.venv/bin/python \\
        /home/yusuke_kaya/projects/kaggle/ROGII/tools/drive_upload_notebook.py \\
        --notebook notebooks/2026-05-13-ravaghi-colab-rerun.ipynb \\
        --folder rogii

(ROGII venv は uv 管理で pip なし、 orbit-wars venv に google-api-python-client
 既存。 cross-repo で再利用)

Output:
    https://colab.research.google.com/drive/<file_id>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except AttributeError:
    pass

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

CLIENT_SECRET = (
    "/mnt/c/Users/yusuke kaya/Downloads/"
    "client_secret_757799584676-gmb117cjvjg22nhevvgc3mcihvk4pe7u.apps.googleusercontent.com.json"
)
SCOPES = ["https://www.googleapis.com/auth/drive.file"]
# Reuse the orbit-wars token (= same OAuth client, same user, same scope).
# Per-app token if you want full isolation: change to f"~/.config/<folder>_google_token.json".
TOKEN_PATH = Path.home() / ".config" / "orbit_wars_google_token.json"


def get_creds():
    creds = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            print("\n=== OAuth consent required ===", flush=True)
            print(
                "Browser で URL を開き Google account で許可 → localhost:8080 redirect 待ち。",
                flush=True,
            )
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET, SCOPES)
            creds = flow.run_local_server(port=8080, open_browser=False)
        TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(TOKEN_PATH, "w") as f:
            f.write(creds.to_json())
        print(f"token saved: {TOKEN_PATH}", flush=True)
    return creds


def find_or_create_folder(service, name: str, parent_id: str | None = None) -> str:
    query = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    if parent_id:
        query += f" and '{parent_id}' in parents"
    resp = service.files().list(q=query, fields="files(id, name)").execute()
    items = resp.get("files", [])
    if items:
        print(f"folder '{name}' exists: {items[0]['id']}", flush=True)
        return items[0]["id"]
    body = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
    if parent_id:
        body["parents"] = [parent_id]
    folder = service.files().create(body=body, fields="id").execute()
    print(f"folder '{name}' created: {folder['id']}", flush=True)
    return folder["id"]


def upload_file(service, local_path: Path, parent_id: str, mime: str) -> dict:
    name = local_path.name
    query = f"name='{name}' and '{parent_id}' in parents and trashed=false"
    resp = service.files().list(q=query, fields="files(id, name)").execute()
    existing = resp.get("files", [])
    media = MediaFileUpload(str(local_path), mimetype=mime, resumable=True)
    if existing:
        fid = existing[0]["id"]
        print(f"updating existing '{name}' ({fid})", flush=True)
        result = (
            service.files()
            .update(fileId=fid, media_body=media, fields="id, webViewLink, name, size")
            .execute()
        )
    else:
        body = {"name": name, "parents": [parent_id]}
        size_mb = local_path.stat().st_size / 1024**2
        print(f"uploading new '{name}' ({size_mb:.2f} MB) ...", flush=True)
        result = (
            service.files()
            .create(body=body, media_body=media, fields="id, webViewLink, name, size")
            .execute()
        )
    size_mb = int(result.get("size", 0)) / 1024**2
    print(f"  done: id={result['id']} size={size_mb:.2f} MB", flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload notebook to Drive + emit Colab link")
    parser.add_argument(
        "--notebook",
        required=True,
        help="Path to .ipynb (relative to repo root or absolute)",
    )
    parser.add_argument(
        "--folder",
        required=True,
        help="Drive folder name (e.g., 'rogii', 'orbit-wars')",
    )
    parser.add_argument(
        "--extra-file",
        action="append",
        default=[],
        help="Optional additional files to upload (= warm-start zips etc.). Repeatable.",
    )
    args = parser.parse_args()

    notebook = Path(args.notebook).resolve()
    if not notebook.exists():
        print(f"ERROR: notebook not found at {notebook}", file=sys.stderr)
        return 1
    if not Path(CLIENT_SECRET).exists():
        print(f"ERROR: client_secret.json not found at {CLIENT_SECRET}", file=sys.stderr)
        return 1

    try:
        creds = get_creds()
    except Exception as exc:
        print(f"OAuth failed: {exc}", file=sys.stderr)
        return 1

    service = build("drive", "v3", credentials=creds)

    try:
        folder_id = find_or_create_folder(service, args.folder)
    except HttpError as exc:
        print(f"folder error: {exc}", file=sys.stderr)
        return 1

    print(f"\n=== Upload notebook {notebook.name} ===", flush=True)
    nb_result = upload_file(service, notebook, folder_id, "application/x-ipynb+json")

    for extra in args.extra_file:
        extra_path = Path(extra).resolve()
        if not extra_path.exists():
            print(f"WARN: extra file not found: {extra_path}", file=sys.stderr)
            continue
        mime = "application/zip" if extra_path.suffix == ".zip" else "application/octet-stream"
        print(f"\n=== Upload extra {extra_path.name} ===", flush=True)
        upload_file(service, extra_path, folder_id, mime)

    colab_url = f"https://colab.research.google.com/drive/{nb_result['id']}"
    drive_folder_url = f"https://drive.google.com/drive/folders/{folder_id}"

    print("\n" + "=" * 70, flush=True)
    print("DONE — Drive seamless link (= 私が再 upload すれば Colab で reload で反映)", flush=True)
    print("=" * 70, flush=True)
    print(f"\nDrive folder: {drive_folder_url}", flush=True)
    print(f"\n★ Open in Colab: {colab_url}\n", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
