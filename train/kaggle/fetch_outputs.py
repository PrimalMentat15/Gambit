"""Download a Kaggle training session's run directory, with per-file retries.

    python train/kaggle/fetch_outputs.py <owner/kernel> <dest_dir> [run_name]

Only files under runs/<run_name>/ are fetched (default m12). Kaggle's file
server intermittently returns HTML 5xx pages, which `kaggle kernels output`
saves as if they were the file; here each file is retried until it is not an
error page and every checkpoint is a valid torch zip archive.
"""
import os
import sys
import time
import zipfile

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

KERNEL, DEST = sys.argv[1], sys.argv[2]
RUN = sys.argv[3] if len(sys.argv) > 3 else "m12"
WANT = (f"/runs/{RUN}/",)
SKIP = (f"/runs/{RUN}/STOP",)
os.makedirs(DEST, exist_ok=True)

api = KaggleApi(); api.authenticate()
owner, slug = KERNEL.split("/")
files, token = [], None
with api.build_kaggle_client() as k:
    while True:
        req = ApiListKernelSessionOutputRequest()
        req.user_name, req.kernel_slug, req.page_size = owner, slug, 200
        if token:
            req.page_token = token
        resp = k.kernels.kernels_api_client.list_kernel_session_output(req)
        files += [(f.file_name, f.url) for f in resp.files]
        token = resp.next_page_token
        if not token:
            break

todo = [(n, u) for n, u in files if any(w in "/" + n for w in WANT) and not any(s in "/" + n for s in SKIP)]
print(f"{len(files)} output files, {len(todo)} wanted", flush=True)


def valid(path, name):
    if os.path.getsize(path) < 1024 and not name.endswith(("meta.json",)):
        return False
    with open(path, "rb") as f:
        head = f.read(64)
    if head.lstrip().lower().startswith((b"<html", b"<!doctype")):
        return False
    if name.endswith(".pt"):
        return zipfile.is_zipfile(path)          # torch.save writes a zip archive
    return True


failed = []
for name, url in todo:
    out = os.path.join(DEST, os.path.basename(name))
    if os.path.exists(out) and valid(out, name):
        print(f"ok (cached) {name}", flush=True)
        continue
    for attempt in range(1, 9):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                tmp = out + ".part"
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            if valid(tmp, name):
                os.replace(tmp, out)
                print(f"ok  {name} ({os.path.getsize(out)/1e6:.1f} MB, attempt {attempt})", flush=True)
                break
            os.remove(tmp)
            raise RuntimeError("got an error page / invalid file")
        except Exception as e:
            print(f"  retry {attempt} {name}: {e}", flush=True)
            time.sleep(min(60, 5 * attempt))
    else:
        failed.append(name)

print("FAILED:", failed or "none", flush=True)
