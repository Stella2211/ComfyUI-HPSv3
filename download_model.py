"""Download a model snapshot in the dedicated inference environment.

This process intentionally keeps Hugging Face imports out of ComfyUI's Python
environment.  The parent process owns validation and publication of the
staged directory.
"""

import sys


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: download_model.py REPOSITORY LOCAL_DIR")
    from huggingface_hub import snapshot_download

    snapshot_download(sys.argv[1], local_dir=sys.argv[2])


if __name__ == "__main__":
    main()
