import os
import urllib.request


THREE_JS_FILES = [
    "https://cdnjs.cloudflare.com/ajax/libs/three.js/0.172.0/three.core.min.js",
    "https://cdnjs.cloudflare.com/ajax/libs/three.js/0.172.0/three.module.min.js",
]


def main() -> None:
    extension_path = os.path.dirname(__file__)
    js_lib_path = os.path.join(extension_path, "web", "js", "lib")
    os.makedirs(js_lib_path, exist_ok=True)

    for url in THREE_JS_FILES:
        file_name = os.path.basename(url)
        file_path = os.path.join(js_lib_path, file_name)

        if os.path.exists(file_path):
            print(f"[DiT360] Exists: {file_path}")
            continue

        print(f"[DiT360] Downloading: {url}")
        urllib.request.urlretrieve(url, file_path)
        print(f"[DiT360] Saved: {file_path}")

    print("[DiT360] Three.js install complete.")


if __name__ == "__main__":
    main()

