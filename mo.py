import modal
import zipfile
import os

app = modal.App("zip-checkpoint")
vol = modal.Volume.from_name("my-volume")

@app.function(volumes={"/mnt/vol": vol})
def zip_and_read(path: str):
    full_path = f"/mnt/vol/{path}"
    zip_path = f"{full_path}.zip"
    
    # If already zipped, skip creating again
    if not os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(full_path, arcname=os.path.basename(path))

    with open(zip_path, "rb") as f:
        return f.read()
