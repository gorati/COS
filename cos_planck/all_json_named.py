import json, pathlib, zipfile

root = pathlib.Path(".")
zip_path = root / "cos_results_renamed_jsons.zip"

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for summary in root.rglob("summary.json"):
        data = json.loads(summary.read_text(encoding="utf-8"))
        out_name = data.get("out") or summary.parent.name
        out_name = pathlib.Path(out_name).name
        new_name = f"{out_name}_summary.json"
        z.write(summary, arcname=new_name)
        print(f"Added {summary} as {new_name}")

print(f"\nKész: {zip_path}")