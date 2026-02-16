"""Clone notebook_cancer_dataset.ipynb to a new notebook with same structure, no outputs."""
import json
import sys

src_path = "Notebooks/notebook_cancer_dataset.ipynb"
dst_path = "Notebooks/notebook_cancer_dataset_replica.ipynb"

with open(src_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Clear execution_count and outputs from all cells
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

with open(dst_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Created", dst_path)
