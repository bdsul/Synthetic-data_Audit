import json
from pathlib import Path


def main() -> None:
    nb_path = Path("Notebooks/adult_income.ipynb")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))

    replaced_cells = 0
    replaced_count = 0

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        if not src:
            continue
        joined = "".join(src)
        if "wine_data" not in joined:
            continue

        new_joined = joined.replace("wine_data", "adult_data")
        if new_joined != joined:
            replaced_cells += 1
            replaced_count += joined.count("wine_data")
            cell["source"] = [line + "\n" for line in new_joined.splitlines()]

    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Updated {nb_path}: replaced {replaced_count} occurrence(s) across {replaced_cells} code cell(s).")


if __name__ == "__main__":
    main()

