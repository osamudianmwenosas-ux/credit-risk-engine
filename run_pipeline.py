from pathlib import Path
from src.credit_risk.pipeline import run

if __name__ == "__main__":
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    results = run(n=25000, seed=42, output_dir=out)
    print(results)
