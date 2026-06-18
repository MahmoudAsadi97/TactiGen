"""
TactiGen — project structure generator.
Creates every directory and an empty __init__.py in each Python package.
Run: python setup_project.py
"""
import os

DIRECTORIES = [
    "app",
    "preprocessing",
    "localization",
    "action_anticipation",
    "tactical_analysis",
    "evidence",
    "rag",
    "report_generation",
    "report_generation/prompts",
    "visualization",
    "orchestration",
    "storage",
    "evaluation",
    "knowledge/tactical_kb",
    "knowledge/faiss_index",
    "data",
    "data/soccernet",
    "data/statsbomb",
    "data/sample_clips",
    "docs",
    "outputs",
    "outputs/models",
    "outputs/reports",
    "outputs/visualizations",
    "outputs/run_logs",
]

PACKAGES = [
    "preprocessing",
    "localization",
    "action_anticipation",
    "tactical_analysis",
    "evidence",
    "rag",
    "report_generation",
    "visualization",
    "orchestration",
    "storage",
    "evaluation",
]

# Only seed .gitkeep into dirs that should persist in git but may be empty.
# (outputs/* are gitignored runtime dirs and are recreated by this script.)
GITKEEP_DIRS = [
    "data/soccernet",
    "data/statsbomb",
    "data/sample_clips",
    "knowledge/tactical_kb",
    "knowledge/faiss_index",
]


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    print(f"Setting up TactiGen project structure in: {root}")

    for d in DIRECTORIES:
        path = os.path.join(root, d)
        os.makedirs(path, exist_ok=True)
        print(f"  [dir]  created: {d}")

    for pkg in PACKAGES:
        init_path = os.path.join(root, pkg, "__init__.py")
        if not os.path.exists(init_path):
            with open(init_path, "w", encoding="utf-8") as f:
                f.write("")
        print(f"  [pkg]  ensured: {pkg}/__init__.py")

    for d in GITKEEP_DIRS:
        keep = os.path.join(root, d, ".gitkeep")
        os.makedirs(os.path.dirname(keep), exist_ok=True)
        if not os.path.exists(keep):
            with open(keep, "w", encoding="utf-8") as f:
                f.write("")
        print(f"  [keep] ensured: {d}/.gitkeep")

    print("Project structure setup complete.")


if __name__ == "__main__":
    main()
