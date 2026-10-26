"""
Utility to clear all prior test runs in the W&B project for a clean slate.
"""
import wandb

def clear_runs():
    api = wandb.Api()
    entity_project = "raviprakashshrivastav7-iiit-ranchi/comic-vlm-translation"
    try:
        runs = api.runs(entity_project)
        count = 0
        for r in runs:
            r.delete()
            count += 1
        print(f"[SUCCESS] Successfully deleted {count} run(s) from W&B project: {entity_project}")
    except Exception as e:
        print(f"[NOTE] Could not list/delete runs: {e}")

if __name__ == "__main__":
    clear_runs()
