import os


def main():
    # ── Configuration ───────────────────────────────────────────────
    csv_dir    = "/nethome/pjajoria/Github/MolICL-Eval/datasets/InContextPrompts"
    tasks_dir  = "/nethome/pjajoria/Github/lm-evaluation-harness/lm_eval/tasks/ICL_for_MPP"
    dataset_repo = "pushkarjajoria/ICL_for_MPP"
    model_name = "gemma2-9b"
    # ─────────────────────────────────────────────────────────────────

    # Ensure tasks directory exists
    os.makedirs(tasks_dir, exist_ok=True)
    print(f"[INFO] Writing YAML files to: {tasks_dir}")

    # List CSV files in csv_dir
    for fname in os.listdir(csv_dir):
        if not fname.endswith('.csv'):
            continue
        # Derive task name (strip extension)
        task_name = os.path.splitext(fname)[0]
        task_name = task_name.split('_')
        task_name.insert(1, model_name)
        task_name = "_".join(task_name)
        yaml_name = f"{task_name}.yaml"
        yaml_path = os.path.join(tasks_dir, yaml_name)

        # YAML content
        yaml_content = f"""
# Auto-generated task for {task_name}
task: {task_name}
dataset_path: {dataset_repo}
dataset_name: null
dataset_kwargs:
  data_files:
    test: {fname}
test_split: test

doc_to_text: prompt
doc_to_target: label
doc_to_choice: ["0", "1"]

metric_list:
  - metric: acc
""".lstrip()

        # Write to file
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        print(f"[WRITE] {yaml_name}")

    print("[DONE] YAML generation complete.")


if __name__ == '__main__':
    main()
