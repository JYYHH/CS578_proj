#!/usr/bin/env bash
set -euo pipefail

# Auto-run: method x base model (+hyperparams) x dataset.
# Extend by editing the arrays below.

PYTHON_BIN="${PYTHON_BIN:-python}"
ENTRYPOINT="${ENTRYPOINT:-main.py}"

# Datasets grouped by task type (used to skip invalid base-model/task combos).
BINARY_DATASETS=(
  "adult"
)
MULTICLASS_DATASETS=(
  "mnist"
)
REGRESSION_DATASETS=(
  "communities_crime"
  "ca_housing"
  # "allstate"
  # "sberbank"
)

# Methods
METHODS=(
  "Single"
  "AdaBoost"
  "Bagging"
  "GradBoost"
)

# Shared runtime args.
COMMON_ARGS=(
  "--n_estimators" "100"
  "--seed" "42"
  "--test_size" "0.2"
)

# Base-model definitions:
#   key format      : "<base_model>|<task_scope>|<extra args...>"
#   task_scope      : class | reg | all
#   extra args      : any args needed by that base model
BASE_MODEL_SPECS=(
  "DecisionTree|all|--max_depth 1"
  "DecisionTree|all|--max_depth 3"
  "DecisionTree|all|--max_depth 5"
  "DecisionTree|all|--max_depth 10"
  "DecisionTree|all|--max_depth 100000"
  # "SVM|class|--kernel rbf --C 1.0"
  "Ridge|reg|--alpha 0.1"
  "Ridge|reg|--alpha 0.3"
  "Ridge|reg|--alpha 1.0"
  "Ridge|reg|--alpha 3.0"
  "Ridge|reg|--alpha 10.0"
  # "LR|class|--C 1.0"
  "NB|class|--var_smoothing 1e-9"
  "MNB|class|--alpha 0.1"
  "MNB|class|--alpha 0.3"
  "MNB|class|--alpha 1.0"
  "MNB|class|--alpha 3.0"
  "MNB|class|--alpha 10.0"
)

run_one() {
  local method="$1"
  local dataset="$2"
  local base_model="$3"
  shift 3

  local cmd=(
    "$PYTHON_BIN" "$ENTRYPOINT"
    "--method" "$method"
    "--dataset" "$dataset"
    "--base_model" "$base_model"
    "${COMMON_ARGS[@]}"
    "$@"
  )

  echo "=============================================================="
  echo "Running: ${cmd[*]}"
  "${cmd[@]}"
}

run_for_task_group() {
  local method="$1"
  local task_scope="$2"
  local base_model="$3"
  shift 3
  local extra_args=("$@")

  if [[ "$task_scope" == "class" || "$task_scope" == "all" ]]; then
    for ds in "${BINARY_DATASETS[@]}" "${MULTICLASS_DATASETS[@]}"; do
      run_one "$method" "$ds" "$base_model" "${extra_args[@]}"
    done
  fi

  if [[ "$task_scope" == "reg" || "$task_scope" == "all" ]]; then
    for ds in "${REGRESSION_DATASETS[@]}"; do
      run_one "$method" "$ds" "$base_model" "${extra_args[@]}"
    done
  fi
}

main() {
  for method in "${METHODS[@]}"; do
    for spec in "${BASE_MODEL_SPECS[@]}"; do
      IFS='|' read -r base_model task_scope extra <<< "$spec"

      # shellcheck disable=SC2206
      extra_args=( $extra )
      run_for_task_group "$method" "$task_scope" "$base_model" "${extra_args[@]}"
    done
  done

  echo "All combinations finished."
}

main "$@"
