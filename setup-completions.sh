#!/usr/bin/env bash
set -euo pipefail

# Configure Bash autocompletion for az / kubectl / minikube.
# Safe to re-run; it will skip duplicate bashrc entries.

COMPLETIONS_DIR="$HOME/.bash_completion.d"
BASHRC="$HOME/.bashrc"

echo "[info] Ensuring bash-completion package present..."
if ! type _init_completion >/dev/null 2>&1; then
  if [ -f /etc/os-release ]; then
    . /etc/os-release
    case "$ID" in
      ubuntu|debian)
        sudo apt-get update -y && sudo apt-get install -y bash-completion ;;
      centos|rhel|rocky|almalinux|fedora)
        sudo yum install -y bash-completion || sudo dnf install -y bash-completion ;;
      *)
        echo "[warn] Unknown distro $ID. Install bash-completion manually if not working."
        ;;
    esac
  fi
fi

mkdir -p "$COMPLETIONS_DIR"

add_source_line() {
  local LINE="$1"
  if ! grep -Fq "$LINE" "$BASHRC" 2>/dev/null; then
    printf '\n%s\n' "$LINE" >> "$BASHRC"
    echo "[info] Added to .bashrc: $LINE"
  else
    echo "[info] .bashrc already contains: $LINE"
  fi
}

echo "[info] Generating az completion..."
if command -v az >/dev/null 2>&1; then
  az completion -s bash > "$COMPLETIONS_DIR/az"
else
  echo "[warn] az CLI not found; skip (install then re-run)."
fi

echo "[info] Generating kubectl completion..."
if command -v kubectl >/dev/null 2>&1; then
  kubectl completion bash > "$COMPLETIONS_DIR/kubectl"
else
  echo "[warn] kubectl not found; skip."
fi

echo "[info] Generating minikube completion..."
if command -v minikube >/dev/null 2>&1; then
  minikube completion bash > "$COMPLETIONS_DIR/minikube"
else
  echo "[warn] minikube not found; skip."
fi

# Ensure bash-completion main script is sourced (varies by distro)
add_source_line "# >>> nv-playground completions >>>"
add_source_line "[ -f /usr/share/bash-completion/bash_completion ] && . /usr/share/bash-completion/bash_completion"
add_source_line "for f in $COMPLETIONS_DIR/*; do [ -r \"$f\" ] && . \"$f\"; done"
add_source_line "# <<< nv-playground completions <<<"

echo "[info] Done. Start a new shell or run: source ~/.bashrc"
echo "[info] Test examples: 'az <TAB>', 'kubectl get <TAB>', 'minikube <TAB>'"
