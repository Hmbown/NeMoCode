# Scoped NeMoCode environment. Usage: `source activate.sh`
# Activates the project venv and loads the NVIDIA API key.
# Run `deactivate` to exit the venv; the API key stays set until you open a new shell.

_here="${0:A:h}"
# Fallback for bash (${0:A} is zsh-only)
if [ -z "$_here" ] || [ ! -d "$_here" ]; then
    _here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
fi

if [ -f "$_here/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$_here/.venv/bin/activate"
else
    echo "activate.sh: .venv not found at $_here/.venv" >&2
    return 1 2>/dev/null || exit 1
fi

if [ -f "$HOME/.config/nemocode/env.sh" ]; then
    # shellcheck disable=SC1091
    source "$HOME/.config/nemocode/env.sh"
fi

unset _here
