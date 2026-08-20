#!/bin/sh
set -eu

script_directory=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repository=$(CDPATH= cd -- "$script_directory/.." && pwd)
cd "$repository"

failed=0

report() {
  printf 'release-check: %s\n' "$1" >&2
  failed=1
}

for generated in build config.mk dist .pytest_cache; do
  if [ -e "$generated" ]; then
    report "generated path remains: $generated (run 'make distclean')"
  fi
done

artifacts=$(find . -path './.git' -prune -o -type f \( \
    -name '*.onnx' -o -name '*.engine' -o -name '*.timing' \
    -o -name '*.mod' -o -name '*.smod' -o -name '*.o' \
    -o -name '*.a' -o -name '*.so' -o -name '*.so.*' \
    -o -name '*.pyc' -o -name '*.ses' \) -print)
if [ -n "$artifacts" ]; then
  printf '%s\n' "$artifacts" >&2
  report 'generated binary or model artifacts remain'
fi

absolute_links=$(find . -path './.git' -prune -o -type l -lname '/*' -print)
if [ -n "$absolute_links" ]; then
  printf '%s\n' "$absolute_links" >&2
  report 'absolute symbolic links remain'
fi

home_root=home
mac_user_root=Users
private_path_pattern="(/${home_root}/[^/[:space:]]+|/${mac_user_root}/[^/[:space:]]+|[A-Za-z]:\\\\${mac_user_root}\\\\[^\\\\[:space:]]+)"
private_paths=$(find . -path './.git' -prune -o -type f -print0 | \
  xargs -0 grep -IEn "$private_path_pattern" 2>/dev/null || true)
if [ -n "$private_paths" ]; then
  printf '%s\n' "$private_paths" >&2
  report 'a user-specific absolute path remains'
fi

if [ -n "${FORTONNX_PRIVATE_PATTERNS:-}" ]; then
  private_matches=$(find . -path './.git' -prune -o -type f -print0 | \
    xargs -0 grep -IFn "$FORTONNX_PRIVATE_PATTERNS" 2>/dev/null || true)
  if [ -n "$private_matches" ]; then
    printf '%s\n' "$private_matches" >&2
    report 'FORTONNX_PRIVATE_PATTERNS matched release content'
  fi
fi

if [ "$failed" -ne 0 ]; then
  exit 1
fi
printf '%s\n' 'FortONNX release hygiene check passed'
