#!/bin/bash
source config/common.sh

set -ex

echo -e '\n'
echo 'Running black check...'

configure_script

python -m black --check main.py

directories=$(get_project_directories)

for directory in $directories; do
  python -m black --check "${directory}"

  check_if_failed
done

echo "Black check passed."
