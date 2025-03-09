#!/bin/bash
source config/common.sh

set -ex

echo -e '\n'
echo 'Running isort check...'

configure_script

python3 -m isort --check-only main.py

directories=$(get_project_directories)

for directory in $directories; do
  python3 -m isort --check-only "${directory}"

  check_if_failed
done

echo "Isort check passed."
