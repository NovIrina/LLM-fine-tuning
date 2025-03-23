#!/bin/bash
source config/common.sh

set -x

echo -e '\n'
echo 'Running mypy check...'

configure_script

directories=$(get_project_directories)

for directory in $directories; do
  python3 -m mypy "${directory}"

  check_if_failed
done

echo "Mypy check passed."
