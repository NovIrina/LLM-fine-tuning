configure_script() {
  if [[ "$OSTYPE" == "linux-gnu"* || "$OSTYPE" == "darwin"* ]]; then
    if [ -f "venv/bin/activate" ]; then
      source venv/bin/activate
      echo "Virtual environment activated (Linux/macOS)."
    else
      echo "Virtual environment not found at venv/bin/activate."
      return 1
    fi
    export PYTHONPATH=$(pwd):$PYTHONPATH
    echo "PYTHONPATH set to $(pwd):$PYTHONPATH"
    which python
    python -m pip list
  elif [[ "$OSTYPE" == "msys"* || "$OSTYPE" == "cygwin"* ]]; then
    if [ -f "venv/Scripts/activate" ]; then
      source venv/Scripts/activate
      echo "Virtual environment activated (Windows)."
    else
      echo "Virtual environment not found at venv/Scripts/activate."
      return 1
    fi
    export PYTHONPATH=$(pwd)
    echo "PYTHONPATH set to $(pwd)"
    which python
    python -m pip list
  else
    echo "Unsupported OS: $OSTYPE"
    return 1
  fi
}

check_if_failed() {
  if [[ $? -ne 0 ]]; then
    echo "Check failed."
    exit 1
  else
    echo "Check passed."
  fi
}

get_project_directories() {
  local directories=('src' 'web_site')
  echo ${directories[@]}
}
