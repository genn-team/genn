#!/bin/bash
# Read desired user and group ID from environment varibles (typically set on docker command line with -e)
USER_ID=${LOCAL_USER_ID:-9001}
GROUP_ID=${LOCAL_GROUP_ID:-$USER_ID}

# Add GeNN user with matching user and group ID
groupadd -g $GROUP_ID genn
useradd --shell /bin/bash -u $USER_ID -g genn -o -c "" -m genn
export HOME=/home/genn

# If script command passed
if [[ "$1" = "script" ]]; then
    # Shift script command itself off arguments
    shift

    # Change to directory script is in and launch
    # **YUCK** this should not really be necessary but PyGeNN does
    # not work nicely running scripts not in working directory
    CWD=$(dirname "$1")
    cd "$CWD"
    exec gosu genn:genn python3 "$@"
# Otherwise, if notebook is passes
elif [[ "$1" = "notebook" ]]; then
    # Extract notebook directory from next command line argument, otherwise use home
    CWD=${2:-$HOME}
    exec gosu genn:genn /usr/local/bin/jupyter-notebook --ip=0.0.0.0 --port=8080 --no-browser --notebook-dir="$CWD"
# Otherwise, change directory to home directory and execute arguments
else
    cd $HOME
    exec gosu genn:genn "$@"
fi
