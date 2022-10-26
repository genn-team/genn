#!/bin/bash
# Run commands in the Docker container with a particular UID and GID.
# The idea is to run the container like
#   docker run -i \
#     -v `pwd`:/work \
#     -e LOCAL_USER_ID=`id -u $USER` \
#     -e LOCAL_GROUP_ID=`id -g $USER` \
#     image-name bash
# where the -e flags pass the env vars to the container, which are read by this script.
# By setting copying this script to the container and setting it to the
# ENTRYPOINT, and subsequent commands run in the container will run as the user
# who ran `docker` on the host, and so any output files will have the correct
# permissions.

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
    exec gosu genn:genn /bin/bash -c "cd \"$CWD\" && python3 \"$@\""
# Otherwise, if notebook is passes
elif [[ "$1" = "notebook" ]]; then
    # Extract working directory from next command line argument, otherwise use home
    CWD=${2:-$HOME}
    exec gosu genn:genn /bin/bash -c "/usr/local/bin/jupyter-notebook --ip=0.0.0.0 --port=8080 --no-browser --notebook-dir=\"$CWD\""
# Otherwise, execute arguments
else
    exec gosu genn:genn "$@"
fi
