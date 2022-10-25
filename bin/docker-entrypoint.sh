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

# Add PyGeNN user with matching user and group ID
groupadd -g $GROUP_ID pygenn
useradd --shell /bin/bash -u $USER_ID -g pygenn -o -c "" -m pygenn
export HOME=/home/pygenn

# If interactive command passed, pass remainder of arguments to python interpreter
if [[ "$1" = "interactive" ]]; then
    shift
    exec gosu pygenn:pygenn python3 "$@"
# Otherwise, if notebook is passes, launch notebook
elif [[ "$1" = "notebook" ]]; then
    exec gosu pygenn:pygenn /usr/bin/jupyter-notebook --ip=0.0.0.0 --port=8080 --no-browser
# Otherwise, execute arguments
else
    exec gosu pygenn:pygenn "$@"
fi
