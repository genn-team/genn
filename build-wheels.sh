#!/bin/bash
set -e -u -x

PLAT=manylinux2014_x86_64

function repair_wheel {
    wheel="$1"
    if ! auditwheel show "$wheel"; then
        echo "Skipping non-platform wheel $wheel"
    else
        auditwheel repair "$wheel" --plat "$PLAT" -w /opt/genn/dist/
    fi
}


# # Install a system package required by our library
# yum install -y atlas-devel

# Compile wheels
for PYBIN in /opt/python/*/bin; do
    # "${PYBIN}/pip" install -r /io/dev-requirements.txt
    "${PYBIN}/pip" install numpy swig
    # "${PYBIN}/pip" wheel /opt/genn/ --no-deps -w dist/
    make install -j `lscpu -p | egrep -v '^#' | sort -u -t, -k 2,4 | wc -l`
    make DYNAMIC=1 LIBRARY_DIRECTORY=${GENN_PATH}/pygenn/genn_wrapper/ -j `lscpu -p | egrep -v '^#' | sort -u -t, -k 2,4 | wc -l`
    # RUN python3 setup.py develop
    "${PYBIN}/python" setup.py bdist_wheel
    "${PYBIN}/python" setup.py bdist_wheel
done

# Bundle external shared libraries into the wheels
for whl in dist/*.whl; do
    repair_wheel "$whl"
done

# # Install packages and test
# for PYBIN in /opt/python/*/bin/; do
#     "${PYBIN}/pip" install python-manylinux-demo --no-index -f /opt/genn/dist
#     (cd "$HOME"; "${PYBIN}/nosetests" pymanylinuxdemo)
# done