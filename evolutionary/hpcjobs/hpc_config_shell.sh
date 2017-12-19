cat >> .gbarrc <<EOF
MODULES=python3
EOF
cat >> .profile <<EOF
# Setup local python3
if tty -s ; then
export PYTHONPATH=
source ~/stdpy3/bin/activate
fi
EOF