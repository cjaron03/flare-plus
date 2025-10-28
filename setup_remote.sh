#!/bin/bash
# setup github remote and push initial commit
#
# usage: ./setup_remote.sh <github-username> <repo-name>
#
# example: ./setup_remote.sh jaroncabral flare-plus

if [ "$#" -ne 2 ]; then
    echo "usage: $0 <github-username> <repo-name>"
    echo "example: $0 jaroncabral flare-plus"
    exit 1
fi

USERNAME=$1
REPONAME=$2
REMOTE_URL="git@github.com:${USERNAME}/${REPONAME}.git"

echo "setting up remote: ${REMOTE_URL}"

# add remote
git remote add origin "${REMOTE_URL}"

# verify remote
echo ""
echo "remote added successfully:"
git remote -v

echo ""
echo "ready to push! run this command when ready:"
echo ""
echo "  git push -u origin main"
echo ""

