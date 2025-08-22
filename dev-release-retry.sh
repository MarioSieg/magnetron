#!/usr/bin/env bash
set -euo pipefail

# Require master
branch=$(git rev-parse --abbrev-ref HEAD)
if [[ "$branch" != "master" ]]; then
  echo "Refusing to release from branch '$branch' (must be 'master')." >&2
  exit 1
fi

VERSION=$(grep -Po '^__version__\s*=\s*"\K[^"]+' python/magnetron/__init__.py)
TAG="v$VERSION"

echo "Re-tagging $TAG at $(git rev-parse --short HEAD)"

git tag -fa "$TAG" -m "Release $TAG (retry)"
git push origin "$TAG" --force
