#!/usr/bin/env bash

# Require master
branch=$(git rev-parse --abbrev-ref HEAD)
if [[ "$branch" != "master" ]]; then
  echo "Refusing to release from branch '$branch' (must be 'master')." >&2
  exit 1
fi

VERSION=$(grep -Po '^__version__\s*=\s*"\K[^"]+' python/magnetron/__init__.py)
TAG="v$VERSION"
gh release create "$TAG" --generate-notes --draft