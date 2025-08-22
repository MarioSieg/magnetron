VERSION=$(grep -Po '^__version__\s*=\s*"\K[^"]+' python/magnetron/__init__.py)
TAG="v$VERSION"

echo "Re-tagging $TAG at $(git rev-parse --short HEAD)"

git tag -fa "$TAG" -m "Release $TAG (retry)"
git push origin "$TAG" --force
