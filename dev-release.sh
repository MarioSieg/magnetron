VERSION=$(grep -Po '^__version__\s*=\s*"\K[^"]+' python/magnetron/__init__.py)
TAG="v$VERSION"

echo "Tagging release $TAG"

git tag -a "$TAG" -m "Release $TAG"
git push origin "$TAG"

if command -v gh >/dev/null; then
  gh release create "$TAG" --notes "Release $TAG"
else
  echo "gh CLI not installed, skipping GitHub release."
fi
