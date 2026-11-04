#!/usr/bin/env bash
# Reproducible inputs for the AlmaLinux-compatible, CGO-free Linux release.
# Run from a clean, committed checkout; requires Go >= go.mod, Node, Yarn 1.22.
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
version=${1:?Usage: build-linux-static.sh <version> [output-dir]}
out=${2:-"$root/_output"}
case "$version" in
  *[!a-zA-Z0-9._+-]*|'') printf 'Invalid version\n' >&2; exit 2 ;;
esac
if [[ -n $(git status --porcelain --untracked-files=normal) ]]; then
  printf 'Refusing release build from a dirty checkout\n' >&2
  exit 1
fi
if [[ ${GOSUMDB:-sum.golang.org} == off ]]; then
  printf 'Go module checksum verification must be enabled\n' >&2
  exit 1
fi
commit=$(git rev-parse HEAD)
built=$(git show -s --format=%cI HEAD)
mkdir -p "$out"
out=$(cd "$out" && pwd)
name="done-hub-${version}-linux-amd64"

(
  cd web
  yarn install --frozen-lockfile --non-interactive
  VITE_APP_VERSION="$version" yarn build
)
go mod verify
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -mod=readonly -trimpath \
  -ldflags="-s -w -X done-hub/common/config.Version=$version -X done-hub/common/config.Commit=$commit -X done-hub/common/config.BuildTime=$built" \
  -o "$out/$name" .

if command -v sha256sum >/dev/null 2>&1; then
  (cd "$out" && sha256sum "$name" > "$name.sha256")
else
  (cd "$out" && shasum -a 256 "$name" > "$name.sha256")
fi
{
  printf 'version=%s\ncommit=%s\nbuild_time=%s\n' "$version" "$commit" "$built"
  go version
  printf 'node=%s\nyarn=%s\n' "$(node --version)" "$(yarn --version)"
} > "$out/$name.build-info.txt"
printf 'Built %s\n' "$out/$name"
