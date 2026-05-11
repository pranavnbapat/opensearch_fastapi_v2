#!/bin/sh
set -eu

# Build and push the OpenSearch FastAPI image to GHCR.
#
# Usage:
#   ./build_and_push.sh          # tag = inferred from docker-compose.yml, else latest
#   ./build_and_push.sh v3       # tag = v3
#
# Optional environment overrides:
#   REGISTRY=ghcr.io/pranavnbapat
#   IMAGE_NAME=ghcr.io/pranavnbapat/opensearch_fastapi
#   PLATFORM=linux/amd64
#   DOCKERFILE=Dockerfile
#   CONTEXT=.
#
# Prerequisites (one-time):
#   docker login ghcr.io -u <github-user> -p <PAT-with-write:packages>
#
# Notes:
# - The image is built for linux/amd64 explicitly so it runs on common x86_64 servers
#   even when the local machine is not x86_64.
# - This script builds directly from Dockerfile instead of relying on docker compose
#   image tags, so the pushed tag is always explicit.

REGISTRY="${REGISTRY:-ghcr.io/pranavnbapat}"
IMAGE_NAME="${IMAGE_NAME:-${REGISTRY}/opensearch_fastapi}"
PLATFORM="${PLATFORM:-linux/amd64}"
DOCKERFILE="${DOCKERFILE:-Dockerfile}"
CONTEXT="${CONTEXT:-.}"

infer_default_tag() {
    compose_file="${CONTEXT%/}/docker-compose.yml"
    if [ -f "${compose_file}" ]; then
        image_ref="$(awk '/^[[:space:]]*image:[[:space:]]*/ {print $2; exit}' "${compose_file}")"
        case "${image_ref}" in
            *:*)
                echo "${image_ref##*:}"
                return 0
                ;;
        esac
    fi
    echo "latest"
}

TAG="${1:-$(infer_default_tag)}"

FULL_IMAGE="${IMAGE_NAME}:${TAG}"

echo "Building image: ${FULL_IMAGE} (${PLATFORM})"
docker build --platform "${PLATFORM}" -f "${DOCKERFILE}" -t "${FULL_IMAGE}" "${CONTEXT}"

echo "Pushing image: ${FULL_IMAGE}"
docker push "${FULL_IMAGE}"

echo "Done."
echo "Image: ${FULL_IMAGE}"
