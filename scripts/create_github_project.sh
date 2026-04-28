#!/usr/bin/env bash
set -euo pipefail

OWNER="${1:-amargandhi}"
REPO="${2:-openai-privacy-filter-coreml}"
PROJECT_TITLE="${3:-OpenAI Privacy Filter Core ML}"

if ! gh auth status >/dev/null 2>&1; then
  echo "gh is not authenticated. Run: gh auth login -h github.com" >&2
  exit 1
fi

PROJECT_URL="$(gh project create --owner "$OWNER" --title "$PROJECT_TITLE" --format json | python -c 'import json,sys; print(json.load(sys.stdin)["url"])')"
echo "Created project: $PROJECT_URL"

gh repo edit "$OWNER/$REPO" --enable-issues=true

create_issue() {
  local title="$1"
  local body="$2"
  local url
  url="$(gh issue create --repo "$OWNER/$REPO" --title "$title" --body "$body")"
  echo "Created issue: $url"
  gh project item-add "$PROJECT_URL" --url "$url" >/dev/null
}

create_issue "Create Python 3.12 conversion environment" "Acceptance: scripts/check_environment.py passes with conversion dependencies."
create_issue "Convert fixed 128-token Core ML package" "Acceptance: build/OpenAIPrivacyFilterLogits_128.mlpackage and provenance sidecar are generated."
create_issue "Compare PyTorch vs Core ML logits" "Acceptance: parity JSON reports max/mean absolute differences and argmax agreement for all fixtures."
create_issue "Run MLX MXFP8 fixture inference" "Acceptance: reports/mlx-mxfp8.json contains spans and expected/actual category comparison."
create_issue "Run MLX BF16 fixture inference" "Acceptance: reports/mlx-bf16.json exists and can be compared to MXFP8."
create_issue "Add official CLI or Transformers fixture baseline" "Acceptance: baseline output is stored in the same JSON schema."
create_issue "Port tokenizer and offset handling to Swift" "Acceptance: token IDs and offsets match Python tokenizer fixtures."
create_issue "Port BIOES/Viterbi decode to Swift" "Acceptance: Swift decoder matches Python decoder on fixture logits."
create_issue "Implement Manifold CoreMLPrivacyBackend" "Acceptance: backend conforms to PrivacyBackend and returns PrivacyScanResult."
create_issue "Benchmark Core ML compute units" "Acceptance: cold load, warm latency, and memory are reported for Core ML compute-unit choices."
create_issue "Add install and provenance lifecycle" "Acceptance: Manifold records model revision, conversion hash, and installed artifact state."
create_issue "Document production limitations" "Acceptance: README documents chunking, long-context limits, fallback behavior, and human-review caveats."

echo "Project populated."
