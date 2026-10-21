#!/usr/bin/env bash
# HyperLink's dataset parts and code sessions, by hand (0.72.8a3).
#
#   T1=http://127.0.0.1:8000 T1_API_KEY=... ./examples/t1api/hyperlink_datasets_and_code.sh data.jsonl
#
# Uploads a dataset, previews rows 0-5, makes a /code sandbox with the file
# copied in and a git repository, then sends one chat message that carries
# both: the model gets the excerpt, and tools for the sandbox. Needs a key
# with `write` (or admin) and a server with T1_CODE_SANDBOX=1. Needs curl
# and jq.
set -euo pipefail

T1="${T1:-http://127.0.0.1:8000}"
FILE="${1:?usage: $0 <dataset file: .jsonl .parquet .csv .txt .md ...>}"
AUTH=(-H "Authorization: Bearer ${T1_API_KEY:?set T1_API_KEY}")
JSON=(-H "Content-Type: application/json")

file_id=$(curl -fsS "${AUTH[@]}" -F "file=@$FILE" "$T1/hyperlink/files" | jq -r .file.file_id)
echo "uploaded $FILE as $file_id"

pick=$(jq -n --arg id "$file_id" '{file_id: $id, rows: "0-5", style: "table"}')
echo "--- what the model will see ---"
curl -fsS "${AUTH[@]}" "${JSON[@]}" -d "$pick" "$T1/hyperlink/datasets/preview" | jq -r .text

box=$(curl -fsS "${AUTH[@]}" "${JSON[@]}" -d '{"name": "example"}' "$T1/code/create/sandbox" | jq -r .id)
curl -fsS "${AUTH[@]}" "${JSON[@]}" -d "{\"file_id\": \"$file_id\"}" "$T1/code/sandbox/$box/import" >/dev/null
for op in '{"op":"init"}' '{"op":"add","paths":["."]}' '{"op":"commit","message":"the dataset"}' '{"op":"log"}'; do
  curl -fsS "${AUTH[@]}" "${JSON[@]}" -d "$op" "$T1/code/sandbox/$box/git" | jq -r '.output // .error'
done

chat=$(curl -fsS "${AUTH[@]}" "${JSON[@]}" -d '{"title": "dataset example"}' "$T1/hyperlink/sessions" | jq -r .session.session_id)
body=$(jq -n --arg id "$file_id" --arg box "$box" --argjson pick "$pick" \
  '{content: "Summarise these rows, then write summary.md in the sandbox and commit it.",
    attachment_ids: [$id], datasets: [$pick], code_sandbox_id: $box}')
# Not -f: when no model is serving, the reason is in the body (a 503).
curl -sS "${AUTH[@]}" "${JSON[@]}" -d "$body" "$T1/hyperlink/sessions/$chat/chat" \
  | jq -r '.assistant_message.content // .error.message // .'
echo "sandbox $box: $T1/code/sandbox/$box"
