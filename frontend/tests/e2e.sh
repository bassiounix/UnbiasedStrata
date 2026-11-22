#!/usr/bin/env bash
# UnbiasedStrata: the front end end to end against the fake engine, with a vocabulary-only GGUF from llama.cpp's
# own test models (no GPU, no weights). Checks the HTTP surface a coding agent touches: /health, /v1/models,
# a non-streaming chat completion, a streaming one, a client that disconnects mid-stream, and an engine error.
#
#   frontend/tests/e2e.sh <unbiased-strata binary> [model.gguf]
set -u
BIN=${1:?usage: e2e.sh <unbiased-strata> [model.gguf]}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
MODEL=${2:-$ROOT/third_party/llama.cpp/models/ggml-vocab-qwen35.gguf}
PORT=${PORT:-18931}
URL=http://127.0.0.1:$PORT
LOG=$(mktemp "${TMPDIR:-/tmp}/us2-e2e-XXXXXX")   # no suffix after the Xs: BSD mktemp rejects it
fail=0
check() { if [ "$1" = 0 ]; then echo "ok   $2"; else echo "FAIL $2"; fail=1; fi; }
# j EXPR: evaluate a Python expression over the JSON on stdin (bound to `d`); the test needs python3 anyway, not jq.
j() { python3 -c 'import json,sys; d=json.load(sys.stdin); r=eval(sys.argv[1]); print(json.dumps(r) if isinstance(r,(dict,list)) else ("true" if r is True else "false" if r is False else "null" if r is None else r))' "$1" 2>/dev/null; }

# tokens the fake engine emits: FAKE_TOKENS so the reply does not depend on the template's special tokens
export FAKE_TOKENS="${FAKE_TOKENS:-9707,11,1879,0}"   # "Hello, world!" in the Qwen vocabulary
export FAKE_DELAY_MS=${FAKE_DELAY_MS:-0}

"$BIN" --engine "$HERE/fake-engine.py" -m "$MODEL" --host 127.0.0.1 --port "$PORT" --alias fake -- --max-context 4096 > "$LOG" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null; wait $PID 2>/dev/null' EXIT

for i in $(seq 1 100); do
    if curl -s -o /dev/null -w '%{http_code}' "$URL/health" 2>/dev/null | grep -q 200; then break; fi
    if ! kill -0 $PID 2>/dev/null; then echo "FAIL server exited early"; cat "$LOG"; exit 1; fi
    sleep 0.2
done

# /health and /v1/models
h=$(curl -s "$URL/health"); check $([ "$(echo "$h" | j 'd["status"]')" = ok ] && echo 0 || echo 1) "/health: $h"
# upstream PolyStrata clients (tools/strata_bench.py) require these two to treat the server as a loaded strata server
check $([ "$(echo "$h" | j 'd["service"]')" = strata ] && [ "$(echo "$h" | j 'd["loaded"]')" = true ] && echo 0 || echo 1) "/health: service=strata loaded=true"
m=$(curl -s "$URL/v1/models"); check $([ "$(echo "$m" | j 'd["data"][0]["id"]')" = fake ] && echo 0 || echo 1) "/v1/models id: $(echo "$m" | j '{k: d["data"][0].get(k) for k in ("id", "context_length")}')"

# non-streaming chat
r=$(curl -s "$URL/v1/chat/completions" -H 'content-type: application/json' -d '{"model":"fake","messages":[{"role":"user","content":"Say hello"}],"max_tokens":32,"temperature":0}')
content=$(echo "$r" | j 'd["choices"][0]["message"]["content"]')
check $([ "$content" = "Hello, world!" ] && echo 0 || echo 1) "chat content: $(echo "$r" | j '{"content": d["choices"][0]["message"]["content"], "finish": d["choices"][0]["finish_reason"], "usage": d["usage"]}')"
check $([ "$(echo "$r" | j 'd["choices"][0]["finish_reason"]')" = stop ] && echo 0 || echo 1) "finish_reason stop"
check $([ "$(echo "$r" | j 'd["usage"]["completion_tokens"]')" = 4 ] && echo 0 || echo 1) "usage.completion_tokens 4"

# streaming chat: role chunk, content deltas, finish chunk, [DONE]
s=$(curl -s -N "$URL/v1/chat/completions" -H 'content-type: application/json' -d '{"model":"fake","stream":true,"messages":[{"role":"user","content":"Say hello"}],"max_tokens":32}')
n_data=$(echo "$s" | grep -c '^data: {')
# the last chunk carries usage (and timings) with an empty choices array, as upstream PolyStrata always does
joined=$(echo "$s" | grep '^data: {' | sed 's/^data: //' | python3 -c 'import json,sys; print("".join((c[0]["delta"].get("content") or "") for c in (json.loads(l)["choices"] for l in sys.stdin if l.strip()) if c), end="")')
check $([ "$joined" = "Hello, world!" ] && echo 0 || echo 1) "stream deltas join to the content ($n_data chunks)"
check $(echo "$s" | grep -q '^data: \[DONE\]' && echo 0 || echo 1) "stream ends with [DONE]"
finish=$(echo "$s" | grep '^data: {' | sed 's/^data: //' | python3 -c 'import json,sys; print([c[0]["finish_reason"] for c in (json.loads(l)["choices"] for l in sys.stdin if l.strip()) if c and c[0].get("finish_reason")][-1])')
check $([ "$finish" = stop ] && echo 0 || echo 1) "stream finish_reason stop"
last=$(echo "$s" | grep '^data: {' | tail -1 | sed 's/^data: //')
check $([ "$(echo "$last" | j 'd["usage"]["completion_tokens"]')" = 4 ] && [ "$(echo "$last" | j 'len(d["choices"])')" = 0 ] && echo 0 || echo 1) "stream last chunk: usage.completion_tokens 4, empty choices"

# a client that disconnects mid-stream: the engine is told to stop and the next request is clean. The fake engine
# reads its env once at start, so this needs a second server whose engine streams a long reply slowly (token 9707,
# "Hello": a raw byte token would not survive the chat parser on the follow-up request).
SLOW_PORT=$((PORT + 1)); SLOW_URL=http://127.0.0.1:$SLOW_PORT; SLOW_LOG=$(mktemp "${TMPDIR:-/tmp}/us2-e2e-slow-XXXXXX")
FAKE_N_TOKENS=100000 FAKE_TOKEN_ID=9707 FAKE_DELAY_MS=10 "$BIN" --engine "$HERE/fake-engine.py" -m "$MODEL" --host 127.0.0.1 --port "$SLOW_PORT" --alias fake -- --max-context 4096 > "$SLOW_LOG" 2>&1 &
SLOW_PID=$!
trap 'kill $PID $SLOW_PID 2>/dev/null; wait $PID $SLOW_PID 2>/dev/null; rm -f "$LOG" "$SLOW_LOG" "$BIG"' EXIT
for i in $(seq 1 100); do
    if curl -s -o /dev/null -w '%{http_code}' "$SLOW_URL/health" 2>/dev/null | grep -q 200; then break; fi
    if ! kill -0 $SLOW_PID 2>/dev/null; then echo "FAIL slow server exited early"; cat "$SLOW_LOG"; exit 1; fi
    sleep 0.2
done
curl -s -N --max-time 1 "$SLOW_URL/v1/chat/completions" -H 'content-type: application/json' -d '{"model":"fake","stream":true,"messages":[{"role":"user","content":"x"}],"max_tokens":4000}' > /dev/null 2>&1
sleep 0.5
r2=$(curl -s --max-time 10 "$SLOW_URL/v1/chat/completions" -H 'content-type: application/json' -d '{"model":"fake","messages":[{"role":"user","content":"again"}],"max_tokens":8}')
check $([ "$(echo "$r2" | j 'd["usage"]["completion_tokens"]')" = 8 ] && [ "$(echo "$r2" | j 'd["choices"][0]["finish_reason"]')" = length ] && echo 0 || echo 1) "request after a disconnected stream: $(echo "$r2" | j '{"finish": d["choices"][0]["finish_reason"], "usage": d["usage"]["completion_tokens"]}')"
check $(grep -q "cancelled by the client" "$SLOW_LOG" && echo 0 || echo 1) "server logged the cancel"

# a prompt that exceeds the context is refused with 400, not 500
big=$(python3 -c 'print("word " * 5000)')
BIG=$(mktemp "${TMPDIR:-/tmp}/us2-e2e-big-XXXXXX")
code=$(curl -s -o "$BIG" -w '%{http_code}' "$URL/v1/chat/completions" -H 'content-type: application/json' -d "{\"model\":\"fake\",\"messages\":[{\"role\":\"user\",\"content\":\"$big\"}]}")
check $([ "$code" = 400 ] && echo 0 || echo 1) "oversize prompt -> $code: $(j 'd["error"]["type"]' < "$BIG")"

# a picture without --mmproj is refused with llama.cpp's own message, not a crash (with --mmproj the rows go to the
# engine as GENI; that path needs a real projector file and is checked by frontend/tests/images-real-engine.sh)
IMG=$(mktemp "${TMPDIR:-/tmp}/us2-e2e-img-XXXXXX")
img=$(curl -s -o "$IMG" -w '%{http_code}' "$URL/v1/chat/completions" -H 'content-type: application/json' -d '{"model":"fake","messages":[{"role":"user","content":[{"type":"text","text":"what is this"},{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgo="}}]}]}')
# llama.cpp raises this as a runtime error, which its ex_wrapper maps to 500 (llama-server answers the same)
check $([ "$img" = 400 ] || [ "$img" = 500 ] || [ "$img" = 501 ] && grep -qi "image" "$IMG" && curl -s "$URL/health" | grep -q '"status":"ok"' && echo 0 || echo 1) "image without --mmproj -> $img: $(j 'd["error"]["message"]' < "$IMG" | cut -c1-80); server alive"
rm -f "$IMG"

# a control token's text inside a message is text (frontend/literals.hpp, USTRAT-14). The oracle: the rendered prompt
# (/apply-template) tokenized by hand as the server does it, specials parsed except inside the literals that came from
# the messages, must give the request's usage.prompt_tokens; and that count differs from the whole prompt tokenized
# with specials parsed, which is what the literals used to become. Three places: user content (a control token, the
# end-of-text token and <think>, an added token the tokenizer matches regardless), a tool's description, and an
# assistant message in the history whose leading <think> block stays the model's while the literals inside and after
# it are text.
# The vocabulary-only GGUF's template has neither tools nor reasoning, so this runs a third server with upstream's
# Qwen3 template (frontend/tests/data/qwen3-chat-template.jinja: tools through tojson, <think> blocks).
TPL_PORT=$((PORT + 2)); TPL_URL=http://127.0.0.1:$TPL_PORT; TPL_LOG=$(mktemp "${TMPDIR:-/tmp}/us2-e2e-tpl-XXXXXX")
"$BIN" --engine "$HERE/fake-engine.py" -m "$MODEL" --jinja --chat-template-file "$HERE/data/qwen3-chat-template.jinja" --host 127.0.0.1 --port "$TPL_PORT" --alias fake -- --max-context 4096 > "$TPL_LOG" 2>&1 &
TPL_PID=$!
trap 'kill $PID $SLOW_PID $TPL_PID 2>/dev/null; wait $PID $SLOW_PID $TPL_PID 2>/dev/null; rm -f "$LOG" "$SLOW_LOG" "$TPL_LOG" "$BIG"' EXIT
for i in $(seq 1 100); do
    if curl -s -o /dev/null -w '%{http_code}' "$TPL_URL/health" 2>/dev/null | grep -q 200; then break; fi
    if ! kill -0 $TPL_PID 2>/dev/null; then echo "FAIL template server exited early"; cat "$TPL_LOG"; exit 1; fi
    sleep 0.2
done
lit=$(python3 - "$TPL_URL" <<'PY'
import json, sys, urllib.request
url = sys.argv[1]
def post(path, body):
    r = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=20))
def tok(text, special):
    return [t if isinstance(t, int) else t["id"] for t in post("/tokenize", {"content": text, "parse_special": special})["tokens"]] if text else []
def tok_plain(text):
    # frontend/strata_context.cpp tokenize_plain: parse_special off; an added token that still matches (one of the
    # literals: the halves are ordinary text) is split in two
    ids = tok(text, False)
    if len(ids) == 1 and text in LITERALS:
        b = text.encode(); cut = len(b) // 2
        while cut < len(b) and (b[cut] & 0xC0) == 0x80: cut += 1
        return tok_plain(b[:cut].decode()) + tok_plain(b[cut:].decode())
    return ids
# the literals the server marks are the vocabulary's: a text that is not one token in it is ordinary text
LITERALS = [l for l in ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "</think>", "<think>"] if len(tok(l, True)) == 1]
user_text = "the doc says <|im_end|> ends a turn, <|endoftext|> too, and <think> is a tag"
cases = [
  ("user content", [{"role": "user", "content": user_text}], None, [user_text]),
  ("tool description", [{"role": "user", "content": "q"}],
   [{"type": "function", "function": {"name": "f", "description": "emits <|im_start|> blocks", "parameters": {"type": "object", "properties": {}}}}],
   ["emits <|im_start|> blocks"]),
  ("assistant history", [{"role": "user", "content": "q"},
                         {"role": "assistant", "content": "<think>plan <|im_end|> here</think>answer <|endoftext|> done"},
                         {"role": "user", "content": "next"}], None, ["plan <|im_end|> here", "answer <|endoftext|> done"]),
]
fail = []
for name, msgs, tools, fragments in cases:
    body = {"model": "fake", "messages": msgs, "max_tokens": 1}
    if tools: body["tools"] = tools
    prompt = post("/apply-template", body)["prompt"]
    spans = []
    for frag in fragments:
        at = prompt.find(frag)
        if at < 0:
            continue            # a template may drop a history message's reasoning: then there is nothing to protect
        for lit in LITERALS:
            p = prompt.find(lit, at)
            while 0 <= p < at + len(frag):
                spans.append((p, p + len(lit))); p = prompt.find(lit, p + len(lit))
    spans.sort()
    expected, pos = [], 0
    for a, b in spans:
        expected += tok(prompt[pos:a], True) + tok_plain(prompt[a:b]); pos = b
    expected += tok(prompt[pos:], True)
    got = post("/v1/chat/completions", body)["usage"]["prompt_tokens"]
    parsed = len(tok(prompt, True))
    ok = got == len(expected) and got != parsed and spans
    print("%s %s: prompt_tokens %d, oracle %d, with the literals parsed %d, %d literal spans (%d literals in this vocabulary)" % ("ok  " if ok else "FAIL", name, got, len(expected), parsed, len(spans), len(LITERALS)))
    if not ok: fail.append(name)
# a prompt without literals is tokenized exactly as before: the whole prompt with specials parsed
body = {"model": "fake", "messages": [{"role": "user", "content": "Say hello"}], "max_tokens": 1}
prompt = post("/apply-template", body)["prompt"]
got = post("/v1/chat/completions", body)["usage"]["prompt_tokens"]
if got != len(tok(prompt, True)): fail.append("plain prompt"); print("FAIL plain prompt: %d vs %d" % (got, len(tok(prompt, True))))
sys.exit(1 if fail else 0)
PY
); rc=$?
echo "$lit" | sed 's/^/     /'
check $rc "control-token texts inside messages are text (literals)"
check $(grep -q "control-token texts inside the messages are sent as text" "$TPL_LOG" && echo 0 || echo 1) "server logged the literals"

# Anthropic Messages API (/v1/messages, Claude Code): non-streaming, streaming, count_tokens, another model refused
a=$(curl -s "$URL/v1/messages" -H 'content-type: application/json' -H 'anthropic-version: 2023-06-01' -d '{"model":"fake","max_tokens":32,"system":"Be brief.","messages":[{"role":"user","content":"Say hello"}],"output_config":{"effort":"high"}}')
check $([ "$(echo "$a" | j 'd["type"]')" = message ] && [ "$(echo "$a" | j '"".join(c.get("text","") for c in d["content"] if c["type"]=="text")')" = "Hello, world!" ] && [ "$(echo "$a" | j 'd["stop_reason"]')" = end_turn ] && echo 0 || echo 1) "anthropic message: $(echo "$a" | j '{"type": d.get("type"), "stop": d.get("stop_reason"), "usage": d.get("usage")}')"
as=$(curl -s -N "$URL/v1/messages" -H 'content-type: application/json' -d '{"model":"fake","max_tokens":32,"stream":true,"messages":[{"role":"user","content":"Say hello"}]}')
ajoined=$(echo "$as" | grep '^data: {' | sed 's/^data: //' | python3 -c 'import json,sys; print("".join(e["delta"].get("text","") for e in map(json.loads, sys.stdin) if e.get("type")=="content_block_delta"), end="")')
check $([ "$ajoined" = "Hello, world!" ] && echo "$as" | grep -q '^event: message_stop' && echo 0 || echo 1) "anthropic stream: text deltas join to the content, ends with message_stop"
ac=$(curl -s "$URL/v1/messages/count_tokens" -H 'content-type: application/json' -d '{"model":"fake","messages":[{"role":"user","content":"Say hello"}]}')
check $([ "$(echo "$ac" | j 'd["input_tokens"] > 0')" = true ] && echo 0 || echo 1) "anthropic count_tokens: $ac"
aw=$(curl -s -o /dev/null -w '%{http_code}' "$URL/v1/messages" -H 'content-type: application/json' -d '{"model":"claude-opus-5-5","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}')
check $([ "$aw" = 404 ] && echo 0 || echo 1) "anthropic request for another model -> $aw"
# Claude Code's shape: system blocks, then a system message after the first user message, which a template that
# allows one system message, first (the Qwen3 one), would refuse
am=$(curl -s "$TPL_URL/v1/messages" -H 'content-type: application/json' -d '{"model":"fake","max_tokens":8,"system":[{"type":"text","text":"You are Claude Code."},{"type":"text","text":"Be brief."}],"messages":[{"role":"user","content":[{"type":"text","text":"<system-reminder>x</system-reminder>"},{"type":"text","text":"hi"}]},{"role":"system","content":[{"type":"text","text":"# Environment\ncwd: /tmp"}]}],"thinking":{"type":"adaptive"},"output_config":{"effort":"low"}}')
check $([ "$(echo "$am" | j 'd["type"]')" = message ] && echo 0 || echo 1) "anthropic system message mid-conversation: $(echo "$am" | j 'd.get("type") or d.get("error",{}).get("message","")[:80]')"

# --parallel: several requests in flight take turns on the one engine (--slice-ms of decoding each), all streaming
# meanwhile; a paused answer goes on from where it stopped (the fake engine's FAKE_RESUME continues a prompt that ends
# with the start of its answer, as the real one continues its live or parked conversation). A fifth server: 3 in
# flight, 150 ms turns, 5 ms per token.
PAR_PORT=$((PORT + 3)); PAR_URL=http://127.0.0.1:$PAR_PORT; PAR_LOG=$(mktemp "${TMPDIR:-/tmp}/us2-e2e-par-XXXXXX")
PAR_GENS=$(mktemp "${TMPDIR:-/tmp}/us2-e2e-pargen-XXXXXX")
FAKE_N_TOKENS=100000 FAKE_TOKEN_ID=9707 FAKE_DELAY_MS=5 FAKE_RESUME=1 FAKE_LOG="$PAR_GENS" "$BIN" --engine "$HERE/fake-engine.py" -m "$MODEL" --host 127.0.0.1 --port "$PAR_PORT" --alias fake --parallel 3 --slice-ms 150 -- --max-context 4096 > "$PAR_LOG" 2>&1 &
PAR_PID=$!
trap 'kill $PID $SLOW_PID $TPL_PID $PAR_PID 2>/dev/null; wait $PID $SLOW_PID $TPL_PID $PAR_PID 2>/dev/null; rm -f "$LOG" "$SLOW_LOG" "$TPL_LOG" "$PAR_LOG" "$PAR_GENS" "$BIG"' EXIT
for i in $(seq 1 100); do
    if curl -s -o /dev/null -w '%{http_code}' "$PAR_URL/health" 2>/dev/null | grep -q 200; then break; fi
    if ! kill -0 $PAR_PID 2>/dev/null; then echo "FAIL parallel server exited early"; cat "$PAR_LOG"; exit 1; fi
    sleep 0.2
done
check $([ "$(curl -s "$PAR_URL/props" | j 'd["total_slots"]')" = 3 ] && echo 0 || echo 1) "parallel: /props total_slots 3"
par=$(python3 - "$PAR_URL" <<'PY'
import json, sys, threading, time, urllib.request
url = sys.argv[1] + "/v1/chat/completions"
def body(text, n, stream):
    return json.dumps({"model": "fake", "stream": stream, "max_tokens": n,
                       "messages": [{"role": "user", "content": text}]}).encode()
def run(i, n, out, stream=True, close_after=0):
    r = urllib.request.urlopen(urllib.request.Request(url, data=body("chat %d" % i, n, stream),
                                                      headers={"content-type": "application/json"}), timeout=60)
    t0 = time.time(); first = None; text = ""; usage = None; finish = None; chunks = 0
    if not stream:
        d = json.load(r)
        out[i] = {"first": time.time() - t0, "end": time.time() - t0, "text": d["choices"][0]["message"]["content"],
                  "usage": d["usage"]["completion_tokens"], "finish": d["choices"][0]["finish_reason"], "t0": t0}
        return
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data: {"):
            continue
        d = json.loads(line[6:])
        if d.get("usage"):
            usage = d["usage"]["completion_tokens"]
        for c in d.get("choices", []):
            delta = c["delta"].get("content") or ""
            if delta and first is None:
                first = time.time()
            text += delta
            chunks += 1 if delta else 0
            finish = c.get("finish_reason") or finish
        if close_after and chunks >= close_after:
            r.close(); out[i] = {"closed": True}; return
    out[i] = {"first": first, "end": time.time(), "text": text, "usage": usage, "finish": finish, "t0": t0}
fail = []
def report(ok, what):
    print(("ok   " if ok else "FAIL ") + what)
    if not ok: fail.append(what)
# three streams at once: each whole and in order, and all of them streaming before any ends
out = {}
ts = [threading.Thread(target=run, args=(i, 120, out)) for i in range(3)]
for t in ts: t.start()
for t in ts: t.join()
whole = all(out[i]["text"] == "Hello" * 120 and out[i]["usage"] == 120 and out[i]["finish"] == "length" for i in range(3))
report(whole, "3 streams at once: each 120 tokens, whole (%s)" % [(len(out[i]["text"]) // 5, out[i]["usage"], out[i]["finish"]) for i in range(3)])
first_last = max(out[i]["first"] for i in range(3)); end_first = min(out[i]["end"] for i in range(3))
report(first_last < end_first, "3 streams at once: every one streamed before the first ended (last first token %.2f s before the first end)" % (end_first - first_last))
# four at once with --parallel 3: the fourth waits in the queue, and is answered (non-streaming here)
out = {}
ts = [threading.Thread(target=run, args=(i, 60, out, i % 2 == 0)) for i in range(4)]
for t in ts: t.start(); time.sleep(0.02)
for t in ts: t.join()
report(all(out[i]["text"] == "Hello" * 60 and out[i]["usage"] == 60 for i in range(4)), "4 requests with --parallel 3: all answered whole")
# a client that leaves while the requests take turns: the others go on, whole
out = {}
ts = [threading.Thread(target=run, args=(0, 150, out)), threading.Thread(target=run, args=(1, 2000, out, True, 20)),
      threading.Thread(target=run, args=(2, 150, out))]
for t in ts: t.start(); time.sleep(0.02)
for t in ts: t.join()
report(out[1].get("closed") and all(out[i]["text"] == "Hello" * 150 for i in (0, 2)), "a client that leaves mid-turn: the other two whole")
sys.exit(1 if fail else 0)
PY
); rc=$?
echo "$par" | sed 's/^/     /'
check $rc "parallel requests take turns"
check $(grep -q "paused after" "$PAR_LOG" && grep -q "resumed (turn" "$PAR_LOG" && echo 0 || echo 1) "parallel: server logged pauses and resumes"
check $(awk '$4 > 0' "$PAR_GENS" | grep -q . && echo 0 || echo 1) "parallel: a later turn continued the answer ($(awk '$4 > 0' "$PAR_GENS" | wc -l) continuing GENs of $(wc -l < "$PAR_GENS"))"
check $(! grep -q "read .* tokens again\|prompt size mismatch" "$PAR_LOG" && echo 0 || echo 1) "parallel: no turn read its conversation again, no size mismatch"
# a request waiting for its turn while another streams: its stream gets SSE pings, and a client that leaves is seen
# while it waits (llama.cpp's reader would not time out while other results arrive; the front end wakes it). Its own
# server: long turns, a ping every second.
WAIT_PORT=$((PORT + 4)); WAIT_LOG=$(mktemp "${TMPDIR:-/tmp}/us2-e2e-wait-XXXXXX")
FAKE_N_TOKENS=100000 FAKE_TOKEN_ID=9707 FAKE_DELAY_MS=5 FAKE_RESUME=1 "$BIN" --engine "$HERE/fake-engine.py" -m "$MODEL" --host 127.0.0.1 --port "$WAIT_PORT" --alias fake --parallel 3 --slice-ms 5000 --sse-ping-interval 1 -- --max-context 4096 > "$WAIT_LOG" 2>&1 &
WAIT_PID=$!
trap 'kill $PID $SLOW_PID $TPL_PID $PAR_PID $WAIT_PID 2>/dev/null; wait $PID $SLOW_PID $TPL_PID $PAR_PID $WAIT_PID 2>/dev/null; rm -f "$LOG" "$SLOW_LOG" "$TPL_LOG" "$PAR_LOG" "$PAR_GENS" "$WAIT_LOG" "$BIG"' EXIT
for i in $(seq 1 100); do
    if curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$WAIT_PORT/health" 2>/dev/null | grep -q 200; then break; fi
    if ! kill -0 $WAIT_PID 2>/dev/null; then echo "FAIL waiting server exited early"; cat "$WAIT_LOG"; exit 1; fi
    sleep 0.2
done
curl -s -N "http://127.0.0.1:$WAIT_PORT/v1/chat/completions" -H 'content-type: application/json' -d '{"model":"fake","stream":true,"max_tokens":1200,"messages":[{"role":"user","content":"a"}]}' > /dev/null &
RUN_CURL=$!
sleep 0.2
pings=$(python3 - "$WAIT_PORT" <<'PY'
import json, socket, sys, time
port = int(sys.argv[1])
def open_req(stream):
    b = json.dumps({"model": "fake", "stream": stream, "max_tokens": 2000, "messages": [{"role": "user", "content": "b"}]})
    s = socket.create_connection(("127.0.0.1", port))
    s.sendall(("POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n%s" % (len(b), b)).encode())
    return s
s = open_req(True)
s.settimeout(0.2)
got, t0 = b"", time.time()
while time.time() - t0 < 2.6:
    try:
        got += s.recv(4096)
    except socket.timeout:
        pass
s.close()
open_req(False).close()
print(got.count(b"\r\n:\n\n"))   # a ping is a chunk of its own: ":\n\n" right after its size line
PY
)
sleep 1.5
check $([ "${pings:-0}" -ge 1 ] && echo 0 || echo 1) "parallel: a stream waiting for its turn gets SSE pings ($pings in 2.6 s)"
check $([ "$(grep -c "cancelled by the client while waiting" "$WAIT_LOG")" = 2 ] && [ "$(curl -s "http://127.0.0.1:$WAIT_PORT/health" | j 'd["requests"]')" = 1 ] && echo 0 || echo 1) "parallel: clients that leave while waiting (stream, non-stream) are cancelled before their turn"
wait $RUN_CURL

# an engine that could not park a paused answer (FAKE_FORGET: the turn reads it again): the turns stop, and the
# requests in flight run one after another, whole
FORGET_PORT=$((PORT + 5)); FORGET_LOG=$(mktemp "${TMPDIR:-/tmp}/us2-e2e-forget-XXXXXX")
FAKE_N_TOKENS=100000 FAKE_TOKEN_ID=9707 FAKE_DELAY_MS=5 FAKE_RESUME=1 FAKE_FORGET=1 "$BIN" --engine "$HERE/fake-engine.py" -m "$MODEL" --host 127.0.0.1 --port "$FORGET_PORT" --alias fake --parallel 2 --slice-ms 150 -- --max-context 4096 > "$FORGET_LOG" 2>&1 &
FORGET_PID=$!
trap 'kill $PID $SLOW_PID $TPL_PID $PAR_PID $WAIT_PID $FORGET_PID 2>/dev/null; wait $PID $SLOW_PID $TPL_PID $PAR_PID $WAIT_PID $FORGET_PID 2>/dev/null; rm -f "$LOG" "$SLOW_LOG" "$TPL_LOG" "$PAR_LOG" "$PAR_GENS" "$WAIT_LOG" "$FORGET_LOG" "$BIG"' EXIT
for i in $(seq 1 100); do
    if curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$FORGET_PORT/health" 2>/dev/null | grep -q 200; then break; fi
    if ! kill -0 $FORGET_PID 2>/dev/null; then echo "FAIL forget server exited early"; cat "$FORGET_LOG"; exit 1; fi
    sleep 0.2
done
for k in 1 2; do
    curl -s "http://127.0.0.1:$FORGET_PORT/v1/chat/completions" -H 'content-type: application/json' -d "{\"model\":\"fake\",\"max_tokens\":200,\"messages\":[{\"role\":\"user\",\"content\":\"c$k\"}]}" > "$FORGET_LOG.$k" &
    sleep 0.05
done
wait %?FORGET_PORT 2>/dev/null; sleep 0.2
while [ "$(curl -s "http://127.0.0.1:$FORGET_PORT/health" | j 'd["requests"]')" != 0 ]; do sleep 0.2; done
fw=$(for k in 1 2; do j 'd["usage"]["completion_tokens"]' < "$FORGET_LOG.$k"; done | paste -sd,); rm -f "$FORGET_LOG.1" "$FORGET_LOG.2"
check $([ "$fw" = "200,200" ] && [ "$(grep -c "paused after" "$FORGET_LOG")" = 2 ] && grep -q "run one after another" "$FORGET_LOG" && echo 0 || echo 1) "parallel: an engine that cannot park stops the turns ($(grep -c "paused after" "$FORGET_LOG") pauses, answers $fw tokens)"

pr=$(curl -s "$PAR_URL/v1/chat/completions" -H 'content-type: application/json' -d '{"model":"fake","messages":[{"role":"user","content":"after"}],"max_tokens":8}')
check $([ "$(echo "$pr" | j 'd["usage"]["completion_tokens"]')" = 8 ] && [ "$(curl -s "$PAR_URL/health" | j 'd["requests"]')" = 0 ] && echo 0 || echo 1) "parallel: a request afterwards is clean, none left in flight"

# tokenize / detokenize round trip
t=$(curl -s "$URL/tokenize" -H 'content-type: application/json' -d '{"content":"Hello, world!"}' | j 'd["tokens"]')
d=$(curl -s "$URL/detokenize" -H 'content-type: application/json' -d "{\"tokens\":$t}" | j 'd["content"]')
check $([ "$d" = "Hello, world!" ] && echo 0 || echo 1) "tokenize/detokenize round trip: $t"

if [ $fail = 0 ]; then echo "e2e: all checks passed"; else echo "--- server log"; tail -40 "$LOG"; echo "--- slow server log"; tail -20 "$SLOW_LOG"; echo "--- template server log"; tail -20 "$TPL_LOG"; echo "--- parallel server log"; tail -30 "$PAR_LOG"; fi
exit $fail
