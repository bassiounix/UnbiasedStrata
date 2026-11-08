#!/usr/bin/env python3
"""Prefill (prompt reading) and decode benchmark for a running strata-agent (or any unbiased-strata server).

Start the server first (`strata-agent`), wait for "listening on", then:

    python3 bench.py agent            # an agent session: cold ~25K-token system prompt, follow-ups, tool-output turns
    python3 bench.py context          # fresh prompts of 1K..64K tokens, 256 generated tokens each
    python3 bench.py context --lengths 2048 8192

Every number comes from the server's own `timings` in each response: `prompt_n` tokens read fresh at
`prompt_per_second`, `cache_n` tokens reused from the conversation cache, `predicted_n` tokens generated at
`predicted_per_second`.  A random tag at the start of each fresh prompt makes it cold (nothing to reuse), so the server
does not need a restart between runs.  Standard library only.
"""
import argparse
import glob
import json
import os
import secrets
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CHARS_PER_TOKEN = 3.2   # this model's tokenizer on the repository's Markdown docs (82,205 chars -> 25,740 tokens)


def load_corpus(path):
    files = sorted(glob.glob(os.path.join(path, '*.md')))
    if not files:
        sys.exit(f'bench: no *.md files in {path} (use --corpus DIR)')
    return ''.join('\n\n' + open(f, encoding='utf-8', errors='ignore').read() for f in files)


def ask(url, messages, max_tokens, timeout):
    body = json.dumps({'model': 'swift-iq3_xxs', 'messages': messages, 'max_tokens': max_tokens,
                       'temperature': 0, 'top_k': 1, 'chat_template_kwargs': {'enable_thinking': False}}).encode()
    req = urllib.request.Request(url.rstrip('/') + '/v1/chat/completions', body, {'Content-Type': 'application/json'})
    t0 = time.time()
    try:
        d = json.load(urllib.request.urlopen(req, timeout=timeout))
    except urllib.error.HTTPError as e:
        return None, f'HTTP {e.code}: {e.read()[:200]!r}'
    except Exception as e:  # noqa: BLE001 - reported in the table
        return None, str(e)[:200]
    t = d.get('timings', {})
    row = {'read': t.get('prompt_n'), 'reused': t.get('cache_n'), 'read_s': round(t.get('prompt_ms', 0) / 1000, 1),
           'prefill_tps': round(t.get('prompt_per_second', 0), 1), 'gen': t.get('predicted_n'),
           'decode_tps': round(t.get('predicted_per_second', 0), 1), 'wall_s': round(time.time() - t0, 1)}
    return (row, d['choices'][0]['message'].get('content') or ''), None


def table(rows):
    out = ['| Request | Tokens read fresh | Reused from cache | Read time | Prompt reading | Generated | Decode |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for label, r, err in rows:
        if err:
            out.append(f'| {label} | failed: {err} | | | | | |')
        else:
            out.append(f"| {label} | {r['read']:,} | {r['reused']:,} | {r['read_s']} s | {r['prefill_tps']} tok/s | "
                       f"{r['gen']} | {r['decode_tps']} tok/s |")
    return '\n'.join(out)


def run_agent(a, corpus):
    tag = secrets.token_hex(6)
    system = (f'[session {tag}] You are a coding agent. The following documentation and skill definitions are your '
              'reference.\n\n' + corpus[: int(a.system_tokens * CHARS_PER_TOKEN)])
    rows = []

    def step(label, msgs, max_tokens=None):
        res, err = ask(a.url, msgs, max_tokens or a.gen, a.timeout)
        rows.append((label, res[0] if res else None, err))
        print(f'  {label}: ' + (err or json.dumps(res[0])), file=sys.stderr, flush=True)
        return res[1] if res else None

    conv = [{'role': 'system', 'content': system},
            {'role': 'user', 'content': 'Using the documentation above, explain in three sentences how KV streaming works.'}]
    answer = step(f'cold: ~{a.system_tokens / 1000:g}K-token system prompt + task', conv)
    if answer is None:
        return rows
    step('next turn, same conversation', conv + [{'role': 'assistant', 'content': answer},
                                                 {'role': 'user', 'content': 'Now name two risks of that design.'}])
    step('new conversation, same system prompt', [conv[0], {'role': 'user', 'content':
         'What does --prefill-head-alias do? Answer in two sentences.'}])
    hist = conv + [{'role': 'assistant', 'content': answer}]
    off = int(a.system_tokens * CHARS_PER_TOKEN)
    for k in range(a.turns):
        tool = corpus[off: off + int(a.turn_tokens * CHARS_PER_TOKEN)]
        off += int(a.turn_tokens * CHARS_PER_TOKEN)
        hist = hist + [{'role': 'user', 'content': f'Tool output #{k + 1}:\n{tool}\n\nSummarize it in two sentences.'}]
        reply = step(f'agent turn {k + 1}: +~{a.turn_tokens / 1000:g}K-token tool output', hist)
        if reply is None:
            break
        hist = hist + [{'role': 'assistant', 'content': reply}]
    return rows


def run_context(a, corpus):
    rows = []
    for n in a.lengths:
        text = corpus[: int(n * CHARS_PER_TOKEN)]
        if len(text) < int(n * CHARS_PER_TOKEN):
            print(f'  (the corpus holds ~{len(corpus) / CHARS_PER_TOKEN:,.0f} tokens; {n} was capped)', file=sys.stderr)
        msgs = [{'role': 'user', 'content': f'[run {secrets.token_hex(6)}]\n\n{text}\n\n'
                 'Write a long, detailed technical summary of the text above.'}]
        res, err = ask(a.url, msgs, a.gen, a.timeout)
        label = f'fresh ~{n // 1024}K-token prompt' if n >= 1024 else f'fresh {n}-token prompt'
        rows.append((label, res[0] if res else None, err))
        print(f'  {label}: ' + (err or json.dumps(res[0])), file=sys.stderr, flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mode', choices=['agent', 'context'])
    p.add_argument('--url', default=os.environ.get('STRATA_URL', 'http://127.0.0.1:18090'))
    p.add_argument('--corpus', default=os.path.normpath(os.path.join(HERE, '..', '..', 'docs')),
                   help='a directory of .md files to build prompts from (default: the repository docs)')
    p.add_argument('--system-tokens', type=int, default=25000, help='agent: the system prompt size (default 25000)')
    p.add_argument('--turns', type=int, default=3, help='agent: tool-output turns (default 3)')
    p.add_argument('--turn-tokens', type=int, default=2500, help='agent: tokens per tool output (default 2500)')
    p.add_argument('--lengths', type=int, nargs='+', default=[1024, 4096, 16384, 32768, 65536],
                   help='context: prompt sizes in tokens')
    p.add_argument('--gen', type=int, default=None, help='tokens to generate (agent 128, context 256)')
    p.add_argument('--timeout', type=int, default=7200)
    p.add_argument('--json', help='also write the rows to this file')
    a = p.parse_args()
    a.gen = a.gen or (128 if a.mode == 'agent' else 256)
    corpus = load_corpus(a.corpus)
    print(f'bench: {a.mode} against {a.url}', file=sys.stderr)
    rows = run_agent(a, corpus) if a.mode == 'agent' else run_context(a, corpus)
    print(table(rows))
    if a.json:
        json.dump([{'request': l, **(r or {}), 'error': e} for l, r, e in rows], open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
