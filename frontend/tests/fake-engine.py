#!/usr/bin/env python3
"""A stand-in for strata / strata-poly that speaks the line protocol (docs/BACKENDS.md), for tests without a GPU.

  GEN <max_new> [key=value ...] <id,id,...>   -> T <id> per token, then DONE
  STOP                                         -> the current request ends with finish=cancel
  QUIT                                         -> exit 0

Environment variables choose the answers, so one script covers the cases:
  FAKE_TOKENS     comma-separated ids to emit (default: the prompt ids echoed back, capped by max_new)
  FAKE_N_TOKENS   emit this many tokens of id FAKE_TOKEN_ID instead (burst tests)
  FAKE_TOKEN_ID   the id used by FAKE_N_TOKENS (default 100)
  FAKE_DELAY_MS   sleep this long between tokens (default 0)
  FAKE_CONTEXT    the context in the READY line (default 4096)
  FAKE_ERR        if set, answer every GEN with "ERR <text>" instead
  FAKE_DIE_AFTER  if set, exit 3 after emitting this many tokens of the first request
  FAKE_NO_STOP    if set, READY without "stop": STOP is ignored and the request runs to its end
  FAKE_PROMPT_N   if set, lie about the prompt size in DONE (mismatch test)
  FAKE_RESUME     if set, a prompt that ends with the start of the answer continues it (the front end's --parallel
                  turns resend the prompt plus the tokens so far), and DONE reports all but its last token reused
  FAKE_FORGET     with FAKE_RESUME: report nothing reused when continuing (an engine that could not park the answer)
  FAKE_LOG        if set, append "GEN <prompt_n> <max_new> <continued> <produced> <finish>" per request to this file
"""
import os
import select
import sys
import time

out = sys.stdout


def say(s):
    out.write(s + "\n")
    out.flush()


env = os.environ.get
say("INFO engine=fake batch_slots=0")
say("READY %s%s" % (env("FAKE_CONTEXT", "4096"), "" if env("FAKE_NO_STOP") else " stop"))

first = True
pending = []  # lines read while checking for STOP that are not STOP/QUIT


def readline():
    if pending:
        return pending.pop(0)
    line = sys.stdin.readline()
    if line == "":
        sys.exit(0)
    return line.rstrip("\n")


def stop_requested():
    """Consume what is readable now; True if a STOP was among it."""
    stop = False
    while select.select([sys.stdin], [], [], 0)[0]:
        line = sys.stdin.readline()
        if line == "":
            sys.exit(0)
        line = line.rstrip("\n")
        if line == "STOP":
            stop = True
        elif line == "QUIT":
            sys.exit(0)
        else:
            pending.append(line)
    return stop


while True:
    line = readline()
    if line == "QUIT":
        sys.exit(0)
    if line == "STOP" or line == "":
        continue
    parts = line.split()
    if parts[0] not in ("GEN", "GENI") or len(parts) < 3:
        say("ERR expected: GEN <max_new> <id,id,...>")
        continue
    max_new = int(parts[1])
    prompt = [int(x) for x in parts[-1].split(",") if x]
    prompt_n = len(prompt)
    if env("FAKE_ERR"):
        say("ERR " + env("FAKE_ERR"))
        continue
    continued = 0
    if env("FAKE_N_TOKENS"):
        tid = int(env("FAKE_TOKEN_ID", "100"))
        n = int(env("FAKE_N_TOKENS"))
        if env("FAKE_RESUME"):
            while continued < min(n, prompt_n) and prompt[prompt_n - 1 - continued] == tid:
                continued += 1
        tokens = [tid] * (n - continued)
    elif env("FAKE_TOKENS"):
        tokens = [int(x) for x in env("FAKE_TOKENS").split(",") if x]
        if env("FAKE_RESUME"):
            for k in range(min(len(tokens), prompt_n), 0, -1):
                if prompt[prompt_n - k:] == tokens[:k]:
                    continued = k
                    break
            tokens = tokens[continued:]
    else:
        tokens = prompt
    reused = prompt_n - 1 if continued and not env("FAKE_FORGET") else 0
    say("RESUME %d" % reused)
    say("PP %d %d 1 1000" % (prompt_n, prompt_n))
    delay = float(env("FAKE_DELAY_MS", "0")) / 1000.0
    die_after = int(env("FAKE_DIE_AFTER", "0") or 0)
    produced = 0
    finish = "stop"
    for t in tokens:
        if produced >= max_new:
            finish = "length"
            break
        if stop_requested() and not env("FAKE_NO_STOP"):
            finish = "cancel"
            break
        say("T %d" % t)
        produced += 1
        if die_after and first and produced >= die_after:
            sys.exit(3)
        if delay:
            time.sleep(delay)
    first = False
    if env("FAKE_LOG"):
        with open(env("FAKE_LOG"), "a") as f:
            f.write("GEN %d %d %d %d %s\n" % (prompt_n, max_new, continued, produced, finish))
    # DONE <generated> <prompt> <prompt_ms> <decode_ms> <finish> <drafts_accepted> <drafts_offered> <reused>
    say("DONE %d %s 12.5 %.1f %s 0 0 %d" % (produced, env("FAKE_PROMPT_N", str(prompt_n)), produced * 10.0, finish, reused))
