// UnbiasedStrata: the server context with a Strata engine behind it.
//
// This is the fork's replacement for llama.cpp's tools/server/server-context.cpp. Everything around it is llama.cpp's
// own, compiled unmodified from the submodule: the HTTP layer (server-http), the task queue and response reader
// (server-queue), the task and result types with their OpenAI/Anthropic/Responses JSON (server-task), request parsing,
// tokenization helpers and SSE formatting (server-common), the request schema (server-schema). What this file adds is
// the part that llama.cpp does with llama_context, llama_batch and common_sampler and that the engine does inside
// its own process: run one request at a time through `strata`/`strata-poly`, turn its `T <id>` lines into partial
// results, and its `DONE` line into the final one.
#pragma once

#include "server-http.h"

#include <memory>
#include <string>
#include <vector>

struct common_params;
struct strata_context_impl;

struct strata_context {
    std::unique_ptr<strata_context_impl> impl;

    strata_context();
    ~strata_context();

    // Registers the HTTP routes on `http`. Handlers run only once http.is_ready is set, after load().
    void init_routes(server_http_context & http);

    // fork+exec the engine. Call before any thread exists (before http.start()).
    bool start_engine(const common_params & params, const std::string & engine_exe, const std::vector<std::string> & engine_args, std::string & err);

    // Loads the vocabulary and chat template from the GGUF (vocab only, no tensors) and waits for the engine's READY.
    bool load(common_params & params, std::string & err);
    // --picture-cache: how many pictures' encoded rows are kept for the turns that resend them (0: none); before load()
    void set_picture_cache(size_t entries, size_t bytes);
    // --prefix-file NAME (needs --slot-save-path DIR): the system prompt kept on disk. The first request with a new
    // long system prompt pins the engine's checkpoint at its end and the session is saved to DIR/NAME after the
    // answer; load() restores DIR/NAME, so after a restart a new chat starts reading after its system prompt.
    void set_prefix_file(const std::string & name);
    // --system-tags-to-user a,b: each <a>...</a> block of the system message moves to the start of the first user
    // message, so a system prompt that ends with per-session details (pi's <cwd>, opencode's <env>) is the same
    // prompt in every session and its cached checkpoint matches; before load()
    void set_system_tags_to_user(const std::vector<std::string> & tags);

    // Blocks the calling thread running the task loop until terminate().
    void start_loop();
    void terminate();

    // Non-blocking: true once the engine process has exited; `why` names the signal or exit code.
    bool engine_exited(std::string & why);
    bool engine_alive() const;

    // the context the engine allocated (READY line)
    long long n_ctx() const;
};
