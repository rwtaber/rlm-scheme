#lang racket/base

;;; racket_server.rkt — Sandboxed Scheme REPL for RLM-Scheme
;;;
;;; This is the security core of RLM-Scheme. It creates a restricted Scheme
;;; evaluator (sandbox) that LLM-generated code runs inside. The sandbox has:
;;;   - No filesystem or network access
;;;   - 256 MB memory limit, 30s CPU timeout
;;;   - Protected scaffold bindings that user code cannot redefine
;;;   - Opaque syntax-object wrappers on all sub-model responses
;;;
;;; Architecture:
;;;   mcp_server.py spawns this as a subprocess. Communication is JSON-over-stdio,
;;;   one object per line in each direction. When sandbox code calls llm-query,
;;;   this server writes a callback request to stdout and blocks on stdin until
;;;   the MCP server responds with the API result. This keeps all real API calls
;;;   in Python while orchestration logic runs in the sandbox.
;;;
;;; Port capture:
;;;   The sandbox redirects stdout/stdin for the user's code (to capture display
;;;   output). We save references to the REAL stdout/stdin before sandbox creation
;;;   so that callback I/O goes through the actual pipes, not the sandbox's
;;;   captured ports.
;;;
;;; Sections:
;;;   1. Requires
;;;   2. Audit log (scope tracking)
;;;   3. Token budget (parameterize-scoped resource control)
;;;   4. Python bridge (isolated subprocess for computation)
;;;   5. LLM callbacks (the callback protocol for sub-model calls)
;;;   6. Sandbox creation and scaffold injection
;;;   7. Eval dispatch (how user code gets evaluated)
;;;   8. JSON command loop (the main server loop)

(require racket/sandbox       ; make-evaluator, sandbox-output, etc.
         racket/control       ; shift/reset (used by finish)
         (prefix-in s: racket/set) ; prefixed to avoid conflict with racket/control's `set`
         racket/string        ; string-trim, string-join
         racket/port          ; get-output (captures sandbox stdout)
         racket/format        ; ~a format specifier
         racket/match         ; pattern matching (used by add-optional-fields)
         json)                ; read-json, write-json, string->jsexpr, jsexpr->string


;; ============================================================
;; Section 1.5: Configuration Constants
;; ============================================================

;; Audit trail preview truncation length (chars)
(define DATUM-PREVIEW-LENGTH 80)

;; Sandbox memory limit (MB)
(define SANDBOX-MEMORY-LIMIT-MB 256)

;; Common error messages (host-side only)
(define ERROR-MCP-CLOSED "MCP server closed connection")


;; ============================================================
;; Section 2: Audit log
;;
;; Every sub-model call, syntax-e unwrap, datum->syntax wrap, and
;; unsafe-* escape hatch is recorded here. The MCP server exposes
;; this via get_scope_log so users can trace exactly what data
;; flowed where in a multi-step pipeline.
;; ============================================================

;; Sentinel struct for (finish val). Defined host-side so sandbox code
;; cannot forge it — only the injected finish function produces these.
;; eval-top-level checks for this to distinguish "finish was called"
;; from "expression returned a non-void value".
(struct finished-value (v))

(define scope-log '())

;; log-scope!: Record an operation in the audit trail for scope tracking.
;;
;; Purpose: Track all data flow operations (llm-query, syntax-e, unsafe-*)
;; so users can trace exactly what data flowed where in multi-step pipelines.
;;
;; Parameters:
;;   op: Operation name (symbol or string) — e.g., 'llm-query, "unsafe-interpolate"
;;   datum: The data being operated on (will be truncated for preview)
;;   scope: Where this happened — "host", "sandbox", "sub-call", or "sub-N"
;;
;; Preview truncation: Full datum remains in sandbox; only first N chars
;; recorded in audit trail to keep log size manageable.
;;
;; Append-only: Maintains operation order. Retrieved via get-scope-log command.
;;
;; Called by: Every scaffold binding that crosses trust boundaries.
(define (log-scope! op datum scope)
  (define preview
    (let ([s (format "~a" datum)])
      (if (> (string-length s) DATUM-PREVIEW-LENGTH)
          (substring s 0 DATUM-PREVIEW-LENGTH)
          s)))
  (set! scope-log
    (append scope-log
      (list (hasheq 'op (if (symbol? op) (symbol->string op) op)
                    'datum_preview preview
                    'scope scope)))))

(define (clear-scope-log!)
  (set! scope-log '()))


;; ============================================================
;; Section 3: Token budget
;;
;; Racket parameters (make-parameter) + parameterize give us
;; lexically-scoped dynamic variables. The token budget starts at
;; +inf.0 (unlimited). User code can scope a budget with:
;;   (parameterize ([token-budget 5000]) ...)
;; Each llm-query decrements the budget by the real token count
;; returned from the API. Nested parameterize forms create
;; independent budgets — the inner scope doesn't affect the outer.
;; ============================================================

(define token-budget (make-parameter +inf.0))

;; decrement-budget!: Deduct tokens from the current parameterized budget.
;;
;; Behavior:
;;   - Reads token-budget parameter (set via parameterize)
;;   - If budget is +inf.0 (unlimited), does nothing
;;   - Otherwise, checks if cost exceeds remaining budget
;;   - Throws error if budget exceeded, else decrements budget
;;
;; Parameterize-scoped: Each (parameterize ([token-budget N]) ...) creates
;; an independent budget scope. Nested scopes don't affect outer scopes.
;;
;; Called by: llm-query-callback after each API response with real token count
(define (decrement-budget! cost)
  (define current (token-budget))
  (when (and (number? current) (< current +inf.0))
    (let ([remaining (- current cost)])
      (when (< remaining 0)
        (error 'llm-query "token budget exceeded"))
      (token-budget remaining))))


;; ============================================================
;; Section 4: Python bridge
;;
;; py_bridge.py is a long-running Python subprocess that handles
;; computation (py-exec, py-eval, py-call). It has full stdlib
;; access but no access to scaffold bindings or the MCP server.
;;
;; IMPORTANT: The bridge must be started BEFORE sandbox creation.
;; start-py-bridge! calls find-executable-path and subprocess,
;; which need filesystem access. The sandbox's security guard
;; blocks filesystem access, so starting the bridge inside the
;; sandbox context would fail. Once running, py-send! only does
;; pipe I/O, which the sandbox allows.
;; ============================================================

(define py-proc #f)
(define py-in #f)   ; port we write to (subprocess's stdin)
(define py-out #f)  ; port we read from (subprocess's stdout)

(define (start-py-bridge!)
  ;; subprocess returns 4 values: process, stdout-port, stdin-port, stderr-port.
  ;; We redirect stderr to our own stderr (current-error-port).
  ;; Use RLM_PYTHON env var if set (to inherit project venv), else bare python3/python.
  (define python-path
    (let ([env-python (getenv "RLM_PYTHON")])
      (if env-python
          (find-executable-path env-python)
          (or (find-executable-path "python3")
              (find-executable-path "python")
              (error 'start-py-bridge "Cannot find python3 or python in PATH")))))
  (define-values (proc out in err)
    (subprocess #f #f (current-error-port)
                python-path
                (path->string
                 (build-path (current-directory) "py_bridge.py"))))
  (set! py-proc proc)
  (set! py-in in)
  (set! py-out out)
  (file-stream-buffer-mode in 'line))

(define (ensure-py-bridge!)
  ;; Called before sandbox creation. Starts the bridge if not already running.
  (unless (and py-proc (eq? (subprocess-status py-proc) 'running))
    (start-py-bridge!)))

(define (py-send! cmd)
  ;; Send a JSON command to py_bridge, read the JSON response.
  ;; This is injected into the sandbox as __py-send! — sandbox code
  ;; calls it via py-exec/py-eval/py-call wrappers.
  (unless (and py-proc (eq? (subprocess-status py-proc) 'running))
    (error 'py-bridge "Python subprocess not running — call ensure-py-bridge! first"))
  (write-string (string-append (jsexpr->string cmd) "\n") py-in)
  (flush-output py-in)
  (define line (read-line py-out 'linefeed))
  (when (eof-object? line)
    (error 'py-bridge "Python subprocess died"))
  (string->jsexpr line))

;; ============================================================
;; Section 5: LLM callbacks
;;
;; The callback protocol is the core of the architecture. When
;; sandbox code calls llm-query, execution reaches a host-side
;; closure (llm-query-callback) that:
;;   1. Writes a JSON request to REAL stdout (the pipe to mcp_server.py)
;;   2. Blocks reading REAL stdin (waiting for the API response)
;;   3. Returns the result text to the sandbox
;;
;; Why real-stdout/real-stdin? The sandbox captures stdout/stdin
;; for the user's code (so `display` output gets collected). But
;; callbacks need to talk to the MCP server through the actual
;; process pipes. We save references to the real ports here,
;; before sandbox creation redirects them.
;;
;; Three callback types:
;;   - llm-query-callback: synchronous, blocks until response
;;   - llm-query-async-callback: returns immediately with a pending ID
;;   - await-callback: blocks until a specific async result is ready
;; ============================================================

;; Save real I/O ports before sandbox creation redirects them.
;; These are used by all callback functions to talk to mcp_server.py.
(define real-stdout (current-output-port))
(define real-stdin (current-input-port))

;; Helper: Add optional fields to hash only if values are present.
;; Used by callbacks to conditionally include temperature, max_tokens, images.
;; Accepts list of (key value [predicate]) pairs.
(define (add-optional-fields base . field-specs)
  (for/fold ([h base])
            ([spec (in-list field-specs)])
    (match spec
      ;; (key value predicate) — include if (predicate value) is true
      [(list key val pred)
       (if (pred val) (hash-set h key val) h)]
      ;; (key value) — include if value is truthy (not #f, not null)
      [(list key val)
       (if (and val (not (null? val))) (hash-set h key val) h)])))

;; --- Synchronous callback ---

(define (llm-query-callback instruction data model recursive temperature max-tokens json-mode images)
  ;; === Phase 1: Validate inputs ===
  (when (and json-mode
             (not (string-contains? (string-downcase instruction) "json")))
    (error 'llm-query
      "#:json #t requires the word 'json' in #:instruction (OpenAI API requirement). Example: \"Return a JSON object with keys: ...\""))

  ;; === Phase 2: Build callback JSON ===
  (define current-budget (token-budget))
  (define payload
    (add-optional-fields
      ;; Base fields — always present
      (hasheq 'op "llm-query"
              'instruction instruction
              'data data
              'model model
              'recursive (if recursive #t #f)
              ;; +inf.0 is not valid JSON, so we send 'null when unlimited.
              'budget (if (< current-budget +inf.0) current-budget 'null)
              'json_mode (if json-mode #t #f))
      ;; Optional fields — only included when explicitly set (not #f)
      (list 'temperature temperature)
      (list 'max_tokens max-tokens)
      (list 'images images)))

  ;; === Phase 3: Send request to MCP server ===
  (write-json payload real-stdout)
  (newline real-stdout)
  (flush-output real-stdout)

  ;; === Phase 4: Block awaiting response ===
  (define line (read-line real-stdin 'linefeed))
  (when (eof-object? line)
    (error 'llm-query ERROR-MCP-CLOSED))
  (define resp (string->jsexpr line))
  (define result-text (hash-ref resp 'result ""))

  ;; === Phase 5: Extract token usage ===
  (define prompt-tokens (hash-ref resp 'prompt_tokens 0))
  (define completion-tokens (hash-ref resp 'completion_tokens 0))
  (define total-tokens (+ prompt-tokens completion-tokens))

  ;; === Phase 6: Log and decrement budget ===
  (log-scope! 'llm-query
              (format "~a tokens (~a in, ~a out): ~a"
                      total-tokens prompt-tokens completion-tokens
                      (let ([s result-text])
                        (if (> (string-length s) 60) (substring s 0 60) s)))
              "sub-call")
  (decrement-budget! total-tokens)
  result-text)

;; --- Async callback ---
;;
;; The async protocol works differently: llm-query-async-callback writes
;; a request and returns immediately with a unique ID. The MCP server
;; dispatches the API call in a thread pool. Later, await-callback sends
;; an "await" message with the ID, and the MCP server blocks until that
;; specific result is ready.

(define async-counter 0)

(define (llm-query-async-callback instruction data model temperature max-tokens json-mode images)
  ;; Validate JSON mode requirements (OpenAI API requirement)
  (when (and json-mode
             (not (string-contains? (string-downcase instruction) "json")))
    (error 'llm-query-async
      "#:json #t requires the word 'json' in #:instruction (OpenAI API requirement). Example: \"Return a JSON object with keys: ...\""))

  ;; Generate a unique ID for this pending call.
  (define id (format "pending_~a" async-counter))
  (set! async-counter (+ async-counter 1))

  ;; Build callback JSON with optional fields.
  (define payload
    (add-optional-fields
      (hasheq 'op "llm-query-async"
              'id id
              'instruction instruction
              'data data
              'model model
              'json_mode (if json-mode #t #f))
      (list 'temperature temperature)
      (list 'max_tokens max-tokens)
      (list 'images images)))

  ;; Write and flush — no response expected, Racket continues immediately.
  (write-json payload real-stdout)
  (newline real-stdout)
  (flush-output real-stdout)
  id)

(define (await-callback id)
  ;; Send an await request, then block until the MCP server has the result.
  (write-json (hasheq 'op "await" 'id id)
              real-stdout)
  (newline real-stdout)
  (flush-output real-stdout)

  (define line (read-line real-stdin 'linefeed))
  (when (eof-object? line)
    (error 'await ERROR-MCP-CLOSED))
  (define resp (string->jsexpr line))
  (define result-text (hash-ref resp 'result ""))

  ;; Token accounting happens at await time, not at dispatch time,
  ;; because that's when we know the actual token count.
  (define prompt-tokens (hash-ref resp 'prompt_tokens 0))
  (define completion-tokens (hash-ref resp 'completion_tokens 0))
  (define total-tokens (+ prompt-tokens completion-tokens))
  (log-scope! 'llm-query-async
              (format "await ~a: ~a tokens" id total-tokens)
              "sub-call")
  (decrement-budget! total-tokens)
  result-text)

(define (await-batch-callback ids)
  ;; Send an await-batch request for multiple IDs, get all results at once.
  (write-json (hasheq 'op "await-batch" 'ids ids)
              real-stdout)
  (newline real-stdout)
  (flush-output real-stdout)

  (define line (read-line real-stdin 'linefeed))
  (when (eof-object? line)
    (error 'await-batch ERROR-MCP-CLOSED))
  (define resp (string->jsexpr line))
  (define results (hash-ref resp 'results '()))

  ;; Process each result: extract text, account for tokens
  (for/list ([result results] [id ids])
    (define result-text (hash-ref result 'result ""))
    (define prompt-tokens (hash-ref result 'prompt_tokens 0))
    (define completion-tokens (hash-ref result 'completion_tokens 0))
    (define total-tokens (+ prompt-tokens completion-tokens))
    (log-scope! 'llm-query-async
                (format "await ~a: ~a tokens" id total-tokens)
                "sub-call")
    (decrement-budget! total-tokens)
    result-text))

(define (await-any-callback ids)
  ;; Send an await-any request, get first completed result + remaining IDs.
  (write-json (hasheq 'op "await-any" 'ids ids)
              real-stdout)
  (newline real-stdout)
  (flush-output real-stdout)

  (define line (read-line real-stdin 'linefeed))
  (when (eof-object? line)
    (error 'await-any ERROR-MCP-CLOSED))
  (define resp (string->jsexpr line))

  ;; Check for error
  (when (hash-has-key? resp 'error)
    (error 'await-any (hash-ref resp 'error)))

  (define completed-id (hash-ref resp 'completed_id #f))
  (define result-text (hash-ref resp 'result ""))
  (define remaining-ids (hash-ref resp 'remaining_ids '()))
  (define prompt-tokens (hash-ref resp 'prompt_tokens 0))
  (define completion-tokens (hash-ref resp 'completion_tokens 0))
  (define total-tokens (+ prompt-tokens completion-tokens))

  (log-scope! 'llm-query-async
              (format "await-any ~a: ~a tokens" completed-id total-tokens)
              "sub-call")
  (decrement-budget! total-tokens)

  ;; Return (values completed-result remaining-ids)
  (values result-text remaining-ids))

(define (tokens-used-callback)
  ;; Query the MCP server for cumulative token usage.
  (write-json (hasheq 'op "tokens-used") real-stdout)
  (newline real-stdout)
  (flush-output real-stdout)
  (define line (read-line real-stdin 'linefeed))
  (when (eof-object? line)
    (error 'tokens-used ERROR-MCP-CLOSED))
  (string->jsexpr line))

(define (rate-limits-callback)
  ;; Query the MCP server for current rate limit state.
  (write-json (hasheq 'op "rate-limits") real-stdout)
  (newline real-stdout)
  (flush-output real-stdout)
  (define line (read-line real-stdin 'linefeed))
  (when (eof-object? line)
    (error 'rate-limits ERROR-MCP-CLOSED))
  (string->jsexpr line))

(define (checkpoint-callback key value)
  ;; L7: Save value to disk under key, return the value for chaining.
  (write-json (hasheq 'op "checkpoint" 'key key 'value value) real-stdout)
  (newline real-stdout)
  (flush-output real-stdout)
  (define line (read-line real-stdin 'linefeed))
  (when (eof-object? line)
    (error 'checkpoint ERROR-MCP-CLOSED))
  (define resp (string->jsexpr line))
  (define status (hash-ref resp 'status "error"))
  (when (equal? status "error")
    (error 'checkpoint (hash-ref resp 'message "checkpoint failed")))
  value)

(define (restore-callback key)
  ;; L7: Load a previously saved checkpoint, return value or #f if not found.
  (write-json (hasheq 'op "restore" 'key key) real-stdout)
  (newline real-stdout)
  (flush-output real-stdout)
  (define line (read-line real-stdin 'linefeed))
  (when (eof-object? line)
    (error 'restore ERROR-MCP-CLOSED))
  (define resp (string->jsexpr line))
  (hash-ref resp 'value #f))

(define (heartbeat-callback)
  ;; Send a heartbeat to the MCP server to reset the idle timeout.
  ;; Used by scaffold functions (map-async) and available to user code
  ;; via (heartbeat) to keep the connection alive during long computations.
  (write-json (hasheq 'op "heartbeat") real-stdout)
  (newline real-stdout)
  (flush-output real-stdout)
  (define line (read-line real-stdin 'linefeed))
  (when (eof-object? line)
    (error 'heartbeat ERROR-MCP-CLOSED))
  ;; Response is just an ACK, no need to parse
  (void))

;; ============================================================
;; Section 6: Sandbox creation and scaffold injection
;;
;; This section creates the restricted Scheme evaluator and injects
;; all scaffold bindings. The bindings fall into five groups:
;;
;;   A. Host-side closures (prefixed with __) — give sandbox code
;;      access to host functions (logging, Python bridge, LLM callbacks)
;;      without exposing filesystem or network.
;;
;;   B. Core bindings — finish, finish-var, context. These are the
;;      basic I/O interface for the sandbox.
;;
;;   C. Scope-tracking wrappers — syntax-e and datum->syntax. These
;;      shadow racket/base's built-in versions with versions that log
;;      every wrap/unwrap to the audit trail.
;;
;;   D. Sub-model call bindings — llm-query, llm-query-async, await,
;;      unsafe-raw-query. These call the host-side callbacks and
;;      wrap/unwrap results.
;;
;;   E. Escape hatches — unsafe-interpolate, unsafe-overwrite,
;;      unsafe-exec-sub-output. These deliberately bypass safety
;;      guarantees. All are logged.
;;
;;   F. Python bridge bindings — py-exec, py-eval, py-call. These
;;      forward to the isolated Python subprocess.
;;
;; Scaffold protection: user code cannot redefine any scaffold binding
;; via (define ...). The eval dispatch (Section 7) checks every define
;; against scaffold-names before evaluating it.
;; ============================================================

;; The sandbox evaluator function. Set by create-sandbox!, used by
;; eval-top-level and handle-command.
(define sandbox-eval #f)

;; Snapshot of all namespace symbols after scaffold injection.
;; Used by get-user-variables to subtract built-in names.
(define initial-symbols (s:set))

;; All names that user code is forbidden from redefining.
;; Includes both user-facing bindings and internal __ prefixed ones.
(define scaffold-names
  (s:set 'finish 'finish-var 'llm-query 'llm-query-async 'await
         'await-all 'await-all-syntax 'map-async
         'syntax-e 'datum->syntax 'tokens-used 'rate-limits
         'py-exec 'py-eval 'py-call 'py-set! 'context 'token-budget
         'unsafe-raw-query 'unsafe-interpolate 'unsafe-overwrite
         'unsafe-exec-sub-output
         'checkpoint 'restore 'heartbeat
         ;; Internal bindings — not user-facing but must be protected
         ;; so user code can't break the scaffold by redefining them.
         '__log-scope! '__py-send! '__llm-query-callback
         '__llm-query-async-callback '__await-callback
         '__tokens-used-callback '__rate-limits-callback
         '__checkpoint-callback '__restore-callback '__heartbeat-callback
         'racket-syntax-e 'racket-datum->syntax))


(define (create-sandbox!)
  ;; Start Python bridge before sandbox creation — needs filesystem access.
  (ensure-py-bridge!)

  ;; ---- Create the evaluator ----
  ;; sandbox-output 'string: captures (display ...) output as a string
  ;; sandbox-error-output 'string: captures stderr similarly
  ;; sandbox-memory-limit: Memory cap in MB (see SANDBOX-MEMORY-LIMIT-MB)
  ;; sandbox-eval-limits: no timeout (#f) — LLM sub-calls can take minutes.
  (define eval
    (parameterize ([sandbox-output 'string]
                   [sandbox-error-output 'string]
                   [sandbox-memory-limit SANDBOX-MEMORY-LIMIT-MB]
                   [sandbox-eval-limits (list #f SANDBOX-MEMORY-LIMIT-MB)])
      (make-evaluator 'racket/base)))

  ;; racket/control provides shift/reset (used by finish) and
  ;; parameterize (used by token-budget scoping).
  ;; racket/list provides take, drop, first, rest, filter-map, etc.
  ;; racket/string provides string-trim, string-split, string-join, etc.
  (eval '(require racket/control))
  (eval '(require racket/list))
  (eval '(require racket/string))

  ;; ---- Group A: Host-side closures ----
  ;; Inject host-side functions as first-class values. These close over
  ;; host bindings (real-stdout, py-in, scope-log, etc.) but run inside
  ;; the sandbox. The sandbox can call them but can't inspect or replace
  ;; their internals.

  (eval `(define token-budget ,token-budget))
  (eval `(define __log-scope! ,log-scope!))
  (eval `(define __py-send! ,py-send!))
  (eval `(define __llm-query-callback ,llm-query-callback))
  (eval `(define __llm-query-async-callback ,llm-query-async-callback))
  (eval `(define __await-callback ,await-callback))
  (eval `(define __await-batch-callback ,await-batch-callback))
  (eval `(define __await-any-callback ,await-any-callback))
  (eval `(define __tokens-used-callback ,tokens-used-callback))
  (eval `(define __rate-limits-callback ,rate-limits-callback))
  (eval `(define __checkpoint-callback ,checkpoint-callback))
  (eval `(define __restore-callback ,restore-callback))
  (eval `(define __heartbeat-callback ,heartbeat-callback))

  ;; ---- Group B: Core bindings ----

  ;; finish: uses shift to jump out of the reset that wraps each
  ;; top-level expression (see eval-top-level in Section 7). The value
  ;; is wrapped in a finished-value sentinel so eval-top-level can
  ;; distinguish "finish was called" from "expression returned a value".
  ;; The word "finish" in a string literal does nothing — only the
  ;; function call triggers shift.
  (eval `(define (finish val)
           (shift k (,finished-value val))))

  ;; finish-var: look up a variable by string name and return its value.
  ;; Useful when the variable name is computed at runtime.
  (eval '(define (finish-var name)
           (finish (eval (string->symbol name)))))

  ;; context: holds data loaded via load_context. For backward compatibility,
  ;; this remains the default unnamed context. Protected from (define context ...)
  ;; but mutable via (set! context ...) from host side.
  (eval '(define context ""))

  ;; context-store: hash table for named context slots (improvement #5).
  ;; Allows users to manage multiple datasets with clear names.
  ;; Example: (load-context "gwas-data" data) then (get-context "gwas-data")
  (eval '(define context-store (make-hash)))

  ;; get-context: retrieve a named context slot (improvement #5).
  ;; Returns #f if the name doesn't exist.
  (eval '(define (get-context name)
           (hash-ref context-store name #f)))

  ;; ---- Error handling (improvement #6) ----
  ;; try/on-error: graceful error handling for sub-model calls.
  ;; Allows map-async to continue even if individual items fail.
  ;; Usage: (try (llm-query ...) on-error (lambda (err) "FAILED"))
  ;; The error handler receives the error message as a string.
  (eval '(define-syntax try
           (syntax-rules (on-error)
             [(try expr on-error handler)
              (with-handlers ([exn:fail? (lambda (e) (handler (exn-message e)))])
                expr)])))

  ;; ---- Group C: Scope-tracking wrappers ----
  ;; We shadow racket/base's syntax-e and datum->syntax with versions
  ;; that log every call to the audit trail. The originals are saved
  ;; as racket-syntax-e and racket-datum->syntax for internal use.

  ;; Save the originals before shadowing.
  (eval '(define racket-syntax-e syntax-e))
  (eval '(define racket-datum->syntax datum->syntax))

  ;; syntax-e: unwrap a syntax object to get the string inside.
  ;; Pass-through for plain strings (so user code doesn't need to check).
  ;; Every unwrap is logged — the audit trail shows when and where
  ;; untrusted sub-model data entered the trusted text layer.
  (eval '(define (syntax-e stx)
           (define val (if (syntax? stx) (racket-syntax-e stx) stx))
           (__log-scope! "syntax-e" val "sandbox")
           val))

  ;; datum->syntax: wrap a plain string in a syntax object with scope
  ;; metadata. The inverse of syntax-e. Every wrap is logged.
  (eval '(define (datum->syntax ctx datum . rest)
           (__log-scope! "datum->syntax" datum "sandbox")
           (apply racket-datum->syntax ctx datum rest)))

  ;; ---- Group D: Sub-model call bindings ----

  ;; llm-query: the primary sub-model call. Calls the host-side callback
  ;; (which writes to real-stdout, blocks on real-stdin), then wraps the
  ;; result in a syntax object. The caller MUST use syntax-e to unwrap
  ;; before using the text — this is the core of injection safety.
  ;;
  ;; Keyword args (all optional):
  ;;   #:instruction — system prompt for the sub-model
  ;;   #:data — user message (untrusted data goes here)
  ;;   #:model — override default sub-model (e.g., "gpt-4o-mini")
  ;;   #:recursive — if #t, sub-model gets its own sandbox
  ;;   #:temperature — sampling temperature (0.0 = deterministic)
  ;;   #:max-tokens — cap response length
  ;;   #:json — if #t, enable JSON mode (guaranteed valid JSON)
  ;;   #:image — a single image (file path or base64 string)
  ;;   #:images — a list of images (file paths or base64 strings)
  (eval '(define (llm-query #:instruction [instruction ""]
                             #:data [data ""]
                             #:model [model ""]
                             #:recursive [recursive #f]
                             #:temperature [temperature #f]
                             #:max-tokens [max-tokens #f]
                             #:json [json-mode #f]
                             #:image [image #f]
                             #:images [images '()])
           (define all-images (if image (cons image images) images))
           (define result-text (__llm-query-callback instruction data model recursive temperature max-tokens json-mode all-images))
           (datum->syntax #f result-text)))

  ;; unsafe-raw-query: like llm-query but returns a plain string instead
  ;; of a syntax object. Bypasses injection safety. Use when the result
  ;; is final output (no further processing) or is code to be executed.
  (eval '(define (unsafe-raw-query #:instruction [instruction ""]
                                    #:data [data ""]
                                    #:model [model ""]
                                    #:recursive [recursive #f]
                                    #:temperature [temperature #f]
                                    #:max-tokens [max-tokens #f]
                                    #:json [json-mode #f]
                                    #:image [image #f]
                                    #:images [images '()])
           (define all-images (if image (cons image images) images))
           (__llm-query-callback instruction data model recursive temperature max-tokens json-mode all-images)))

  ;; llm-query-async: non-blocking sub-call. Returns an opaque handle
  ;; (a list: '(async-handle "pending_N")). The MCP server dispatches
  ;; the API call in a thread pool. Use (await handle) to collect the
  ;; result later. Does NOT support #:recursive (would need nested
  ;; sandbox lifecycle management in async context).
  (eval '(define (llm-query-async #:instruction [instruction ""]
                                   #:data [data ""]
                                   #:model [model ""]
                                   #:temperature [temperature #f]
                                   #:max-tokens [max-tokens #f]
                                   #:json [json-mode #f]
                                   #:image [image #f]
                                   #:images [images '()]
                                   #:recursive [recursive #f])
           ;; Check for unsupported #:recursive flag
           (when recursive
             (error 'llm-query-async
                    "#:recursive is not supported with async calls. Use synchronous llm-query for recursive delegation with #:recursive #t"))
           (define all-images (if image (cons image images) images))
           (define id (__llm-query-async-callback instruction data model temperature max-tokens json-mode all-images))
           (list 'async-handle id)))

  ;; await: block until an async sub-call completes. Returns a syntax
  ;; object (same as llm-query). Token budget is decremented at await
  ;; time, not at dispatch time, because that's when we know the actual
  ;; token count from the API response.
  (eval '(define (await handle)
           (unless (and (list? handle) (eq? (car handle) 'async-handle))
             (error 'await "expected an async handle from llm-query-async"))
           (define id (cadr handle))
           (define result-text (__await-callback id))
           (datum->syntax #f result-text)))

  ;; await-all: await multiple async handles and return a list of
  ;; unwrapped strings. Uses batch await for efficiency (waits concurrently instead of sequentially).
  (eval '(define (await-all handles)
           (if (null? handles)
               '()
               (let ([ids (map (lambda (h)
                                (unless (and (list? h) (eq? (car h) 'async-handle))
                                  (error 'await-all "expected async handles"))
                                (cadr h))
                              handles)])
                 (__await-batch-callback ids)))))

  ;; await-all-syntax: like await-all but returns syntax objects
  ;; (for callers who want to preserve scope tracking).
  (eval '(define (await-all-syntax handles)
           (if (null? handles)
               '()
               (let ([ids (map (lambda (h)
                                (unless (and (list? h) (eq? (car h) 'async-handle))
                                  (error 'await-all-syntax "expected async handles"))
                                (cadr h))
                              handles)])
                 (define raw-results (__await-batch-callback ids))
                 (map (lambda (res) (datum->syntax #f res)) raw-results)))))

  ;; await-any: wait for ANY handle to complete, return completed result + remaining handles.
  ;; Returns (values completed-result remaining-handles) where completed-result is unwrapped string.
  ;; Use for race patterns, progressive results, timeout patterns, etc.
  (eval '(define (await-any handles)
           (when (null? handles)
             (error 'await-any "cannot await-any on empty list"))
           (let ([ids (map (lambda (h)
                            (unless (and (list? h) (eq? (car h) 'async-handle))
                              (error 'await-any "expected async handles"))
                            (cadr h))
                          handles)])
             (define-values (completed-text remaining-ids) (__await-any-callback ids))
             ;; Reconstruct remaining handles from remaining IDs
             (define remaining-handles
               (map (lambda (id) (list 'async-handle id)) remaining-ids))
             (values completed-text remaining-handles))))

  ;; map-async: Efficient parallel fan-out with optional concurrency limit.
  ;;
  ;; Architecture: Uses pipelined batching — launches initial window of N items,
  ;; then for each completed item, immediately launches next item. This maintains
  ;; N active calls at all times until all items are processed, maximizing
  ;; throughput while respecting concurrency limits.
  ;;
  ;; Three execution paths:
  ;;   1. Empty list: return '() immediately
  ;;   2. Items <= max-concurrent: launch all at once, await-batch
  ;;   3. Items > max-concurrent: use pipelined processing (rolling window)
  ;;
  ;; Why faster than manual map + await:
  ;;   - No sequential bottleneck: next item launches immediately on completion
  ;;   - Efficient batching: await-any returns first completed, not blocking on slowest
  ;;   - Order preservation: results stored by original index, not completion order
  ;;
  ;; Parameters:
  ;;   fn: Function taking one item, returning async handle from llm-query-async
  ;;   items: List of items to process
  ;;   #:max-concurrent: Optional limit (default 20). Set to #f to launch all at once.
  ;;
  ;; Returns: List of unwrapped strings in original item order
  (eval '(define (map-async fn items #:max-concurrent [max-conc 20])
           ;; Constants for progress reporting
           (define MIN-PROGRESS-INTERVAL 10)  ; Report at least every 10 items

           (cond
             [(null? items) '()]
             [(or (not max-conc) (<= (length items) max-conc))
              ;; Simple case: launch all at once
              (define handles (map fn items))
              ;; Validate first result is an async handle
              (when (not (null? handles))
                (define first-handle (car handles))
                (unless (and (list? first-handle)
                            (not (null? first-handle))
                            (eq? (car first-handle) 'async-handle))
                  (error 'map-async
                    "lambda must return an async handle from llm-query-async, not a sync result from llm-query. Use (llm-query-async ...) inside your lambda, not (llm-query ...)")))
              (await-all handles)]
             [else
              ;; Pipelined case: maintain rolling window of max-conc active calls
              (define n (length items))
              (define result-vec (make-vector n #f))

              ;; Nested pipeline processor - closes over fn, n, result-vec
              ;; This is idiomatic Scheme: nested functions capture outer scope
              (define (process-pipeline active-handles active-indices remaining-items next-idx completed)
                (if (null? active-handles)
                    ;; Base case: all work complete
                    (vector->list result-vec)
                    (call-with-values
                      (lambda () (await-any active-handles))
                      (lambda (result rest-handles)
                        ;; 1. Store result at original index (preserves input order)
                        (vector-set! result-vec (car active-indices) result)
                        (define new-completed (+ completed 1))

                        ;; 2. Progress reporting: emit every 10 items or every 10%
                        (define progress-interval (max MIN-PROGRESS-INTERVAL (quotient n 10)))
                        (when (or (= new-completed n)
                                  (zero? (modulo new-completed progress-interval)))
                          (eprintf "map-async: ~a/~a completed\n" new-completed n)
                          ;; Send heartbeat to reset MCP idle timeout during long fan-outs
                          (heartbeat))

                        ;; 3. Decision: launch next item or drain remaining handles
                        (if (null? remaining-items)
                            ;; No more items to launch, just process remaining active handles
                            (process-pipeline rest-handles (cdr active-indices) '() next-idx new-completed)
                            ;; Launch next item immediately to maintain concurrency window
                            (let ([new-handle (fn (car remaining-items))])
                              (process-pipeline
                                (append rest-handles (list new-handle))
                                (append (cdr active-indices) (list next-idx))
                                (cdr remaining-items)
                                (+ next-idx 1)
                                new-completed)))))))

              ;; Launch initial concurrency window
              (define init-count (min max-conc n))
              (define init-batch (take items init-count))
              (define init-remaining (drop items init-count))
              (define init-handles (map fn init-batch))

              ;; Validate first result is an async handle
              (when (not (null? init-handles))
                (define first-handle (car init-handles))
                (unless (and (list? first-handle)
                            (not (null? first-handle))
                            (eq? (car first-handle) 'async-handle))
                  (error 'map-async
                    "lambda must return an async handle from llm-query-async, not a sync result from llm-query. Use (llm-query-async ...) inside your lambda, not (llm-query ...)")))

              (define init-indices (range 0 init-count))
              (process-pipeline init-handles init-indices init-remaining init-count 0)])))

  ;; ---- Group E: Escape hatches ----
  ;; These deliberately bypass safety guarantees. All are logged in the
  ;; audit trail so the scope log shows exactly where breaks occurred.

  ;; unsafe-interpolate: strip the syntax wrapper without the logging
  ;; that syntax-e provides. Logged as "unsafe-interpolate" instead.
  (eval '(define (unsafe-interpolate stx)
           (define val (if (syntax? stx) (racket-syntax-e stx) stx))
           (__log-scope! "unsafe-interpolate" val "sandbox")
           val))

  ;; unsafe-overwrite: overwrite any variable binding (including
  ;; protected scaffold bindings) via set!.
  (eval '(define (unsafe-overwrite name val)
           (__log-scope! "unsafe-overwrite" (format "~a = ~a" name val) "sandbox")
           (eval `(set! ,name (quote ,val)))))

  ;; unsafe-exec-sub-output: evaluate arbitrary code from a string.
  ;; The string is parsed into an S-expression and eval'd in the sandbox.
  ;; Use when a sub-model was asked to generate code and you need to run it.
  (eval '(define (unsafe-exec-sub-output stx)
           (define code-str (if (syntax? stx) (racket-syntax-e stx) stx))
           (__log-scope! "unsafe-exec-sub-output" code-str "sandbox")
           (eval (read (open-input-string code-str)))))

  ;; ---- tokens-used: cumulative token usage ----
  ;; Queries the MCP server for total prompt/completion/call counts.
  (eval '(define (tokens-used)
           (__tokens-used-callback)))

  ;; ---- rate-limits: current API rate limit state ----
  ;; Queries the MCP server for the most recent rate limit headers.
  (eval '(define (rate-limits)
           (__rate-limits-callback)))

  ;; ---- checkpoint: save value to disk under key ----
  ;; L7: Persist values across timeouts. Values must be JSON-serializable.
  ;; Returns the value for chaining.
  (eval '(define (checkpoint key value)
           (__log-scope! "checkpoint" key "sandbox")
           (__checkpoint-callback key value)))

  ;; ---- restore: load a previously saved checkpoint ----
  ;; L7: Retrieve a checkpointed value. Returns the value or #f if not found.
  (eval '(define (restore key)
           (__log-scope! "restore" key "sandbox")
           (__restore-callback key)))

  ;; ---- heartbeat: reset MCP server idle timeout ----
  ;; Sends a heartbeat to prevent the MCP server from killing the Racket
  ;; process during long computations. Called automatically by map-async
  ;; between batches, but also available to user code for custom long-running
  ;; loops. Returns void.
  (eval '(define (heartbeat)
           (__heartbeat-callback)))

  ;; ---- Group F: Python bridge bindings ----
  ;; These forward to the isolated Python subprocess via __py-send!.
  ;; Python has full stdlib access but no access to scaffold bindings,
  ;; the MCP server, or the sandbox's namespace.

  ;; py-exec: run Python code, return captured stdout.
  ;; On error, include the full Python traceback for easier debugging.
  (eval '(define (py-exec code)
           (__log-scope! "py-exec" code "sandbox")
           (define resp (__py-send! (hasheq 'op "exec" 'code code)))
           (when (string=? (hash-ref resp 'status "") "error")
             (define tb (hash-ref resp 'traceback #f))
             (define msg (hash-ref resp 'message "unknown error"))
             (error 'py-exec (if tb (string-append msg "\n" tb) msg)))
           (hash-ref resp 'stdout "")))

  ;; py-eval: evaluate a Python expression, return the value.
  ;; JSON-serializable values come back directly; complex objects return
  ;; a reference handle {"__ref__": "obj_N"} for use with py-call.
  (eval '(define (py-eval expr)
           (__log-scope! "py-eval" expr "sandbox")
           (define resp (__py-send! (hasheq 'op "eval" 'expr expr)))
           (when (string=? (hash-ref resp 'status "") "error")
             (define tb (hash-ref resp 'traceback #f))
             (define msg (hash-ref resp 'message "unknown error"))
             (error 'py-eval (if tb (string-append msg "\n" tb) msg)))
           (hash-ref resp 'value "")))

  ;; py-call: call a method on a Python reference handle.
  ;; The handle stays in Python's memory; only the method result
  ;; comes back over the JSON pipe.
  (eval '(define (py-call ref method . args)
           (__log-scope! "py-call" (format "~a.~a" ref method) "sandbox")
           (define resp (__py-send! (hasheq 'op "call"
                                            'ref ref
                                            'method method
                                            'args args)))
           (when (string=? (hash-ref resp 'status "") "error")
             (define tb (hash-ref resp 'traceback #f))
             (define msg (hash-ref resp 'message "unknown error"))
             (error 'py-call (if tb (string-append msg "\n" tb) msg)))
           (hash-ref resp 'value "")))

  ;; py-set!: safely transfer a Scheme string to a named Python variable.
  ;; The value is sent as JSON data over the pipe — never embedded in code —
  ;; so quotes, backslashes, and unicode are all safe.
  (eval '(define (py-set! name value)
           (__log-scope! "py-set!" (format "~a = ~a" name (if (> (string-length (format "~a" value)) 40)
                                                              (substring (format "~a" value) 0 40)
                                                              (format "~a" value))) "sandbox")
           (define resp (__py-send! (hasheq 'op "set-var"
                                            'name name
                                            'value value)))
           (when (string=? (hash-ref resp 'status "") "error")
             (error 'py-set! (hash-ref resp 'message "unknown error")))
           (void)))

  ;; ---- Snapshot initial namespace ----
  ;; After all scaffold bindings are injected, snapshot every symbol in
  ;; the namespace. get-user-variables subtracts this set to find only
  ;; user-defined names.
  (set! sandbox-eval eval)
  (set! initial-symbols
    (s:list->set
     (call-in-sandbox-context eval
       (lambda () (namespace-mapped-symbols (current-namespace)))))))


(define (get-user-variables)
  ;; Returns a list of variable name strings defined by user code.
  ;; Subtracts initial symbols (everything from racket/base + scaffold)
  ;; and scaffold-names (explicit protection list) from the current namespace.
  (define current
    (s:list->set
     (call-in-sandbox-context sandbox-eval
       (lambda () (namespace-mapped-symbols (current-namespace))))))
  (map symbol->string
       (s:set->list (s:set-subtract current initial-symbols scaffold-names))))

;; ============================================================
;; Section 7: Eval dispatch
;;
;; User code arrives as a string of S-expressions. This section
;; parses it, classifies each top-level form, and evaluates it
;; with the correct semantics:
;;
;;   - (define ...) — evaluated directly (NOT inside reset).
;;     Racket's (define ...) is a declaration, not an expression,
;;     so it cannot appear inside (reset ...). Both value and
;;     function defines are handled identically. Scaffold protection
;;     is checked before evaluation — if the name is in scaffold-names,
;;     the define is rejected with an error.
;;
;;   - (begin ...) — unwrapped and each sub-form dispatched recursively.
;;
;;   - Anything else — wrapped in (reset expr). This is how (finish val)
;;     works: finish calls (shift k val), which jumps out of the reset
;;     and returns val as the expression's result. Without reset, shift
;;     would have no prompt to jump to.
;; ============================================================

(define (read-all-exprs str)
  ;; Parse a string into a list of S-expressions.
  ;; Reads until EOF, returns in source order.
  (define in (open-input-string str))
  (let loop ([exprs '()])
    (define e (read in))
    (if (eof-object? e)
        (reverse exprs)
        (loop (cons e exprs)))))

(define (is-define? expr)
  ;; Is this a (define ...) form?
  (and (pair? expr) (eq? (car expr) 'define)))

(define (is-begin? expr)
  ;; Is this a (begin ...) form?
  (and (pair? expr) (eq? (car expr) 'begin)))

(define (define-name expr)
  ;; Extract the name being defined from a define form.
  ;; Handles both (define x 42) and (define (f x) ...).
  (define target (cadr expr))
  (if (pair? target) (car target) target))

(define (check-scaffold-protection! name)
  ;; Raise an error if the user tries to redefine a scaffold binding.
  ;; This is the enforcement mechanism for namespace safety — it runs
  ;; before the define is evaluated, so the binding never gets replaced.
  (when (s:set-member? scaffold-names name)
    (error 'define
           (format "cannot redefine scaffold binding '~a'" name))))

(define (eval-top-level exprs)
  ;; Evaluate a list of top-level expressions with define/begin/reset dispatch.
  ;; Returns three values: status ("ok" or "finished"), result string, stdout string.
  ;; Stops early if any expression produces a "finished" status (via finish/shift).
  (define collected-stdout "")

  (define-values (status result)
    (for/fold ([status "ok"] [result ""])
              ([expr (in-list exprs)]
               #:break (string=? status "finished"))
      (cond
        ;; (define ...) — evaluate directly, check scaffold protection first.
        [(is-define? expr)
         (check-scaffold-protection! (define-name expr))
         (sandbox-eval expr)
         (let ([stdout (get-output sandbox-eval)])
           (set! collected-stdout (string-append collected-stdout stdout))
           (values "ok" ""))]

        ;; (begin ...) — unwrap and dispatch each sub-form recursively.
        [(is-begin? expr)
         (define-values (s r out) (eval-top-level (cdr expr)))
         (set! collected-stdout (string-append collected-stdout out))
         (values s r)]

        ;; Everything else — wrap in (reset ...) so (finish ...) can
        ;; use (shift ...) to jump out. Only a finished-value sentinel
        ;; (produced by finish) triggers "finished" status. Other non-void
        ;; returns (e.g. from py-exec, llm-query, bare expressions) are
        ;; treated as normal and evaluation continues.
        [else
         (define val (sandbox-eval `(reset ,expr)))
         (let ([stdout (get-output sandbox-eval)])
           (set! collected-stdout (string-append collected-stdout stdout)))
         (if (finished-value? val)
             (values "finished" (format "~a" (finished-value-v val)))
             (values "ok" ""))])))

  (values status result collected-stdout))


;; ============================================================
;; Section 8: JSON command loop
;;
;; The main server loop. Reads JSON commands from stdin, dispatches
;; them to handle-command, writes JSON responses to stdout. Each
;; command is one of: eval, load-context, get-scope-log,
;; get-variables, reset.
;;
;; Error handling: any unhandled exception during command processing
;; is caught and returned as a JSON error response, keeping the
;; server alive for the next command.
;; ============================================================

(define (handle-command cmd)
  (define op (hash-ref cmd 'op ""))
  (cond
    ;; eval — parse and evaluate Scheme code in the sandbox.
    [(string=? op "eval")
     (define code (hash-ref cmd 'code ""))
     (with-handlers ([exn:fail?
                      (lambda (e)
                        (hasheq 'status "error"
                                'message (exn-message e)))])
       (define exprs (read-all-exprs code))
       (define-values (status result stdout) (eval-top-level exprs))
       (hasheq 'status status
               'result result
               'stdout stdout))]

    ;; load-context — set context variable and optionally store in named slot.
    ;; Accepts optional 'name parameter for named context slots (improvement #5).
    [(string=? op "load-context")
     (define data (hash-ref cmd 'data ""))
     (define name (hash-ref cmd 'name #f))

     ;; If name provided, store in named slot
     (when name
       (sandbox-eval `(hash-set! context-store ,name ,data)))

     ;; Always update default context variable for backward compatibility
     (sandbox-eval `(set! context ,data))

     ;; Also forward to py_bridge so Python code can access context
     (when (and py-proc (eq? (subprocess-status py-proc) 'running))
       (py-send! (hasheq 'op "set-context" 'data data)))

     (hasheq 'status "ok"
             'result (if name
                        (format "context loaded as '~a'" name)
                        "context loaded")
             'meta (hasheq 'bytes (string-length data)
                           'type "string"))]

    ;; get-scope-log — return the audit trail as JSON.
    [(string=? op "get-scope-log")
     (hasheq 'status "ok"
             'result (jsexpr->string scope-log))]

    ;; get-variables — return user-defined variable names.
    [(string=? op "get-variables")
     (hasheq 'status "ok"
             'result (jsexpr->string (get-user-variables)))]

    ;; inspect-state — return comprehensive sandbox state for debugging.
    [(string=? op "inspect-state")
     (define vars (get-user-variables))
     (define py-available (and py-proc (eq? (subprocess-status py-proc) 'running)))
     (hasheq 'status "ok"
             'variables vars
             'python_available py-available)]

    ;; reset — tear down and recreate everything.
    [(string=? op "reset")
     (clear-scope-log!)
     ;; Kill py_bridge first — create-sandbox! will start a fresh one
     ;; via ensure-py-bridge!.
     (when py-proc
       (subprocess-kill py-proc #t)
       (set! py-proc #f))
     (create-sandbox!)
     (hasheq 'status "ok"
             'result "sandbox reset")]

    [else
     (hasheq 'status "error"
             'message (format "unknown op: ~a" op))]))


(define (main)
  ;; Create the sandbox (also starts py_bridge).
  (create-sandbox!)

  ;; Main loop: read one JSON command per line, process it, write
  ;; one JSON response per line. The loop runs until stdin closes
  ;; (i.e., the MCP server terminates this process).
  (let loop ()
    (define line (read-line (current-input-port) 'linefeed))
    (unless (eof-object? line)
      (define trimmed (string-trim line))
      (unless (string=? trimmed "")
        (define response
          ;; Catch any exception so a single bad command doesn't crash
          ;; the server — return it as an error response instead.
          (with-handlers ([exn:fail?
                           (lambda (e)
                             (hasheq 'status "error"
                                     'message (exn-message e)))])
            (define cmd (string->jsexpr trimmed))
            (handle-command cmd)))
        (write-json response (current-output-port))
        (newline (current-output-port))
        (flush-output (current-output-port)))
      (loop))))

(main)
