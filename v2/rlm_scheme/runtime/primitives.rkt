#lang racket/base

;; primitives.rkt — All public primitive bindings (section 9)
;;
;; Defines the sandbox namespace: llm-query, map-async, tree-reduce,
;; gate, finish, etc. Each primitive calls back to the Python host
;; via the callback protocol for actual LLM/IO operations.

(require json racket/list racket/string racket/control)

(provide make-primitive-bindings)

;; Build the list of (name . procedure) pairs for sandbox injection.
;; `cb-state` is a callback-state from callbacks.rkt.
;; `host-call` and `host-notify` are the callback procedures.
(define (make-primitive-bindings host-call-fn host-notify-fn)

  ;; -- Syntax wrappers --
  ;; In the real system, syntax objects carry provenance metadata.
  ;; For now, we use a simple tagged hash.
  (define (make-syntax value #:source [source "llm"])
    (hasheq 'type "syntax" 'value value 'source source))

  (define (syntax-e stx)
    (if (and (hash? stx) (equal? (hash-ref stx 'type #f) "syntax"))
        (hash-ref stx 'value)
        stx))

  (define (datum->syntax datum)
    (make-syntax datum #:source "datum"))

  ;; -- LLM primitives --
  (define (llm-query #:instruction instruction
                     #:data data
                     #:model [model "fast_text_model"]
                     #:recursive [recursive #f]
                     #:temperature [temperature #f]
                     #:max-tokens [max-tokens #f]
                     #:json [json-mode #f]
                     #:image [image #f]
                     #:images [images '()])
    (define params
      (make-hasheq
       (filter cdr
               (list (cons 'instruction instruction)
                     (cons 'data data)
                     (cons 'model model)
                     (cons 'recursive recursive)
                     (cons 'temperature temperature)
                     (cons 'max_tokens max-tokens)
                     (cons 'json_mode json-mode)
                     (cons 'image image)
                     (cons 'images images)))))
    (define result (host-call-fn "llm_query" params))
    (make-syntax (hash-ref result 'text "")))

  (define (llm-query-async #:instruction instruction
                           #:data data
                           #:model [model "fast_text_model"]
                           #:temperature [temperature #f]
                           #:max-tokens [max-tokens #f]
                           #:json [json-mode #f]
                           #:image [image #f]
                           #:images [images '()])
    (define params
      (make-hasheq
       (filter cdr
               (list (cons 'instruction instruction)
                     (cons 'data data)
                     (cons 'model model)
                     (cons 'temperature temperature)
                     (cons 'max_tokens max-tokens)
                     (cons 'json_mode json-mode)
                     (cons 'image image)
                     (cons 'images images)))))
    (define result (host-call-fn "llm_query_async" params))
    ;; Return an opaque handle
    (hash-ref result 'handle))

  ;; -- Await primitives --
  (define (await-handle handle)
    (define result (host-call-fn "await" (hasheq 'handle handle)))
    (make-syntax (hash-ref result 'text "")))

  (define (await-all handles)
    (define result (host-call-fn "await_all" (hasheq 'handles handles)))
    (hash-ref result 'texts '()))

  (define (await-any handles)
    (define result (host-call-fn "await_any" (hasheq 'handles handles)))
    (values (hash-ref result 'text "")
            (hash-ref result 'remaining '())))

  ;; -- Parallel primitives --
  (define (map-async fn items #:max-concurrent [max-concurrent #f])
    (define handles
      (for/list ([item (in-list items)]
                 [i (in-naturals)])
        ;; Notify progress
        (when (and (> i 0) (= (remainder i 10) 0))
          (host-notify-fn "progress"
                          (hasheq 'completed i 'total (length items))))
        (fn item)))
    ;; Await all handles
    (define result (host-call-fn "await_all" (hasheq 'handles handles)))
    (hash-ref result 'texts '()))

  (define (parallel thunks #:max-concurrent [max-concurrent #f])
    (define handles (map (lambda (thunk) (thunk)) thunks))
    (define result (host-call-fn "await_all" (hasheq 'handles handles)))
    (hash-ref result 'texts '()))

  (define (race thunks)
    (define handles (map (lambda (thunk) (thunk)) thunks))
    (define result (host-call-fn "race" (hasheq 'handles handles)))
    (hash-ref result 'text ""))

  ;; -- Reduction primitives --
  (define (tree-reduce reducer items #:branch-factor [bf 5] #:leaf-fn [leaf-fn #f])
    (when (null? items)
      (error 'tree-reduce "Cannot reduce empty input"))
    (define processed
      (if leaf-fn (map leaf-fn items) items))
    (let loop ([current processed])
      (if (<= (length current) 1)
          (car current)
          (let ([groups (partition-into-groups current bf)])
            (loop (map (lambda (group) (apply reducer group)) groups))))))

  (define (fold-sequential reducer initial items)
    (foldl (lambda (item acc) (reducer acc item)) initial items))

  ;; -- Control primitives --
  (define (sequence . fns)
    (lambda (input)
      (foldl (lambda (fn acc) (fn acc)) input fns)))

  (define (choose predicate then-fn else-fn)
    (lambda (input)
      (if (predicate input)
          (then-fn input)
          (else-fn input))))

  (define (iterate-until step-fn predicate init #:max-iter [max-iter 10])
    (let loop ([state init] [i 0])
      (if (or (predicate state) (>= i max-iter))
          state
          (loop (step-fn state) (add1 i)))))

  ;; -- Gate primitive --
  (define (gate name value #:message [message ""] #:required [required #t])
    (define result
      (host-call-fn "gate"
                    (hasheq 'name name
                            'value (if (string? value) value
                                       (jsexpr->string value))
                            'message message
                            'required required)))
    (when (equal? (hash-ref result 'decision "approve") "reject")
      (error 'gate (format "Gate '~a' rejected: ~a"
                           name (hash-ref result 'reason ""))))
    value)

  ;; -- Delegation --
  (define (recursive-spawn template-name slot-values)
    (lambda (data)
      (define result
        (host-call-fn "recursive_spawn"
                      (hasheq 'template_name template-name
                              'slot_values slot-values
                              'data data)))
      (make-syntax (hash-ref result 'text ""))))

  ;; -- Modifier primitives --
  (define (memoized fn #:key-fn [key-fn values])
    (define cache (make-hash))
    (lambda args
      (define key (apply key-fn args))
      (hash-ref! cache key (lambda () (apply fn args)))))

  (define (with-validation fn validator)
    (lambda args
      (define result (apply fn args))
      (unless (validator result)
        (error 'with-validation "Validation failed"))
      result))

  (define (try-fallback primary-fn fallback-fn)
    (lambda args
      (with-handlers ([exn:fail?
                       (lambda (e)
                         (apply fallback-fn args))])
        (apply primary-fn args))))

  ;; -- State primitives --
  (define (checkpoint key value)
    (host-call-fn "checkpoint" (hasheq 'key key 'value value))
    value)

  (define (restore key)
    (define result (host-call-fn "restore" (hasheq 'key key)))
    (hash-ref result 'value #f))

  (define (tokens-used)
    (host-call-fn "tokens_used" (hasheq)))

  (define (rate-limits)
    (host-call-fn "rate_limits" (hasheq)))

  (define (heartbeat)
    (host-notify-fn "heartbeat" (hasheq)))

  ;; -- Context/data primitives --
  (define (__context-ref context-id json-path)
    (define result
      (host-call-fn "context_ref"
                    (hasheq 'context_id context-id
                            'json_path json-path)))
    (hash-ref result 'data))

  (define (__join-json items)
    (define result
      (host-call-fn "join_json" (hasheq 'items items)))
    (hash-ref result 'json))

  ;; -- Python bridge --
  (define (py-set! name value)
    (host-call-fn "py_set" (hasheq 'name name 'value value))
    (void))

  (define (py-exec code)
    (define result (host-call-fn "py_exec" (hasheq 'code code)))
    (hash-ref result 'output ""))

  (define (py-eval expr)
    (define result (host-call-fn "py_eval" (hasheq 'expr expr)))
    (hash-ref result 'value))

  (define (py-call ref method . args)
    (define result
      (host-call-fn "py_call"
                    (hasheq 'ref ref 'method method 'args args)))
    (hash-ref result 'value))

  ;; -- Finish (delimited continuation) --
  ;; In the real system this uses shift/reset.
  ;; The sandbox wraps artifact evaluation in a reset prompt,
  ;; and finish invokes shift to escape with the value.
  (define finish-tag (make-continuation-prompt-tag 'finish))

  (define (finish value)
    (abort-current-continuation finish-tag value))

  ;; Helper to partition a list into groups of at most n elements
  (define (partition-into-groups lst n)
    (cond
      [(null? lst) '()]
      [(< (length lst) n) (list lst)]
      [else (cons (take lst n)
                  (partition-into-groups (drop lst n) n))]))

  ;; Return all bindings as an association list
  (list
   (cons 'llm-query llm-query)
   (cons 'llm-query-async llm-query-async)
   (cons 'await await-handle)
   (cons 'await-all await-all)
   (cons 'await-any await-any)
   (cons 'map-async map-async)
   (cons 'parallel parallel)
   (cons 'race race)
   (cons 'tree-reduce tree-reduce)
   (cons 'fold-sequential fold-sequential)
   (cons 'sequence sequence)
   (cons 'choose choose)
   (cons 'iterate-until iterate-until)
   (cons 'gate gate)
   (cons 'recursive-spawn recursive-spawn)
   (cons 'memoized memoized)
   (cons 'with-validation with-validation)
   (cons 'try-fallback try-fallback)
   (cons 'checkpoint checkpoint)
   (cons 'restore restore)
   (cons 'tokens-used tokens-used)
   (cons 'rate-limits rate-limits)
   (cons 'heartbeat heartbeat)
   (cons 'py-set! py-set!)
   (cons 'py-exec py-exec)
   (cons 'py-eval py-eval)
   (cons 'py-call py-call)
   (cons 'syntax-e syntax-e)
   (cons 'datum->syntax datum->syntax)
   (cons '__context-ref __context-ref)
   (cons '__join-json __join-json)
   (cons 'finish finish)
   (cons 'finish-tag finish-tag)))
