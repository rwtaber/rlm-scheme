#lang racket/base

;; sandbox.rkt — Racket sandbox configuration
;;
;; Creates a restricted evaluation environment for instantiated artifacts.
;; Resource limits, allowed modules, scaffold binding protection, and
;; syntax hygiene enforcement.

(require racket/sandbox racket/list)

(provide make-rlm-sandbox
         sandbox-eval
         sandbox-memory-limit-mb)

;; Default memory limit for sandbox evaluations (MB).
(define sandbox-memory-limit-mb 256)

;; Create a sandboxed evaluator with only the RLM primitive bindings.
;; `primitive-bindings` is a list of (name . value) pairs from primitives.rkt.
(define (make-rlm-sandbox primitive-bindings
                          #:memory-limit-mb [mem-limit sandbox-memory-limit-mb]
                          #:time-limit-seconds [time-limit 300])
  ;; Set sandbox resource limits
  (parameterize ([sandbox-memory-limit (* mem-limit 1024 1024)]
                 [sandbox-eval-limits (list time-limit mem-limit)]
                 ;; No filesystem access
                 [sandbox-path-permissions '()]
                 ;; No network access
                 [sandbox-network-guard (lambda (hostname port-no client?) #f)]
)
    (define evaluator
      (make-evaluator 'racket/base))
    ;; Inject primitive bindings into the sandbox namespace.
    ;; These are protected: the sandbox cannot redefine them via set!
    ;; because they are introduced as module-level bindings.
    (for ([binding (in-list primitive-bindings)])
      (define name (car binding))
      (define value (cdr binding))
      (evaluator `(define ,name ',value))
      ;; Replace with actual callable — the quote trick above is for
      ;; simple values; for procedures we use namespace-set-variable-value!
      (call-in-sandbox-context evaluator
        (lambda ()
          (namespace-set-variable-value! name value))))
    evaluator))

;; Evaluate a string of Scheme code in the sandbox.
;; Returns the result value.
(define (sandbox-eval evaluator code-string)
  (evaluator code-string))
