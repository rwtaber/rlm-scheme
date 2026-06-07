#lang racket/base

;; racket_server.rkt — Main entry point
;;
;; Accepts JSON commands from the Python host over stdin/stdout.
;; Dispatches to the sandbox for artifact evaluation and returns results.
;;
;; Protocol:
;;   Python sends: {"id":<int>, "command":"<name>", "params":{...}}
;;   Racket responds: {"id":<int>, "status":"ok", "result":{...}}
;;   Or error:        {"id":<int>, "status":"error", "error":"<message>"}
;;
;; Commands:
;;   "eval"     — evaluate an artifact string in the sandbox
;;   "ping"     — health check, returns {"alive": true}
;;   "shutdown" — clean exit

(require json racket/port racket/string racket/control
         "callbacks.rkt"
         "primitives.rkt"
         "sandbox.rkt")

;; Read one JSON line from stdin. Returns #f on EOF.
(define (read-command)
  (define line (read-line (current-input-port) 'any))
  (if (eof-object? line)
      #f
      (string->jsexpr line)))

;; Write a JSON response to stdout.
(define (write-response resp)
  (write-json resp (current-output-port))
  (newline (current-output-port))
  (flush-output (current-output-port)))

;; Handle the "eval" command: evaluate artifact code in the sandbox.
(define (handle-eval params evaluator finish-tag)
  (define code (hash-ref params 'code ""))
  ;; Wrap evaluation in a reset prompt so (finish value) works
  (define result
    (call-with-continuation-prompt
     (lambda ()
       (sandbox-eval evaluator code))
     finish-tag
     (lambda (value) value)))
  ;; Convert result to JSON-safe value
  (define json-result
    (cond
      [(string? result) result]
      [(number? result) result]
      [(boolean? result) result]
      [(hash? result)
       (if (equal? (hash-ref result 'type #f) "syntax")
           (hash-ref result 'value "")
           result)]
      [(list? result) result]
      [(void? result) 'null]
      [else (format "~a" result)]))
  (hasheq 'value json-result))

;; Main command loop
(define (main)
  (define cb-state (make-callback-state))

  ;; Create host-call and host-notify closures that use cb-state
  (define (do-host-call method params)
    (host-call cb-state method params))
  (define (do-host-notify method params)
    (host-notify method params))

  ;; Build primitive bindings
  (define bindings (make-primitive-bindings do-host-call do-host-notify))

  ;; Extract finish-tag from bindings
  (define finish-tag-binding
    (assoc 'finish-tag bindings))
  (define ft (if finish-tag-binding (cdr finish-tag-binding) #f))

  ;; Create sandbox with primitive bindings (excluding finish-tag meta-binding)
  (define sandbox-bindings
    (filter (lambda (b) (not (eq? (car b) 'finish-tag))) bindings))
  (define evaluator (make-rlm-sandbox sandbox-bindings))

  ;; Command loop
  (let loop ()
    (define cmd (read-command))
    (when cmd
      (define id (hash-ref cmd 'id 0))
      (define command (hash-ref cmd 'command ""))
      (define params (hash-ref cmd 'params (hasheq)))
      (define response
        (with-handlers
            ([exn:fail?
              (lambda (e)
                (hasheq 'id id
                        'status "error"
                        'error (exn-message e)))])
          (define result
            (cond
              [(equal? command "eval")
               (handle-eval params evaluator ft)]
              [(equal? command "ping")
               (hasheq 'alive #t)]
              [(equal? command "shutdown")
               (write-response (hasheq 'id id 'status "ok"
                                       'result (hasheq 'shutdown #t)))
               (exit 0)]
              [else
               (error (format "Unknown command: ~a" command))]))
          (hasheq 'id id 'status "ok" 'result result)))
      (write-response response)
      (loop))))

(main)
