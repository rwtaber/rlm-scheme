#lang racket/base

;; callbacks.rkt — Host callback protocol
;;
;; JSON-RPC-style messages sent from Racket to the Python host over stdout.
;; The Python host reads these, performs the requested operation (LLM call,
;; checkpoint, Python bridge, etc.), and sends the response back over stdin.
;;
;; Every callback follows the request/response pattern:
;;   Racket writes: {"jsonrpc":"2.0","id":<int>,"method":"<name>","params":{...}}
;;   Python responds: {"jsonrpc":"2.0","id":<int>,"result":{...}}
;;   Or error:        {"jsonrpc":"2.0","id":<int>,"error":{"code":<int>,"message":"..."}}

(require json racket/port racket/string)

(provide host-call
         host-notify
         make-callback-state
         callback-state-next-id)

;; Mutable state for request ID generation.
(struct callback-state ([next-id #:mutable]) #:transparent)

(define (make-callback-state)
  (callback-state 1))

;; Internal: write a JSON-RPC request to stdout and read the response from stdin.
;; Returns the "result" value on success, raises on error.
(define (host-call state method params)
  (define id (callback-state-next-id state))
  (set-callback-state-next-id! state (add1 id))
  (define request
    (hasheq 'jsonrpc "2.0"
            'id id
            'method method
            'params params))
  ;; Write request as single JSON line
  (write-json request (current-output-port))
  (newline (current-output-port))
  (flush-output (current-output-port))
  ;; Read response line
  (define line (read-line (current-input-port) 'any))
  (when (eof-object? line)
    (error 'host-call "Host closed connection"))
  (define response (string->jsexpr line))
  ;; Check for error
  (when (hash-has-key? response 'error)
    (define err (hash-ref response 'error))
    (error 'host-call
           (format "Host error ~a: ~a"
                   (hash-ref err 'code -1)
                   (hash-ref err 'message "unknown"))))
  (hash-ref response 'result))

;; Send a notification (no response expected).
;; Used for progress, heartbeat, streaming partial results.
(define (host-notify method params)
  (define notification
    (hasheq 'jsonrpc "2.0"
            'method method
            'params params))
  (write-json notification (current-output-port))
  (newline (current-output-port))
  (flush-output (current-output-port)))
