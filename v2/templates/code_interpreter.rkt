(define-meta name "code_interpreter")
(define-meta version "1.0.0")
(define-meta summary "Generate and execute Python code for complex computation.")
(define-meta task-shapes '(Direct Aggregate))
(define-meta data-shapes '(Singular Tabular Unknown))
(define-meta output-shape 'one)

(define-meta trigger '())
(define-meta reject '())

(define-meta slots
  '((context_id       (type string) (pattern "^ctx_") (required #t))
    (instruction      (type string) (min-length 1) (required #t))
    (model            (type string) (default "quality_text_model"))
    (max_iterations   (type integer) (min 1) (max 5) (default 2))
    (allowed_imports  (type string) (default "json csv statistics collections re"))))

(define-meta structural-profile
  '((expected-calls "max_iterations (worst case)")
    (critical-path  "max_iterations")
    (max-concurrency-slot #f)
    (recursive-depth 0)
    (uses-python-bridge #t)
    (uses-multimodal #f)))

(define-meta verification-rules
  '(context_id_exists
    expected_calls_within_policy
    only_primitive_bindings))

(define-meta uses-llm-generated-code #t)
(define-meta code-generation-policy
  '((max-code-length 500)
    (allowed-imports (json csv statistics collections re))
    (max-retries 2)
    (sandbox-timeout-seconds 10)))

(define-meta streamable #f)
(define-meta cacheable #f)

(define data (__context-ref "{{context_id}}" "$"))

(define result
  (iterate-until
    (lambda (state)
      (let* ((code
               (syntax-e
                 (llm-query
                   #:instruction (string-append
                     "Write Python code to: {{instruction}}\n"
                     "Allowed imports: {{allowed_imports}}\n"
                     "Previous error: " (or (hash-ref state 'error #f) "none"))
                   #:data data
                   #:model "{{model}}"
                   #:json #f)))
             (exec-result
               (try-fallback
                 (lambda () (py-exec code))
                 (lambda () (hash 'error (string-append "Execution failed"))))))
        (if (hash? exec-result)
            (hash 'code code 'result #f 'error (hash-ref exec-result 'error ""))
            (hash 'code code 'result exec-result 'error #f))))
    (lambda (state) (and (hash-ref state 'result #f) (not (hash-ref state 'error #f))))
    (hash 'error "no attempt yet")
    #:max-iter {{max_iterations}}))

(finish (hash-ref result 'result))
