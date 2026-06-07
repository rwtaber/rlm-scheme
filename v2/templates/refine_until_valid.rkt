(define-meta name "refine_until_valid")
(define-meta version "1.0.0")
(define-meta summary "Refine output in loop until validation passes.")
(define-meta task-shapes '(Refine Validate))
(define-meta data-shapes '(Singular Unknown))
(define-meta output-shape 'one)

(define-meta trigger '())
(define-meta reject '())

(define-meta slots
  '((context_id             (type string) (pattern "^ctx_") (required #t))
    (instruction            (type string) (min-length 1) (required #t))
    (validator_instruction  (type string) (min-length 1) (required #t))
    (model                  (type string) (default "quality_text_model"))
    (max_iterations         (type integer) (min 1) (max 10) (default 3))))

(define-meta structural-profile
  '((expected-calls "2 * max_iterations (worst case)")
    (critical-path  "2 * max_iterations")
    (max-concurrency-slot #f)
    (recursive-depth 0)
    (uses-python-bridge #f)
    (uses-multimodal #f)))

(define-meta verification-rules
  '(context_id_exists
    expected_calls_within_policy
    only_primitive_bindings))

(define-meta streamable #f)
(define-meta cacheable #f)
(define-meta uses-llm-generated-code #f)

(define data (__context-ref "{{context_id}}" "$"))

(define result
  (iterate-until
    (lambda (state)
      (let* ((attempt
               (syntax-e
                 (llm-query
                   #:instruction "{{instruction}}"
                   #:data (if (hash-ref state 'feedback #f)
                              (string-append (hash-ref state 'feedback) "\n\nOriginal data:\n" data)
                              data)
                   #:model "{{model}}")))
             (validation
               (syntax-e
                 (llm-query
                   #:instruction "{{validator_instruction}}"
                   #:data attempt
                   #:model "{{model}}"
                   #:json #t))))
        (hash 'result attempt
              'valid (equal? (hash-ref validation 'valid #f) "true")
              'feedback (hash-ref validation 'feedback ""))))
    (lambda (state) (hash-ref state 'valid #f))
    (hash 'valid #f)
    #:max-iter {{max_iterations}}))

(finish (hash-ref result 'result))
