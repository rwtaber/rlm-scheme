(define-meta name "compare_candidates")
(define-meta version "1.0.0")
(define-meta summary "Run multiple strategies in parallel, select best result.")
(define-meta task-shapes '(Compare Search))
(define-meta data-shapes '(Singular Paired))
(define-meta output-shape 'one)

(define-meta trigger '())
(define-meta reject '())

(define-meta slots
  '((context_id           (type string) (pattern "^ctx_") (required #t))
    (candidate_instructions (type string) (required #t))
    (selection_instruction  (type string) (min-length 10) (required #t))
    (model                  (type string) (default "quality_text_model"))
    (max_concurrent         (type integer) (min 1) (max 10) (default 5))))

(define-meta structural-profile
  '((expected-calls "K + 1")
    (critical-path  "2")
    (max-concurrency-slot max_concurrent)
    (recursive-depth 0)
    (uses-python-bridge #f)
    (uses-multimodal #f)))

(define-meta verification-rules
  '(context_id_exists
    expected_calls_within_policy
    only_primitive_bindings))

(define-meta streamable #f)
(define-meta cacheable #t)
(define-meta uses-llm-generated-code #f)

(define data (__context-ref "{{context_id}}" "$"))

(define candidates
  (parallel
    (list
      (lambda ()
        (llm-query-async
          #:instruction "{{candidate_instructions}}"
          #:data data
          #:model "{{model}}")))
    #:max-concurrent {{max_concurrent}}))

(define selected
  (syntax-e
    (llm-query
      #:instruction "{{selection_instruction}}"
      #:data (__join-json candidates)
      #:model "{{model}}")))

(finish selected)
