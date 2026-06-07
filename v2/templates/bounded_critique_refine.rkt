(define-meta name "bounded_critique_refine")
(define-meta version "1.0.0")
(define-meta summary "Critique output, refine based on feedback, bounded iterations.")
(define-meta task-shapes '(Refine))
(define-meta data-shapes '(Singular Unknown))
(define-meta output-shape 'one)

(define-meta trigger '())
(define-meta reject '())

(define-meta slots
  '((context_id       (type string) (pattern "^ctx_") (required #t))
    (instruction      (type string) (min-length 1) (required #t))
    (critique_instruction (type string) (min-length 1) (required #t))
    (model            (type string) (default "quality_text_model"))
    (critique_model   (type string) (default "quality_text_model"))
    (max_iterations   (type integer) (min 1) (max 10) (default 3))))

(define-meta structural-profile
  '((expected-calls "3 * max_iterations (worst case)")
    (critical-path  "3 * max_iterations")
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
      (let* ((draft
               (syntax-e
                 (llm-query
                   #:instruction "{{instruction}}"
                   #:data (string-append data "\n\nPrevious critique: "
                                         (or (hash-ref state 'critique) "none"))
                   #:model "{{model}}")))
             (critique
               (syntax-e
                 (llm-query
                   #:instruction "{{critique_instruction}}"
                   #:data draft
                   #:model "{{critique_model}}"
                   #:json #t)))
             (satisfied (equal? (hash-ref critique 'satisfied #f) "true")))
        (hash 'draft draft 'critique critique 'satisfied satisfied)))
    (lambda (state) (hash-ref state 'satisfied #f))
    (hash 'satisfied #f)
    #:max-iter {{max_iterations}}))

(finish (hash-ref result 'draft))
