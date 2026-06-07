(define-meta name "race_candidates")
(define-meta version "1.0.0")
(define-meta summary "Launch multiple strategies, return first completed.")
(define-meta task-shapes '(Search))
(define-meta data-shapes '(Singular Unknown))
(define-meta output-shape 'one)

(define-meta trigger '())
(define-meta reject '())

(define-meta slots
  '((context_id  (type string) (pattern "^ctx_") (required #t))
    (instruction (type string) (min-length 1) (required #t))
    (model_a     (type string) (default "fast_text_model"))
    (model_b     (type string) (default "quality_text_model"))))

(define-meta structural-profile
  '((expected-calls "K (but only 1 completes)")
    (critical-path  "1")
    (max-concurrency-slot #f)
    (recursive-depth 0)
    (uses-python-bridge #f)
    (uses-multimodal #f)))

(define-meta verification-rules
  '(context_id_exists
    only_primitive_bindings))

(define-meta streamable #f)
(define-meta cacheable #f)
(define-meta uses-llm-generated-code #f)

(define data (__context-ref "{{context_id}}" "$"))

(define result
  (race
    (list
      (lambda ()
        (llm-query-async
          #:instruction "{{instruction}}"
          #:data data
          #:model "{{model_a}}"))
      (lambda ()
        (llm-query-async
          #:instruction "{{instruction}}"
          #:data data
          #:model "{{model_b}}")))))

(finish result)
