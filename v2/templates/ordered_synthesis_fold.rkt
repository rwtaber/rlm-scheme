(define-meta name "ordered_synthesis_fold")
(define-meta version "1.0.0")
(define-meta summary "Ordered sequential accumulation for order-sensitive synthesis.")
(define-meta task-shapes '(Synthesize))
(define-meta data-shapes '(FlatList ChunkedSingular))
(define-meta output-shape 'one)

(define-meta trigger
  '((> item_count 1)
    (eq? ordered #t)
    (eq? order_sensitive #t)))

(define-meta reject '())

(define-meta slots
  '((context_id       (type string) (pattern "^ctx_") (required #t))
    (items_path       (type string) (default "$"))
    (fold_instruction (type string) (min-length 10) (required #t))
    (fold_model       (type string) (default "quality_text_model"))
    (initial_value    (type string) (default ""))))

(define-meta structural-profile
  '((expected-calls "N")
    (critical-path  "N")
    (max-concurrency-slot #f)
    (recursive-depth 0)
    (uses-python-bridge #f)
    (uses-multimodal #f)))

(define-meta verification-rules
  '(context_id_exists
    items_path_resolves_to_list
    expected_calls_within_policy
    only_primitive_bindings))

(define-meta streamable #f)
(define-meta cacheable #t)
(define-meta uses-llm-generated-code #f)

(define items (__context-ref "{{context_id}}" "{{items_path}}"))

(define result
  (fold-sequential
    (lambda (acc item)
      (syntax-e
        (llm-query
          #:instruction "{{fold_instruction}}"
          #:data (__join-json (list acc item))
          #:model "{{fold_model}}")))
    "{{initial_value}}"
    items))

(finish result)
