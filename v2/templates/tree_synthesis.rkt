(define-meta name "tree_synthesis")
(define-meta version "1.0.0")
(define-meta summary "Recursive tree reduction over items for associative synthesis.")
(define-meta task-shapes '(Synthesize))
(define-meta data-shapes '(FlatList ChunkedSingular))
(define-meta output-shape 'one)

(define-meta trigger
  '((> item_count 1)
    (eq? output_type 'one)))

(define-meta reject
  '((and (eq? ordered #t) (eq? order_sensitive #t))))

(define-meta slots
  '((context_id         (type string) (pattern "^ctx_") (required #t))
    (items_path         (type string) (default "$"))
    (reduce_instruction (type string) (min-length 10) (required #t))
    (reduce_model       (type string) (default "quality_text_model"))
    (branch_factor      (type integer) (min 2) (max 10) (default 5))))

(define-meta structural-profile
  '((expected-calls "ceil(N/B) + ceil(ceil(N/B)/B) + ... + 1")
    (critical-path  "ceil(log_B(N))")
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
  (tree-reduce
    (lambda group
      (syntax-e
        (llm-query
          #:instruction "{{reduce_instruction}}"
          #:data (__join-json group)
          #:model "{{reduce_model}}")))
    items
    #:branch-factor {{branch_factor}}))

(finish result)
