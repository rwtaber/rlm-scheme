(define-meta name "batch_map")
(define-meta version "1.0.0")
(define-meta summary "Apply operation independently to each item; preserves order.")
(define-meta task-shapes '(Batch Classify Validate))
(define-meta data-shapes '(FlatList Tabular))
(define-meta output-shape 'many)

(define-meta trigger
  '((> item_count 1)
    (eq? independent #t)))

(define-meta reject '())

(define-meta slots
  '((context_id       (type string) (pattern "^ctx_") (required #t))
    (items_path       (type string) (default "$"))
    (map_instruction  (type string) (min-length 1) (required #t))
    (map_model        (type string) (default "fast_text_model"))
    (max_concurrent   (type integer) (min 1) (max 50) (default 20))
    (json_mode        (type boolean) (default #f))))

(define-meta structural-profile
  '((expected-calls "N")
    (critical-path  "1")
    (max-concurrency-slot max_concurrent)
    (recursive-depth 0)
    (uses-python-bridge #f)
    (uses-multimodal #f)))

(define-meta verification-rules
  '(context_id_exists
    items_path_resolves_to_list
    expected_calls_within_policy
    max_concurrency_within_policy
    only_primitive_bindings))

(define-meta streamable #t)
(define-meta cacheable #t)
(define-meta uses-llm-generated-code #f)

(define items (__context-ref "{{context_id}}" "{{items_path}}"))

(define results
  (map-async
    (lambda (item)
      (llm-query-async
        #:instruction "{{map_instruction}}"
        #:data item
        #:model "{{map_model}}"
        #:json {{json_mode}}))
    items
    #:max-concurrent {{max_concurrent}}))

(finish results)
