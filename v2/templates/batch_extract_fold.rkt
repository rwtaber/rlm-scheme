(define-meta name "batch_extract_fold")
(define-meta version "1.0.0")
(define-meta summary
  "Extract in parallel, then synthesize in order via sequential fold.")
(define-meta task-shapes '(Batch Synthesize Composite))
(define-meta data-shapes '(FlatList ChunkedSingular))
(define-meta output-shape 'one)

(define-meta trigger
  '((> item_count 1)
    (eq? independent #t)
    (eq? output_type 'one)
    (eq? has_second_phase #t)
    (eq? ordered #t)))

(define-meta reject '())

(define-meta slots
  '((context_id       (type string) (pattern "^ctx_") (required #t))
    (items_path       (type string) (default "$"))
    (map_instruction  (type string) (min-length 10) (required #t))
    (fold_instruction (type string) (min-length 10) (required #t))
    (map_model        (type string) (default "fast_text_model"))
    (fold_model       (type string) (default "quality_text_model"))
    (max_concurrent   (type integer) (min 1) (max 50) (default 20))
    (json_mode        (type boolean) (default #f))))

(define-meta structural-profile
  '((expected-calls "2N")
    (critical-path  "1 + N")
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

(define extracted
  (map-async
    (lambda (item)
      (llm-query-async
        #:instruction "{{map_instruction}}"
        #:data item
        #:model "{{map_model}}"
        #:json {{json_mode}}))
    items
    #:max-concurrent {{max_concurrent}}))

(define synthesized
  (fold-sequential
    (lambda (acc item)
      (syntax-e
        (llm-query
          #:instruction "{{fold_instruction}}"
          #:data (__join-json (list acc item))
          #:model "{{fold_model}}")))
    ""
    extracted))

(finish synthesized)
