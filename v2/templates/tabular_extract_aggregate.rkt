(define-meta name "tabular_extract_aggregate")
(define-meta version "1.0.0")
(define-meta summary "Extract from rows/records, aggregate via Python compute.")
(define-meta task-shapes '(Aggregate))
(define-meta data-shapes '(Tabular FlatList))
(define-meta output-shape 'one)

(define-meta trigger
  '((> item_count 1)
    (eq? independent #t)))

(define-meta reject '())

(define-meta slots
  '((context_id          (type string) (pattern "^ctx_") (required #t))
    (items_path          (type string) (default "$"))
    (extract_instruction (type string) (min-length 10) (required #t))
    (extract_model       (type string) (default "fast_text_model"))
    (aggregation_code    (type string) (min-length 1) (required #t))
    (max_concurrent      (type integer) (min 1) (max 50) (default 20))))

(define-meta structural-profile
  '((expected-calls "N")
    (critical-path  "2")
    (max-concurrency-slot max_concurrent)
    (recursive-depth 0)
    (uses-python-bridge #t)
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
        #:instruction "{{extract_instruction}}"
        #:data item
        #:model "{{extract_model}}"
        #:json #t))
    items
    #:max-concurrent {{max_concurrent}}))

(py-set! "extracted" extracted)

(define aggregated
  (py-exec "{{aggregation_code}}"))

(finish aggregated)
