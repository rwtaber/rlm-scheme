(define-meta name "tiered_review")
(define-meta version "1.0.0")
(define-meta summary
  "Fast pass over items, filter uncertain ones, expensive review of uncertain subset.")
(define-meta task-shapes '(Batch Classify Validate))
(define-meta data-shapes '(FlatList Tabular))
(define-meta output-shape 'many)

(define-meta trigger
  '((> item_count 1)
    (eq? independent #t)))

(define-meta reject '())

(define-meta slots
  '((context_id              (type string) (pattern "^ctx_") (required #t))
    (items_path              (type string) (default "$"))
    (fast_instruction        (type string) (min-length 10) (required #t))
    (expensive_instruction   (type string) (min-length 10) (required #t))
    (fast_model              (type string) (default "fast_text_model"))
    (expensive_model         (type string) (default "quality_text_model"))
    (uncertainty_threshold   (type number) (min 0) (max 1) (default 0.5))
    (max_concurrent          (type integer) (min 1) (max 50) (default 20))))

(define-meta structural-profile
  '((expected-calls "N + U (where U = uncertain count)")
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

;; Fast pass
(define fast_results
  (map-async
    (lambda (item)
      (llm-query-async
        #:instruction "{{fast_instruction}}"
        #:data item
        #:model "{{fast_model}}"
        #:json #t))
    items
    #:max-concurrent {{max_concurrent}}))

;; Filter uncertain items using Python bridge
(py-set! "fast_results" fast_results)
(py-set! "items" items)
(py-set! "threshold" {{uncertainty_threshold}})

(define uncertain_indices
  (py-eval "import json; [i for i, r in enumerate(fast_results) if json.loads(r).get('uncertainty', 1.0) >= threshold]"))

(define uncertain_items
  (py-eval "[items[i] for i in uncertain_indices]"))

;; Expensive review of uncertain subset
(define expensive_results
  (map-async
    (lambda (item)
      (llm-query-async
        #:instruction "{{expensive_instruction}}"
        #:data item
        #:model "{{expensive_model}}"
        #:json #t))
    uncertain_items
    #:max-concurrent {{max_concurrent}}))

;; Merge results
(py-set! "expensive_results" expensive_results)
(define merged
  (py-eval "merged = list(fast_results); [merged.__setitem__(uncertain_indices[i], expensive_results[i]) for i in range(len(uncertain_indices))]; merged"))

(finish merged)
