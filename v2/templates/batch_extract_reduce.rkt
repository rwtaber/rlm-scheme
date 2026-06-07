(define-meta name "batch_extract_reduce")
(define-meta version "1.0.0")
(define-meta summary
  "Run independent extraction over many items, then synthesize results with tree reduction.")
(define-meta task-shapes '(Batch Synthesize Composite))
(define-meta data-shapes '(FlatList ChunkedSingular Tabular))
(define-meta output-shape 'one)

(define-meta trigger
  '((> item_count 1)
    (eq? independent #t)
    (eq? output_type 'one)
    (eq? has_second_phase #t)))

(define-meta reject
  '((and (eq? ordered #t) (eq? order_sensitive #t))
    (eq? requires_pairwise_comparison #t)))

(define-meta slots
  '((context_id         (type string) (pattern "^ctx_") (required #t))
    (items_path         (type string) (default "$"))
    (map_instruction    (type string) (min-length 10) (required #t))
    (reduce_instruction (type string) (min-length 10) (required #t))
    (map_model          (type string) (default "fast_text_model"))
    (reduce_model       (type string) (default "quality_text_model"))
    (max_concurrent     (type integer) (min 1) (max 50) (default 20))
    (branch_factor      (type integer) (min 2) (max 10) (default 5))
    (json_mode          (type boolean) (default #f))
    (checkpoint_every   (type integer) (nullable #t) (min 1) (default #f))))

(define-meta structural-profile
  '((expected-calls "N + ceil(N/B) + ceil(ceil(N/B)/B) + ... + 1")
    (critical-path  "1 + ceil(log_B(N))")
    (max-concurrency-slot max_concurrent)
    (recursive-depth 0)
    (uses-python-bridge #f)
    (uses-multimodal #f)))

(define-meta verification-rules
  '(context_id_exists
    items_path_resolves_to_list
    map_model_supports_json_if_json_mode
    expected_calls_within_policy
    max_concurrency_within_policy
    only_primitive_bindings))

(define-meta output-schema
  '((type object)
    (properties
      ((findings (type array)
                 (items ((type object)
                         (properties
                           ((paper_id (type string))
                            (ace2_mentions (type array))
                            (evidence (type string))
                            (uncertainty (type string)))))))
       (summary (type string))))))

(define-meta streamable #t)
(define-meta cacheable #t)

(define-meta budget-policy
  '((on-low-budget   switch-model)
    (low-budget-threshold 0.20)
    (fallback-model  "fast_text_model")
    (on-exhausted    checkpoint-and-stop)))

(define-meta gates
  '((review_extractions
      (description "Review extraction results before synthesis")
      (required #f))))

(define-meta error-policies
  '((extract (on-error fail_fast) (checkpoint-every 25))
    (synthesize (on-error fail_fast))))

(define-meta uses-llm-generated-code #f)

(define-meta examples
  '(((task "Extract claims from papers and synthesize a literature review.")
     (slot_values
       (items_path "$.papers")
       (map_instruction "Extract the core claim, evidence, and uncertainty as JSON.")
       (reduce_instruction "Synthesize the extracted claims into a literature review.")))))

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

(gate "review_extractions" extracted
      #:message "Review extraction results before synthesis.")

(define synthesized
  (tree-reduce
    (lambda group
      (syntax-e
        (llm-query
          #:instruction "{{reduce_instruction}}"
          #:data (__join-json group)
          #:model "{{reduce_model}}")))
    extracted
    #:branch-factor {{branch_factor}}))

(finish synthesized)
