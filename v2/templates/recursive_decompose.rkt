(define-meta name "recursive_decompose")
(define-meta version "1.0.0")
(define-meta summary "Recursively decompose into sub-tasks using nested template spawning.")
(define-meta task-shapes '(Decompose))
(define-meta data-shapes '(Hierarchy Singular))
(define-meta output-shape 'one)

(define-meta trigger '())
(define-meta reject '())

(define-meta slots
  '((context_id              (type string) (pattern "^ctx_") (required #t))
    (instruction             (type string) (min-length 10) (required #t))
    (sub_template_name       (type string) (required #t))
    (model                   (type string) (default "quality_text_model"))
    (reduce_instruction      (type string) (min-length 10) (required #t))
    (max_concurrent          (type integer) (min 1) (max 20) (default 5))
    (branch_factor           (type integer) (min 2) (max 10) (default 3))))

(define-meta structural-profile
  '((expected-calls "1 + K * sub_template_calls + ceil(K/B) + ...")
    (critical-path  "2 + sub_template_critical_path + ceil(log_B(K))")
    (max-concurrency-slot max_concurrent)
    (recursive-depth 1)
    (uses-python-bridge #t)
    (uses-multimodal #f)))

(define-meta verification-rules
  '(context_id_exists
    expected_calls_within_policy
    max_concurrency_within_policy
    only_primitive_bindings))

(define-meta streamable #t)
(define-meta cacheable #f)
(define-meta uses-llm-generated-code #f)

(define data (__context-ref "{{context_id}}" "$"))

;; Decompose into sub-tasks
(define subtasks_json
  (syntax-e
    (llm-query
      #:instruction "{{instruction}}"
      #:data data
      #:model "{{model}}"
      #:json #t)))

(py-set! "subtasks_json" subtasks_json)
(define subtasks (py-eval "import json; json.loads(subtasks_json)"))

;; Process each subtask via recursive-spawn
(define sub-process
  (recursive-spawn "{{sub_template_name}}" (hash)))

(define sub-results
  (map-async
    (lambda (subtask) (sub-process subtask))
    subtasks
    #:max-concurrent {{max_concurrent}}))

;; Synthesize results
(define result
  (tree-reduce
    (lambda group
      (syntax-e
        (llm-query
          #:instruction "{{reduce_instruction}}"
          #:data (__join-json group)
          #:model "{{model}}")))
    sub-results
    #:branch-factor {{branch_factor}}))

(finish result)
