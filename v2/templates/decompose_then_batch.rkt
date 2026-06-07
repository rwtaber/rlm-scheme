(define-meta name "decompose_then_batch")
(define-meta version "1.0.0")
(define-meta summary "Decompose input into parts via LLM, then process each part.")
(define-meta task-shapes '(Decompose Composite))
(define-meta data-shapes '(Singular ChunkedSingular))
(define-meta output-shape 'many)

(define-meta trigger '())
(define-meta reject '())

(define-meta slots
  '((context_id              (type string) (pattern "^ctx_") (required #t))
    (decompose_instruction   (type string) (min-length 10) (required #t))
    (map_instruction         (type string) (min-length 10) (required #t))
    (decompose_model         (type string) (default "quality_text_model"))
    (map_model               (type string) (default "fast_text_model"))
    (max_concurrent          (type integer) (min 1) (max 50) (default 20))))

(define-meta structural-profile
  '((expected-calls "1 + K (where K = decomposed parts)")
    (critical-path  "2")
    (max-concurrency-slot max_concurrent)
    (recursive-depth 0)
    (uses-python-bridge #t)
    (uses-multimodal #f)))

(define-meta verification-rules
  '(context_id_exists
    expected_calls_within_policy
    max_concurrency_within_policy
    only_primitive_bindings))

(define-meta streamable #t)
(define-meta cacheable #t)
(define-meta uses-llm-generated-code #f)

(define data (__context-ref "{{context_id}}" "$"))

;; Decompose into parts
(define parts_json
  (syntax-e
    (llm-query
      #:instruction "{{decompose_instruction}}"
      #:data data
      #:model "{{decompose_model}}"
      #:json #t)))

;; Parse the JSON array of parts
(py-set! "parts_json" parts_json)
(define parts (py-eval "import json; json.loads(parts_json)"))

;; Process each part
(define results
  (map-async
    (lambda (part)
      (llm-query-async
        #:instruction "{{map_instruction}}"
        #:data part
        #:model "{{map_model}}"))
    parts
    #:max-concurrent {{max_concurrent}}))

(finish results)
