(define-meta name "direct_call")
(define-meta version "1.0.0")
(define-meta summary "Single LLM query; returns result directly.")
(define-meta task-shapes '(Direct))
(define-meta data-shapes '(Singular Unknown))
(define-meta output-shape 'one)

(define-meta trigger '())
(define-meta reject '())

(define-meta slots
  '((context_id   (type string) (pattern "^ctx_") (required #t))
    (instruction  (type string) (min-length 1) (required #t))
    (model        (type string) (default "fast_text_model"))
    (temperature  (type number) (nullable #t) (default #f))
    (max_tokens   (type integer) (nullable #t) (default #f))
    (json_mode    (type boolean) (default #f))))

(define-meta structural-profile
  '((expected-calls "1")
    (critical-path  "1")
    (max-concurrency-slot #f)
    (recursive-depth 0)
    (uses-python-bridge #f)
    (uses-multimodal #f)))

(define-meta verification-rules
  '(context_id_exists
    only_primitive_bindings))

(define-meta streamable #f)
(define-meta cacheable #t)
(define-meta uses-llm-generated-code #f)

(define data (__context-ref "{{context_id}}" "$"))

(define result
  (syntax-e
    (llm-query
      #:instruction "{{instruction}}"
      #:data data
      #:model "{{model}}"
      #:temperature {{temperature}}
      #:max-tokens {{max_tokens}}
      #:json {{json_mode}})))

(finish result)
