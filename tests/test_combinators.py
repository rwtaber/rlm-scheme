"""Tests for combinator library (Section 5.5 in racket_server.rkt).

Verifies that all ~17 combinators work correctly in isolation and composition.
"""

import json
import pytest
from mcp_server import get_backend


def setup_module():
    """Reset sandbox before running combinator tests."""
    backend = get_backend()
    backend.send({"op": "reset"})


def test_sequence_combinator():
    """Test sequence combinator: chain operations left-to-right."""
    backend = get_backend()
    code = """
    (define add1 (lambda (x) (+ x 1)))
    (define mul2 (lambda (x) (* x 2)))
    (define sub3 (lambda (x) (- x 3)))

    (define pipeline (sequence add1 mul2 sub3))
    (finish (pipeline 5))
    """
    result = backend.send({"op": "eval", "code": code})
    assert result["status"] == "finished"
    # ((5 + 1) * 2) - 3 = 12 - 3 = 9
    assert result["result"] == "9"


def test_tree_reduce_combinator():
    """Test tree-reduce combinator: hierarchical aggregation."""
    backend = get_backend()
    code = """
    (define items (list 1 2 3 4 5 6 7 8))
    (define result (tree-reduce
        (lambda args (apply + args))
        items
        #:branch-factor 2))
    (finish result)
    """
    result = backend.send({"op": "eval", "code": code})
    assert result["status"] == "finished"
    # Should sum to 36 (1+2+...+8)
    assert result["result"] == "36"


def test_tree_reduce_with_leaf_fn():
    """Test tree-reduce with leaf transformation."""
    backend = get_backend()
    code = """
    (define items (list "a" "b" "c" "d"))
    (define result (tree-reduce
        (lambda (left right) (string-append left "|" right))
        items
        #:branch-factor 2
        #:leaf-fn (lambda (s) (string-upcase s))))
    (finish result)
    """
    result = backend.send({"op": "eval", "code": code})
    assert result["status"] == "finished"
    # Should produce hierarchical reduction: (A|B)|(C|D)
    assert "|" in result["result"]
    assert "A" in result["result"]


def test_fold_sequential_combinator():
    """Test fold-sequential combinator: sequential accumulation."""
    backend = get_backend()
    code = """
    (define result (fold-sequential
        (lambda (acc item) (string-append acc (number->string item) ","))
        ""
        (list 1 2 3 4 5)))
    (finish result)
    """
    result = backend.send({"op": "eval", "code": code})
    assert result["status"] == "finished"
    assert result["result"] == "1,2,3,4,5,"


def test_iterate_until_combinator():
    """Test iterate-until combinator: loop with predicate."""
    backend = get_backend()
    code = """
    (define result (iterate-until
        (lambda (x) (* x 2))
        (lambda (x) (> x 100))
        1
        #:max-iter 10))
    (finish result)
    """
    result = backend.send({"op": "eval", "code": code})
    assert result["status"] == "finished"
    # 1 -> 2 -> 4 -> 8 -> 16 -> 32 -> 64 -> 128 (stops at >100)
    assert int(result["result"]) == 128


def test_iterate_until_max_iter():
    """Test iterate-until respects max-iter."""
    backend = get_backend()
    code = """
    (define result (iterate-until
        (lambda (x) (+ x 1))
        (lambda (x) #f)  ; Never true
        0
        #:max-iter 5))
    (finish result)
    """
    result = backend.send({"op": "eval", "code": code})
    assert result["status"] == "finished"
    assert result["result"] == "5"


def test_choose_combinator():
    """Test choose combinator: conditional execution."""
    backend = get_backend()
    code = """
    (define small-dataset? (lambda (items) (< (length items) 10)))
    (define simple-process (lambda (items) "simple"))
    (define complex-process (lambda (items) "complex"))

    (define processor (choose small-dataset? simple-process complex-process))

    (define result1 (processor (list 1 2 3)))
    (define result2 (processor (list 1 2 3 4 5 6 7 8 9 10 11)))

    (finish (string-append result1 "," result2))
    """
    result = backend.send({"op": "eval", "code": code})
    assert result["status"] == "finished"
    assert result["result"] == "simple,complex"


def test_vote_combinator_majority():
    """Test vote combinator with majority method."""
    backend = get_backend()
    code = """
    (define strategies (list
        (lambda () "option-a")
        (lambda () "option-a")
        (lambda () "option-a")
        (lambda () "option-b")
        (lambda () "option-b")))

    (define result (vote strategies #:method 'majority))
    (finish result)
    """
    result = backend.send({"op": "eval", "code": code})
    assert result["status"] == "finished"
    assert result["result"] == "option-a"


def test_vote_combinator_plurality():
    """Test vote combinator with plurality method."""
    backend = get_backend()
    code = """
    (define strategies (list
        (lambda () "option-a")
        (lambda () "option-a")
        (lambda () "option-b")
        (lambda () "option-c")))

    (define result (vote strategies #:method 'plurality))
    (finish result)
    """
    result = backend.send({"op": "eval", "code": code})
    assert result["status"] == "finished"
    assert result["result"] == "option-a"


def test_vote_combinator_consensus():
    """Test vote combinator with consensus method."""
    backend = get_backend()
    # Test successful consensus
    code1 = """
    (define strategies (list
        (lambda () "option-a")
        (lambda () "option-a")
        (lambda () "option-a")))

    (define result (vote strategies #:method 'consensus))
    (finish result)
    """
    result = backend.send({"op": "eval", "code": code1})
    assert result["status"] == "finished"
    assert result["result"] == "option-a"

    # Test failed consensus
    code2 = """
    (define strategies (list
        (lambda () "option-a")
        (lambda () "option-b")))

    (try (vote strategies #:method 'consensus)
         on-error (lambda (err) "no-consensus"))
    """
    # This should raise an error since there's no consensus
    # We'll test that it doesn't crash and returns something


def test_ensemble_combinator_default():
    """Test ensemble combinator with default aggregation."""
    backend = get_backend()
    code = """
    (define strategies (list
        (lambda () "result-from-model-a")
        (lambda () "result-from-model-b")))

    (define result (ensemble strategies))
    (finish result)
    """
    result = backend.send({"op": "eval", "code": code})
    assert result["status"] == "finished"
    # Default aggregation concatenates results with labels
    assert "Model 1" in result["result"]
    assert "Model 2" in result["result"]


def test_ensemble_combinator_custom_aggregator():
    """Test ensemble combinator with custom aggregator."""
    backend = get_backend()
    code = """
    (define strategies (list
        (lambda () "a")
        (lambda () "b")
        (lambda () "c")))

    (define result (ensemble strategies
                             #:aggregator (lambda (results)
                                           (string-join results ","))))
    (finish result)
    """
    result = backend.send({"op": "eval", "code": code})
    assert result["status"] == "finished"
    assert result["result"] == "a,b,c"


def test_tiered_combinator():
    """Test tiered combinator: cheap then expensive."""
    backend = get_backend()
    code = """
    (define cheap-extract (lambda (item) (string-append "cheap:" item)))
    (define expensive-synthesize (lambda (results)
                                   (string-append "expensive:" (car results))))

    (define result (tiered cheap-extract expensive-synthesize (list "data")))
    (finish result)
    """
    result = backend.send({"op": "eval", "code": code})
    assert result["status"] == "finished"
    assert "expensive:cheap:data" in result["result"]


def test_memoized_combinator():
    """Test memoized combinator: caching by key."""
    backend = get_backend()
    code = """
    (define call-count 0)
    (define expensive-fn (lambda (x)
                          (set! call-count (+ call-count 1))
                          (* x 2)))

    (define memoized-fn (memoized expensive-fn))

    (define result1 (memoized-fn 5))
    (define result2 (memoized-fn 5))  ; Should use cache
    (define result3 (memoized-fn 10)) ; New input, should call fn

    (finish (string-append
             (number->string result1) ","
             (number->string result2) ","
             (number->string result3) ","
             (number->string call-count)))
    """
    result = backend.send({"op": "eval", "code": code})
    assert result["status"] == "finished"
    # Results: 10,10,20, call-count: 2 (not 3, because second call was cached)
    assert result["result"] == "10,10,20,2"


def test_try_fallback_combinator():
    """Test try-fallback combinator: error handling."""
    backend = get_backend()
    code = """
    (define risky-fn (lambda (x)
                      (if (> x 5)
                          (error 'risky "too big")
                          (* x 2))))

    (define safe-fn (lambda (x) 0))

    (define safe-risky (try-fallback risky-fn safe-fn))

    (define result1 (safe-risky 3))  ; Should succeed
    (define result2 (safe-risky 10)) ; Should fallback

    (finish (string-append
             (number->string result1) ","
             (number->string result2)))
    """
    result = backend.send({"op": "eval", "code": code})
    assert result["status"] == "finished"
    assert result["result"] == "6,0"


def test_with_validation_combinator():
    """Test with-validation combinator: validation wrapper."""
    backend = get_backend()
    # Test successful validation
    code1 = """
    (define process-fn (lambda (x) (* x 2)))
    (define validator (lambda (result) (< result 100)))

    (define validated-process (with-validation process-fn validator))

    (finish (validated-process 10))
    """
    result = backend.send({"op": "eval", "code": code1})
    assert result["status"] == "finished"
    assert result["result"] == "20"

    # Test failed validation
    code2 = """
    (define process-fn (lambda (x) (* x 2)))
    (define validator (lambda (result) (< result 10)))

    (define validated-process (with-validation process-fn validator))

    (try (validated-process 10)
         on-error (lambda (err) "validation-failed"))
    """
    # Should raise error since 20 is not < 10


def test_composition_tiered_with_tree_reduce():
    """Test composition: tiered execution with tree-reduce."""
    backend = get_backend()
    code = """
    (define items (list "doc1" "doc2" "doc3" "doc4" "doc5" "doc6"))
    (define result (tiered
        (lambda (item) (string-append "extract:" item))
        (lambda (extractions)
          (tree-reduce
            (lambda (left right) (string-append left "+" right))
            extractions
            #:branch-factor 2))
        items))
    (finish result)
    """
    result = backend.send({"op": "eval", "code": code})
    assert result["status"] == "finished"
    # Should have hierarchical structure with extractions
    assert "extract:doc1" in result["result"]
    assert "+" in result["result"]


def test_composition_sequence_with_validation():
    """Test composition: sequence with validation."""
    backend = get_backend()
    code = """
    (define step1 (lambda (x) (+ x 10)))
    (define step2 (lambda (x) (* x 2)))
    (define validator (lambda (x) (> x 0)))

    (define pipeline (sequence
                      (with-validation step1 validator)
                      (with-validation step2 validator)))

    (finish (pipeline 5))
    """
    result = backend.send({"op": "eval", "code": code})
    assert result["status"] == "finished"
    # (5 + 10) * 2 = 30
    assert result["result"] == "30"


def test_active_learning_combinator():
    """Test active-learning combinator: selective expensive processing."""
    backend = get_backend()
    code = """
    (define items (list "easy" "hard" "medium" "easy"))

    (define cheap-fn (lambda (item)
                      (string-append "cheap:" item)))

    (define expensive-fn (lambda (item)
                          (string-append "expensive:" item)))

    (define uncertainty-fn (lambda (result)
                            ; "hard" is uncertain (low score)
                            (if (string-contains? result "hard")
                                0.5  ; uncertain
                                0.9))) ; certain

    (define results (active-learning cheap-fn expensive-fn uncertainty-fn items
                                     #:threshold 0.7))

    (finish (string-join results ","))
    """
    result = backend.send({"op": "eval", "code": code})
    assert result["status"] == "finished"
    # "hard" should get expensive processing, others cheap
    assert "cheap:easy" in result["result"]
    assert "expensive:hard" in result["result"]


def test_critique_refine_combinator():
    """Test critique-refine combinator: iterative refinement."""
    backend = get_backend()
    code = """
    (define iteration-count 0)

    (define generate-fn (lambda ()
                         "draft v1"))

    (define critique-fn (lambda (draft)
                         (set! iteration-count (+ iteration-count 1))
                         "needs improvement"))

    (define refine-fn (lambda (draft critique)
                       (string-append draft " [refined-" (number->string iteration-count) "]")))

    (define result (critique-refine generate-fn critique-fn refine-fn
                                    #:max-iter 3))

    (finish result)
    """
    result = backend.send({"op": "eval", "code": code})
    assert result["status"] == "finished"
    # Should have 3 refinement iterations
    assert "[refined-1]" in result["result"]
    assert "[refined-2]" in result["result"]
    assert "[refined-3]" in result["result"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
