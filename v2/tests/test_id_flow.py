"""Tests for ID generation and validation."""

import pytest

from rlm_scheme.ids import (
    ALL_PREFIXES,
    ART_PREFIX,
    CACHE_PREFIX,
    CALL_PREFIX,
    CTX_PREFIX,
    DRY_PREFIX,
    EXEC_PREFIX,
    PLAN_PREFIX,
    VER_PREFIX,
    generate_id,
    validate_id,
)


class TestGenerateId:
    def test_all_prefixes_produce_valid_ids(self):
        for prefix in ALL_PREFIXES:
            id_ = generate_id(prefix)
            assert id_.startswith(prefix)
            assert validate_id(id_, prefix)

    def test_ids_are_unique(self):
        ids = {generate_id(CTX_PREFIX) for _ in range(100)}
        assert len(ids) == 100

    def test_id_format(self):
        id_ = generate_id(CTX_PREFIX)
        # prefix + 16 hex chars
        assert len(id_) == len(CTX_PREFIX) + 16
        hex_part = id_[len(CTX_PREFIX):]
        assert all(c in "0123456789abcdef" for c in hex_part)

    def test_unknown_prefix_raises(self):
        with pytest.raises(ValueError, match="Unknown prefix"):
            generate_id("bad_")


class TestValidateId:
    def test_valid_ids(self):
        assert validate_id("ctx_0123456789abcdef", CTX_PREFIX)
        assert validate_id("plan_fedcba9876543210", PLAN_PREFIX)
        assert validate_id("art_aaaaaaaaaaaaaaaa", ART_PREFIX)

    def test_wrong_prefix(self):
        id_ = generate_id(CTX_PREFIX)
        assert not validate_id(id_, PLAN_PREFIX)

    def test_too_short(self):
        assert not validate_id("ctx_0123", CTX_PREFIX)

    def test_too_long(self):
        assert not validate_id("ctx_0123456789abcdef0", CTX_PREFIX)

    def test_uppercase_hex_rejected(self):
        assert not validate_id("ctx_0123456789ABCDEF", CTX_PREFIX)

    def test_non_string(self):
        assert not validate_id(12345, CTX_PREFIX)  # type: ignore[arg-type]

    def test_empty_string(self):
        assert not validate_id("", CTX_PREFIX)


class TestIdPipeline:
    """Simulate the context -> plan -> execution ID flow."""

    def test_full_flow(self):
        ctx_id = generate_id(CTX_PREFIX)
        plan_id = generate_id(PLAN_PREFIX)
        art_id = generate_id(ART_PREFIX)
        dry_id = generate_id(DRY_PREFIX)
        ver_id = generate_id(VER_PREFIX)
        exec_id = generate_id(EXEC_PREFIX)
        call_id = generate_id(CALL_PREFIX)
        cache_id = generate_id(CACHE_PREFIX)

        assert validate_id(ctx_id, CTX_PREFIX)
        assert validate_id(plan_id, PLAN_PREFIX)
        assert validate_id(art_id, ART_PREFIX)
        assert validate_id(dry_id, DRY_PREFIX)
        assert validate_id(ver_id, VER_PREFIX)
        assert validate_id(exec_id, EXEC_PREFIX)
        assert validate_id(call_id, CALL_PREFIX)
        assert validate_id(cache_id, CACHE_PREFIX)

        # All IDs are distinct
        all_ids = {ctx_id, plan_id, art_id, dry_id, ver_id, exec_id, call_id, cache_id}
        assert len(all_ids) == 8
