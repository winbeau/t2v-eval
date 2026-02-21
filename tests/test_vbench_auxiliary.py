"""
Tests for scripts/vbench_runner/auxiliary.py — prompt processing and auxiliary info.

Covers:
  - normalize_prompt_text()
  - simplify_prompt_text()
  - tokenize_prompt_words()
  - extract_object_token()
  - infer_auxiliary_from_prompt() — all 6 dimension branches
  - build_auxiliary_prompt_lookup()
  - resolve_auxiliary_payload()
"""

import json
from pathlib import Path

import pytest

from scripts.vbench_runner.auxiliary import (
    build_auxiliary_prompt_lookup,
    extract_object_token,
    infer_auxiliary_from_prompt,
    normalize_prompt_text,
    resolve_auxiliary_payload,
    simplify_prompt_text,
    tokenize_prompt_words,
)


# ---------------------------------------------------------------------------
# normalize_prompt_text
# ---------------------------------------------------------------------------
class TestNormalizePromptText:
    def test_lowercase_and_strip(self):
        assert normalize_prompt_text("  A Cat Walking  ") == "a cat walking"

    def test_collapse_whitespace(self):
        assert normalize_prompt_text("a   cat\t\nwalking") == "a cat walking"

    def test_empty_string(self):
        assert normalize_prompt_text("") == ""

    def test_none_input(self):
        assert normalize_prompt_text(None) == ""


# ---------------------------------------------------------------------------
# simplify_prompt_text
# ---------------------------------------------------------------------------
class TestSimplifyPromptText:
    def test_removes_punctuation(self):
        result = simplify_prompt_text("A cat, walking! On the street.")
        assert "," not in result
        assert "!" not in result
        assert "." not in result

    def test_lowercase(self):
        result = simplify_prompt_text("HELLO WORLD")
        assert result == "hello world"

    def test_preserves_alphanumeric(self):
        result = simplify_prompt_text("test123")
        assert "test123" in result


# ---------------------------------------------------------------------------
# tokenize_prompt_words
# ---------------------------------------------------------------------------
class TestTokenizePromptWords:
    def test_basic(self):
        result = tokenize_prompt_words("A cat walking on the street")
        assert result == ["a", "cat", "walking", "on", "the", "street"]

    def test_with_punctuation(self):
        result = tokenize_prompt_words("Hello, world!")
        assert result == ["hello", "world"]

    def test_empty(self):
        assert tokenize_prompt_words("") == []

    def test_none(self):
        assert tokenize_prompt_words(None) == []

    def test_numbers_excluded(self):
        """Only alphabetic tokens extracted."""
        result = tokenize_prompt_words("video 42 shows a cat")
        assert "42" not in result
        assert "cat" in result


# ---------------------------------------------------------------------------
# extract_object_token
# ---------------------------------------------------------------------------
class TestExtractObjectToken:
    def test_basic_extraction(self):
        result = extract_object_token("a red cat walking on the street")
        # Stops words and color words filtered, returns last remaining token
        assert result in ("cat", "walking", "street")

    def test_default_when_empty(self):
        assert extract_object_token("") == "object"

    def test_custom_default(self):
        assert extract_object_token("", default="thing") == "thing"

    def test_all_stopwords(self):
        """When all words are stop/color words, return default."""
        assert extract_object_token("the of in on") == "object"

    def test_color_words_filtered(self):
        result = extract_object_token("a red blue green cat")
        assert result == "cat"


# ---------------------------------------------------------------------------
# infer_auxiliary_from_prompt — dimension-specific branches
# ---------------------------------------------------------------------------
class TestInferAuxiliaryFromPrompt:
    # --- appearance_style ---
    def test_appearance_style_explicit(self):
        result = infer_auxiliary_from_prompt(
            "appearance_style",
            "A cat in the style of impressionism",
        )
        assert result is not None
        assert "appearance_style" in result
        assert "impressionism" in result["appearance_style"]

    def test_appearance_style_suffix(self):
        result = infer_auxiliary_from_prompt(
            "appearance_style",
            "A cat in watercolor style",
        )
        assert result is not None
        assert "watercolor style" in result["appearance_style"]

    def test_appearance_style_fallback(self):
        result = infer_auxiliary_from_prompt(
            "appearance_style",
            "A cat walking",
        )
        assert result is not None
        assert result["appearance_style"] == "realistic style"

    # --- scene ---
    def test_scene_extraction(self):
        result = infer_auxiliary_from_prompt(
            "scene",
            "A beach with waves crashing",
        )
        assert result is not None
        assert "scene" in result
        assert "scene" in result["scene"]

    # --- object_class ---
    def test_object_class(self):
        result = infer_auxiliary_from_prompt(
            "object_class",
            "A cat sitting on a sofa",
        )
        assert result is not None
        assert "object" in result

    # --- multiple_objects with "and" ---
    def test_multiple_objects_with_and(self):
        result = infer_auxiliary_from_prompt(
            "multiple_objects",
            "A cat and a dog playing together",
        )
        assert result is not None
        assert "object" in result
        assert "and" in result["object"]

    def test_multiple_objects_without_and(self):
        result = infer_auxiliary_from_prompt(
            "multiple_objects",
            "A cat sitting near a dog",
        )
        assert result is not None
        assert "object" in result

    # --- spatial_relationship ---
    def test_spatial_with_relation(self):
        result = infer_auxiliary_from_prompt(
            "spatial_relationship",
            "A cat on the left of a dog",
        )
        assert result is not None
        assert "spatial_relationship" in result
        sr = result["spatial_relationship"]
        assert "object_a" in sr
        assert "object_b" in sr
        assert sr["relationship"] == "on the left of"

    def test_spatial_without_relation(self):
        result = infer_auxiliary_from_prompt(
            "spatial_relationship",
            "A cat walking",
        )
        assert result is not None
        assert result["spatial_relationship"]["relationship"] == "on the left of"

    # --- color ---
    def test_color_found(self):
        result = infer_auxiliary_from_prompt("color", "A red car on the road")
        assert result is not None
        assert result["color"] == "red"

    def test_color_grey_normalized(self):
        result = infer_auxiliary_from_prompt("color", "A grey cat")
        assert result is not None
        assert result["color"] == "gray"  # grey → gray

    def test_color_fallback(self):
        result = infer_auxiliary_from_prompt("color", "A car on the road")
        assert result is not None
        assert result["color"] == "red"  # default

    # --- unknown dimension ---
    def test_unknown_dimension(self):
        result = infer_auxiliary_from_prompt("nonexistent", "some prompt")
        assert result is None


# ---------------------------------------------------------------------------
# build_auxiliary_prompt_lookup
# ---------------------------------------------------------------------------
class TestBuildAuxiliaryPromptLookup:
    def test_missing_file(self, tmp_path):
        exact, simple = build_auxiliary_prompt_lookup(tmp_path / "missing.json")
        assert exact == {}
        assert simple == {}

    def test_valid_file(self, tmp_path):
        data = [
            {
                "prompt_en": "A red car on the road",
                "auxiliary_info": {
                    "color": {"color": "red"},
                    "object_class": {"object": "car"},
                },
            }
        ]
        json_path = tmp_path / "full_info.json"
        json_path.write_text(json.dumps(data))

        exact, simple = build_auxiliary_prompt_lookup(json_path)
        assert "color" in exact
        assert len(exact["color"]) == 1
        assert "object_class" in exact

    def test_skips_invalid_items(self, tmp_path):
        data = [
            "not_a_dict",
            {"prompt_en": "", "auxiliary_info": {"color": {"color": "red"}}},
            {"prompt_en": "valid", "auxiliary_info": "not_a_dict"},
        ]
        json_path = tmp_path / "full_info.json"
        json_path.write_text(json.dumps(data))

        exact, simple = build_auxiliary_prompt_lookup(json_path)
        # All items should be skipped or have empty prompts
        total = sum(len(v) for v in exact.values())
        assert total == 0


# ---------------------------------------------------------------------------
# resolve_auxiliary_payload
# ---------------------------------------------------------------------------
class TestResolveAuxiliaryPayload:
    def test_exact_match(self):
        exact = {"color": {"a red car on the road": {"color": "red"}}}
        payload, source = resolve_auxiliary_payload(
            "color", "A Red Car On The Road", exact, {}
        )
        assert payload == {"color": "red"}
        assert source == "exact"

    def test_simplified_match(self):
        simple = {"color": {"a red car on the road": {"color": "red"}}}
        payload, source = resolve_auxiliary_payload(
            "color", "A red car, on the road!", {}, simple
        )
        assert payload == {"color": "red"}
        assert source == "simplified"

    def test_heuristic_fallback(self):
        payload, source = resolve_auxiliary_payload(
            "color", "A blue sky over the mountains", {}, {}
        )
        assert payload is not None
        assert source == "heuristic"
        assert payload["color"] == "blue"

    def test_missing_dimension(self):
        payload, source = resolve_auxiliary_payload(
            "nonexistent_dim", "some prompt", {}, {}
        )
        assert payload is None
        assert source == "missing"
