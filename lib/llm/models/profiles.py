"""Compatibility wrapper around the unified model catalog."""

from __future__ import annotations

from .catalog import ModelProfile, default_model_profiles, resolve_model_profile

__all__ = ["ModelProfile", "default_model_profiles", "resolve_model_profile"]
