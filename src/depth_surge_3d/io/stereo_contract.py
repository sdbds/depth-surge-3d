"""Leaf-level constants and errors shared by stereo control-plane modules."""

from __future__ import annotations


STEREO_CONTROL_FRAME_CAP = 4_194_304


class StereoControlError(RuntimeError):
    """Base error for the bounded replayable control plane."""


class StereoControlBudgetError(StereoControlError):
    """The frame or resident control-plane bound cannot be met."""
