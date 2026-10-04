"""Candidate-independent PDF forensics: inventory of every content surface."""

from .inspect import DocumentInventory, ImageLayer, PageInventory, inspect_pdf, revision_prefixes

__all__ = ["DocumentInventory", "ImageLayer", "PageInventory", "inspect_pdf", "revision_prefixes"]
