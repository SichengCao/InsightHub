"""Tests for scoring module."""

import pytest
from insighthub.analysis.scoring import compute_global


def test_compute_global_guardrail():
    """Test that guardrail works correctly."""
    # Create fake reviews with stars (pos/neu/neg ~ 34/32/34)
    reviews_with_stars = []
    
    # Add positive reviews (34%)
    for i in range(34):
        reviews_with_stars.append({'stars': 4.5, 'label': 'POSITIVE'})
    
    # Add neutral reviews (32%)
    for i in range(32):
        reviews_with_stars.append({'stars': 3.0, 'label': 'NEUTRAL'})
    
    # Add negative reviews (34%)
    for i in range(34):
        reviews_with_stars.append({'stars': 1.5, 'label': 'NEGATIVE'})
    
    avg_stars = compute_global(reviews_with_stars)
    
    # With guardrail, should be between 2.2 and 3.4
    assert 2.2 <= avg_stars <= 3.4, f"Average stars {avg_stars} outside expected range"


def test_compute_global_neutral_baseline():
    """Test that neutral reviews default to 3.0."""
    reviews_with_stars = [
        {'stars': None, 'label': 'NEUTRAL'},
        {'stars': 3.0, 'label': 'NEUTRAL'},
        {'stars': 3.0, 'label': 'NEUTRAL'}
    ]
    
    avg_stars = compute_global(reviews_with_stars)
    assert avg_stars == 3.0


def test_compute_global_empty():
    """Test empty input."""
    avg_stars = compute_global([])
    assert avg_stars == 3.0


def test_compute_global_clamping():
    """Test that stars are clamped to [1, 5]."""
    reviews_with_stars = [
        {'stars': 0.5, 'label': 'NEGATIVE'},  # Should be clamped to 1.0
        {'stars': 6.0, 'label': 'POSITIVE'},  # Should be clamped to 5.0
    ]
    
    avg_stars = compute_global(reviews_with_stars)
    assert 1.0 <= avg_stars <= 5.0
