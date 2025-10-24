"""Test platform bias calibration functionality."""

import pytest
from src.insighthub.core.scoring import (
    apply_platform_bias, 
    set_bias_enabled, 
    set_shrinkage_k, 
    _shrink_factor,
    PLATFORM_BIAS
)


class TestPlatformBiasCalibration:
    """Test platform bias calibration functions."""
    
    def test_shrink_factor_calculation(self):
        """Test shrinkage factor calculation."""
        # Test basic cases
        assert _shrink_factor(0) == 0.0
        assert _shrink_factor(1) == 1.0 / (1.0 + 40)  # default k=40
        assert _shrink_factor(40) == 40.0 / (40.0 + 40)  # 0.5
        
        # Test with custom k
        assert _shrink_factor(10, k=10) == 10.0 / (10.0 + 10)  # 0.5
        assert _shrink_factor(20, k=5) == 20.0 / (20.0 + 5)  # 0.8
    
    def test_platform_bias_application(self):
        """Test platform bias application."""
        # Test with bias enabled
        set_bias_enabled(True)
        
        # Test Reddit (negative bias)
        reddit_score = apply_platform_bias(3.0, "reddit", 50)
        assert reddit_score < 3.0  # Should be lower due to negative bias
        
        # Test YouTube (positive bias)
        youtube_score = apply_platform_bias(3.0, "youtube", 50)
        assert youtube_score > 3.0  # Should be higher due to positive bias
        
        # Test Yelp (neutral bias)
        yelp_score = apply_platform_bias(3.0, "yelp", 50)
        assert yelp_score == 3.0  # Should be unchanged
        
        # Test unknown platform
        unknown_score = apply_platform_bias(3.0, "unknown", 50)
        assert unknown_score == 3.0  # Should be unchanged
    
    def test_bias_disabled(self):
        """Test that bias is not applied when disabled."""
        set_bias_enabled(False)
        
        # All scores should remain unchanged
        reddit_score = apply_platform_bias(3.0, "reddit", 50)
        youtube_score = apply_platform_bias(3.0, "youtube", 50)
        
        assert reddit_score == 3.0
        assert youtube_score == 3.0
    
    def test_score_clamping(self):
        """Test that scores are properly clamped to [1.0, 5.0]."""
        set_bias_enabled(True)
        
        # Test extreme cases that would go out of bounds
        very_low_score = apply_platform_bias(1.0, "reddit", 1000)  # Large negative bias
        very_high_score = apply_platform_bias(5.0, "youtube", 1000)  # Large positive bias
        
        assert very_low_score >= 1.0
        assert very_high_score <= 5.0
    
    def test_shrinkage_parameter_settings(self):
        """Test shrinkage parameter settings."""
        # Test setting different k values
        set_shrinkage_k(20)
        assert _shrink_factor(20) == 20.0 / (20.0 + 20)  # 0.5
        
        set_shrinkage_k(100)
        assert _shrink_factor(20) == 20.0 / (20.0 + 100)  # 0.167
        
        # Test minimum k value
        set_shrinkage_k(0)  # Should be clamped to 1
        assert _shrink_factor(1) == 1.0 / (1.0 + 1)  # 0.5
    
    def test_platform_bias_values(self):
        """Test that platform bias values are as expected."""
        assert PLATFORM_BIAS["reddit"] == -0.30
        assert PLATFORM_BIAS["youtube"] == 0.20
        assert PLATFORM_BIAS["yelp"] == 0.00
        assert PLATFORM_BIAS["google"] == 0.10
        assert PLATFORM_BIAS["xiaohongshu"] == 0.30


if __name__ == "__main__":
    pytest.main([__file__])
