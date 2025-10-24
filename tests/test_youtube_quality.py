"""Test YouTube client quality filtering and deduplication."""

import pytest
from unittest.mock import Mock, patch
from src.insighthub.services.youtube_client import YouTubeService


class TestYouTubeClientQuality:
    """Test YouTube client quality filtering and deduplication."""
    
    def setup_method(self):
        """Set up test instance."""
        self.service = YouTubeService()
    
    def test_quality_filtering(self):
        """Test comment quality filtering."""
        # Test valid comments
        assert self.service._quality_ok("This is a great product! I love it.")
        assert self.service._quality_ok("Very helpful review with detailed information.")
        
        # Test invalid comments
        assert not self.service._quality_ok("")  # Empty
        assert not self.service._quality_ok("Short")  # Too short
        assert not self.service._quality_ok("😀😀😀😀😀")  # Pure emoji
        assert not self.service._quality_ok("!!!!!")  # Pure punctuation
        assert not self.service._quality_ok("   ")  # Whitespace only
    
    def test_per_video_limit_calculation(self):
        """Test per-video limit calculation."""
        # Test basic cases
        assert self.service._per_video_limit(50) == max(5, min(50, 50 // 8))  # 6
        assert self.service._per_video_limit(100) == max(5, min(50, 100 // 8))  # 12
        assert self.service._per_video_limit(500) == max(5, min(50, 500 // 8))  # 50 (capped)
        assert self.service._per_video_limit(10) == max(5, min(50, 10 // 8))  # 5 (minimum)
    
    def test_video_relevance_filtering(self):
        """Test video relevance filtering."""
        # Test relevant videos
        relevant_video = {
            "title": "Best iPhone 15 Pro Review",
            "description": "In this video I review the iPhone 15 Pro camera features and performance."
        }
        assert self.service._is_relevant_video(relevant_video, "iPhone 15 Pro")
        assert self.service._is_relevant_video(relevant_video, "iPhone camera")
        
        # Test irrelevant videos
        irrelevant_video = {
            "title": "Random Gaming Video",
            "description": "Playing some random games today."
        }
        assert not self.service._is_relevant_video(irrelevant_video, "iPhone 15 Pro")
        assert not self.service._is_relevant_video(irrelevant_video, "camera review")
    
    def test_comment_deduplication_logic(self):
        """Test comment deduplication logic."""
        import hashlib
        
        # Test deduplication key generation
        comment1 = {"author": "User1", "text": "Great product!"}
        comment2 = {"author": "User1", "text": "Great product!"}  # Same content
        comment3 = {"author": "User2", "text": "Great product!"}  # Different author
        
        key1 = f"{comment1['author']}:{hashlib.md5(comment1['text'].strip().lower().encode()).hexdigest()}"
        key2 = f"{comment2['author']}:{hashlib.md5(comment2['text'].strip().lower().encode()).hexdigest()}"
        key3 = f"{comment3['author']}:{hashlib.md5(comment3['text'].strip().lower().encode()).hexdigest()}"
        
        # Same author + same content should have same key
        assert key1 == key2
        # Different author + same content should have different key
        assert key1 != key3
    
    def test_mock_data_generation(self):
        """Test mock data generation for testing."""
        # Test mock videos
        mock_videos = self.service._get_mock_videos("iPhone 15", 3)
        assert len(mock_videos) == 3
        assert all("iPhone 15" in video["title"] for video in mock_videos)
        assert all(video["video_id"].startswith("mock_video_") for video in mock_videos)
        
        # Test mock comments
        mock_comments = self.service._get_mock_comments("test_video", 5)
        assert len(mock_comments) == 5
        assert all(comment["comment_id"].startswith("mock_comment_") for comment in mock_comments)
        assert all("test_video" in comment["text"] for comment in mock_comments)
    
    @patch('src.insighthub.services.youtube_client.build')
    def test_youtube_initialization_with_key(self, mock_build):
        """Test YouTube service initialization with API key."""
        # Mock settings with API key
        with patch('src.insighthub.services.youtube_client.settings') as mock_settings:
            mock_settings.youtube_api_key = "test_key"
            mock_build.return_value = Mock()
            
            service = YouTubeService()
            assert service.youtube is not None
            mock_build.assert_called_once_with("youtube", "v3", developerKey="test_key")
    
    @patch('src.insighthub.services.youtube_client.build')
    def test_youtube_initialization_without_key(self, mock_build):
        """Test YouTube service initialization without API key."""
        # Mock settings without API key
        with patch('src.insighthub.services.youtube_client.settings') as mock_settings:
            mock_settings.youtube_api_key = ""
            
            service = YouTubeService()
            assert service.youtube is None
            mock_build.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__])
