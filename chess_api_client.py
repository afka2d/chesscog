#!/usr/bin/env python3
"""
Chess Position Scanner API Client
Simple client library for integrating with the production API
"""

import requests
import json
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class ChessPositionScannerClient:
    """
    Client for the Chess Position Scanner API
    """
    
    def __init__(self, api_url: str = "https://api.chesspositionscanner.store"):
        """
        Initialize the client
        
        Args:
            api_url: Base URL of the API (default: production URL)
        """
        self.api_url = api_url.rstrip('/')
        self.health_url = f"{self.api_url}/health"
        self.recognize_url = f"{self.api_url}/recognize_chess_position_with_corners"
        
    def health_check(self) -> Dict:
        """
        Check if the API is healthy
        
        Returns:
            Dict with health status and model information
        """
        try:
            response = requests.get(self.health_url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}
    
    def recognize_position(
        self, 
        image_path: str, 
        corners: List[List[int]], 
        turn: str = "white"
    ) -> Dict:
        """
        Recognize chess position from image
        
        Args:
            image_path: Path to the chess board image
            corners: List of 4 corner coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            turn: Current player turn ("white" or "black")
            
        Returns:
            Dict with FEN notation, pieces, and occupancy data
        """
        try:
            with open(image_path, 'rb') as f:
                files = {'image': f}
                data = {
                    'corners': json.dumps(corners),
                    'turn': turn
                }
                
                response = requests.post(
                    self.recognize_url, 
                    files=files, 
                    data=data, 
                    timeout=30
                )
                response.raise_for_status()
                return response.json()
                
        except Exception as e:
            logger.error(f"Recognition failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "fen": "8/8/8/8/8/8/8/8 w - - 0 1",
                "pieces": [None] * 64,
                "occupancy": [False] * 64
            }
    
    def recognize_position_from_bytes(
        self, 
        image_bytes: bytes, 
        corners: List[List[int]], 
        turn: str = "white"
    ) -> Dict:
        """
        Recognize chess position from image bytes
        
        Args:
            image_bytes: Image data as bytes
            corners: List of 4 corner coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            turn: Current player turn ("white" or "black")
            
        Returns:
            Dict with FEN notation, pieces, and occupancy data
        """
        try:
            files = {'image': ('image.jpg', image_bytes, 'image/jpeg')}
            data = {
                'corners': json.dumps(corners),
                'turn': turn
            }
            
            response = requests.post(
                self.recognize_url, 
                files=files, 
                data=data, 
                timeout=30
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Recognition failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "fen": "8/8/8/8/8/8/8/8 w - - 0 1",
                "pieces": [None] * 64,
                "occupancy": [False] * 64
            }
    
    def is_healthy(self) -> bool:
        """
        Check if the API is healthy (simple boolean check)
        
        Returns:
            True if API is healthy, False otherwise
        """
        health = self.health_check()
        return health.get("status") == "healthy"
    
    def get_model_status(self) -> Dict:
        """
        Get detailed model status
        
        Returns:
            Dict with model loading status
        """
        health = self.health_check()
        return health.get("models", {})

# Convenience functions for easy integration
def recognize_chess_position(
    image_path: str, 
    corners: List[List[int]], 
    turn: str = "white",
    api_url: str = "https://api.chesspositionscanner.store"
) -> Dict:
    """
    Convenience function to recognize chess position
    
    Args:
        image_path: Path to the chess board image
        corners: List of 4 corner coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        turn: Current player turn ("white" or "black")
        api_url: API URL (default: production URL)
        
    Returns:
        Dict with FEN notation, pieces, and occupancy data
    """
    client = ChessPositionScannerClient(api_url)
    return client.recognize_position(image_path, corners, turn)

def check_api_health(api_url: str = "https://api.chesspositionscanner.store") -> bool:
    """
    Convenience function to check API health
    
    Args:
        api_url: API URL (default: production URL)
        
    Returns:
        True if API is healthy, False otherwise
    """
    client = ChessPositionScannerClient(api_url)
    return client.is_healthy()

# Example usage
if __name__ == "__main__":
    # Example usage
    client = ChessPositionScannerClient()
    
    # Check health
    if client.is_healthy():
        print("✅ API is healthy")
        
        # Example recognition (you would use real image and corners)
        corners = [[100, 100], [1100, 100], [1100, 1100], [100, 1100]]
        result = client.recognize_position("test_image.jpg", corners)
        
        if result.get("success"):
            print(f"FEN: {result['fen']}")
            print(f"Pieces found: {sum(1 for p in result['pieces'] if p is not None)}")
        else:
            print(f"Recognition failed: {result.get('error')}")
    else:
        print("❌ API is not healthy")
