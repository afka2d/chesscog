#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def fen_to_board(fen):
    """Convert FEN to 2D board representation"""
    # Split FEN into parts
    parts = fen.split()
    board_fen = parts[0]
    
    # Create empty board
    board = [['' for _ in range(8)] for _ in range(8)]
    
    # Parse board part of FEN
    rows = board_fen.split('/')
    for row_idx, row in enumerate(rows):
        col_idx = 0
        for char in row:
            if char.isdigit():
                # Empty squares
                col_idx += int(char)
            else:
                # Piece
                board[row_idx][col_idx] = char
                col_idx += 1
    
    return board

def visualize_chess_board(fen, save_path=None):
    """Create a visual representation of the chess board"""
    board = fen_to_board(fen)
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Define piece symbols (Unicode chess symbols)
    piece_symbols = {
        'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',  # White
        'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟'   # Black
    }
    
    # Draw the board
    for row in range(8):
        for col in range(8):
            # Alternate colors
            color = '#f0d9b5' if (row + col) % 2 == 0 else '#b58863'
            
            # Draw square
            rect = patches.Rectangle((col, 7-row), 1, 1, 
                                   linewidth=1, edgecolor='black', 
                                   facecolor=color)
            ax.add_patch(rect)
            
            # Add piece if present
            piece = board[row][col]
            if piece:
                symbol = piece_symbols.get(piece, piece)
                ax.text(col + 0.5, 7-row + 0.5, symbol, 
                       fontsize=24, ha='center', va='center',
                       weight='bold')
    
    # Set up the plot
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.set_xticks(range(9))
    ax.set_yticks(range(9))
    ax.set_xticklabels(['', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
    ax.set_yticklabels(['8', '7', '6', '5', '4', '3', '2', '1', ''])
    ax.grid(True, alpha=0.3)
    
    # Add title with FEN
    plt.title(f'Chess Position from API\nFEN: {fen}', fontsize=12, pad=20)
    
    # Remove ticks
    ax.tick_params(axis='both', which='both', length=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Board visualization saved to: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    # FEN from the API response (IMG_6904)
    fen = "2bq4/1p3p2/8/8/8/8/2PPPP2/1NBQK1N1 w - - 0 1"
    
    # Create visualization
    visualize_chess_board(fen, save_path="detected_chess_position.png")
