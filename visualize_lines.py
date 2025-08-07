import cv2
import numpy as np
import yaml

def load_config():
    with open("config/corner_detection.yaml", "r") as f:
        return yaml.safe_load(f)

def visualize_detection(image_path):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    # Load config
    cfg = load_config()
    
    # Create copies for visualization
    lines_img = img.copy()
    edges_img = img.copy()
    
    # Edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(
        gray,
        cfg['EDGE_DETECTION']['LOW_THRESHOLD'],
        cfg['EDGE_DETECTION']['HIGH_THRESHOLD'],
        apertureSize=cfg['EDGE_DETECTION']['APERTURE']
    )
    
    # Convert edges to BGR for visualization
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_img = cv2.addWeighted(edges_img, 0.7, edges_bgr, 0.3, 0)
    
    # Line detection
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=cfg['LINE_DETECTION']['THRESHOLD'],
        minLineLength=cfg['LINE_DETECTION']['MIN_LINE_LENGTH'],
        maxLineGap=cfg['LINE_DETECTION']['MAX_LINE_GAP']
    )
    
    if lines is not None:
        lines = lines.reshape(-1, 2, 2)
        print(f"Number of lines detected: {len(lines)}")
        
        # Draw all detected lines
        for line in lines:
            x1, y1 = line[0]
            x2, y2 = line[1]
            cv2.line(lines_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
    
    # Save visualizations
    cv2.imwrite('edges.jpg', edges_img)
    cv2.imwrite('lines.jpg', lines_img)
    print("Visualizations saved as 'edges.jpg' and 'lines.jpg'")

if __name__ == "__main__":
    visualize_detection("app_processed.jpeg") 