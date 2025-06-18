"""Module to perform chessboard localization.

The entire board localization pipeline is implemented end-to-end in this module.
Use the :meth:`find_corners` to run it.

This module simultaneously acts as a script:

.. code-block:: console

    $ python -m chesscog.corner_detection.detect_corners --help
    usage: detect_corners.py [-h] [--config CONFIG] file
    
    Chessboard corner detector.
    
    positional arguments:
      file             URI of the input image file
    
    optional arguments:
      -h, --help       show this help message and exit
      --config CONFIG  path to the config file
"""

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import AgglomerativeClustering, DBSCAN
import cv2
import numpy as np
import typing
from recap import URI, CfgNode as CN

from chesscog.core import sort_corner_points
from chesscog.core.coordinates import from_homogenous_coordinates, to_homogenous_coordinates
from chesscog.core.exceptions import ChessboardNotLocatedException


def find_corners(cfg: CN, img: np.ndarray) -> typing.Tuple[np.ndarray, dict]:
    """Determine the four corner points of the chessboard in an image.

    Args:
        cfg (CN): the configuration object
        img (np.ndarray): the input image (as a numpy array)

    Raises:
        ChessboardNotLocatedException: if the chessboard could not be found

    Returns:
        typing.Tuple[np.ndarray, dict]: the pixel coordinates of the four corners and a dictionary of debug images
    """
    debug_images = {}
    
    # Debug prints for config
    print("\nAPI Debug - Config Structure:")
    print("RANSAC config:", cfg.RANSAC)
    print("RANSAC OFFSET_TOLERANCE:", getattr(cfg.RANSAC, "OFFSET_TOLERANCE", "Not found"))
    print("RANSAC BEST_SOLUTION_TOLERANCE:", getattr(cfg.RANSAC, "BEST_SOLUTION_TOLERANCE", "Not found"))
    print("\nRANSAC Config Details:")
    print("RANSAC keys:", dir(cfg.RANSAC))
    print("RANSAC dict:", cfg.RANSAC.__dict__)
    print("Full config:", cfg)
    
    img, img_scale = resize_image(cfg, img)
    debug_images['resized'] = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = _detect_edges(cfg.EDGE_DETECTION, gray)
    debug_images['edges'] = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Debug print for edges
    print("\nAPI Debug - Edge Detection:")
    print(f"Input image shape: {img.shape}")
    print(f"Grayscale image shape: {gray.shape}")
    print(f"Edges shape: {edges.shape}")
    print(f"Edges min/max values: {edges.min()}/{edges.max()}")
    
    lines = _detect_lines(cfg, edges)
    
    # Draw detected lines
    lines_img = img.copy()
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(lines_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    debug_images['lines'] = lines_img
    
    all_horizontal_lines, all_vertical_lines = _cluster_horizontal_and_vertical_lines(lines)
    
    if len(all_horizontal_lines) == 0 or len(all_vertical_lines) == 0:
        raise ValueError("No horizontal or vertical lines found after clustering. Try adjusting LINE_DETECTION parameters.")
    
    horizontal_lines = _eliminate_similar_lines(all_horizontal_lines, all_vertical_lines)
    vertical_lines = _eliminate_similar_lines(all_vertical_lines, all_horizontal_lines)
    
    # Draw filtered horizontal and vertical lines
    filtered_lines_img = img.copy()
    for rho, theta in horizontal_lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(filtered_lines_img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for horizontal
    
    for rho, theta in vertical_lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(filtered_lines_img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Red for vertical
    debug_images['filtered_lines'] = filtered_lines_img
    
    all_intersection_points = _get_intersection_points(horizontal_lines, vertical_lines)
    
    if len(all_intersection_points) == 0:
        print("WARNING: No intersection points found!")
        raise ChessboardNotLocatedException("No intersection points found")
    
    # Draw intersection points
    intersections_img = img.copy()
    for point in all_intersection_points.reshape(-1, 2):
        cv2.circle(intersections_img, (int(point[0]), int(point[1])), 3, (0, 255, 255), -1)
    debug_images['intersections'] = intersections_img
    
    # RANSAC
    best_num_inliers = 0
    best_configuration = None
    iterations = 0
    max_iterations = 1000
    
    print("\nAPI Debug - Starting RANSAC:")
    print(f"Initial intersection points shape: {all_intersection_points.shape}")
    print(f"Horizontal lines shape: {horizontal_lines.shape}")
    print(f"Vertical lines shape: {vertical_lines.shape}")
    
    try:
        while iterations < max_iterations:
            iterations += 1
            try:
                row1, row2 = _choose_from_range(len(horizontal_lines))
                col1, col2 = _choose_from_range(len(vertical_lines))
                
                print(f"\nRANSAC Iteration {iterations}:")
                print(f"Selected rows: {row1}, {row2}")
                print(f"Selected columns: {col1}, {col2}")
                
                points = np.array([
                    all_intersection_points[row1, col1],
                    all_intersection_points[row1, col2],
                    all_intersection_points[row2, col1],
                    all_intersection_points[row2, col2]
                ])
                
                if not np.all(np.isfinite(points)):
                    print("Invalid points detected, skipping iteration")
                    continue
                
                print(f"Selected points shape: {points.shape}")
                print(f"Selected points:\n{points}")
                
                transformation_matrix = _compute_homography(all_intersection_points,
                                                        row1, row2, col1, col2)
                
                if not np.all(np.isfinite(transformation_matrix)):
                    print("Invalid transformation matrix, skipping iteration")
                    continue
                
                print(f"Transformation matrix shape: {transformation_matrix.shape}")
                print(f"Transformation matrix:\n{transformation_matrix}")
                
                warped_points = _warp_points(transformation_matrix, all_intersection_points)
                
                if not np.all(np.isfinite(warped_points)):
                    print("Invalid warped points, skipping iteration")
                    continue
                    
                print(f"Warped points shape: {warped_points.shape}")
                print(f"Warped points min/max values: {np.min(warped_points)}/{np.max(warped_points)}")
                
                inliers = np.ones(warped_points.shape[:-1], dtype=bool)
                for i in range(warped_points.shape[0]):
                    for j in range(warped_points.shape[1]):
                        point = warped_points[i, j]
                        if not (0 <= point[0] <= 1 and 0 <= point[1] <= 1):
                            inliers[i, j] = False
                
                num_inliers = np.sum(inliers)
                print(f"Number of inliers: {num_inliers}")
                
                if num_inliers > best_num_inliers:
                    best_num_inliers = num_inliers
                    best_configuration = (row1, row2, col1, col2)
                    print(f"New best configuration found with {num_inliers} inliers")
                
                if num_inliers >= 30:
                    print("Found solution with sufficient inliers, exiting RANSAC")
                    break
                
            except Exception as e:
                print(f"Error in RANSAC iteration: {str(e)}")
                continue
        
        if iterations >= max_iterations:
            print(f"RANSAC reached maximum iterations ({max_iterations})")
            if best_num_inliers == 0:
                raise ChessboardNotLocatedException("No valid solution found after maximum iterations")
            print(f"Using best solution found with {best_num_inliers} inliers")
        
        # Use the best configuration found
        row1, row2, col1, col2 = best_configuration
        corners = np.array([
            all_intersection_points[row1, col1],
            all_intersection_points[row1, col2],
            all_intersection_points[row2, col1],
            all_intersection_points[row2, col2]
        ])
        
        # Draw final corners
        corners_img = img.copy()
        for corner in corners:
            cv2.circle(corners_img, (int(corner[0]), int(corner[1])), 5, (0, 0, 255), -1)
        cv2.polylines(corners_img, [corners.astype(np.int32)], True, (0, 255, 0), 2)
        debug_images['corners'] = corners_img
        
        return sort_corner_points(corners), debug_images
        
    except Exception as e:
        print(f"Error in RANSAC: {str(e)}")
        raise ChessboardNotLocatedException(f"RANSAC failed: {str(e)}")


def resize_image(cfg: CN, img: np.ndarray) -> typing.Tuple[np.ndarray, float]:
    """Resize an image for use in the corner detection pipeline, maintaining the aspect ratio.

    Args:
        cfg (CN): the configuration object
        img (np.ndarray): the input image

    Returns:
        typing.Tuple[np.ndarray, float]: the resized image along with the scale of this new image
    """
    h, w, _ = img.shape
    if w == cfg.IMAGE.WIDTH:
        return img, 1
    scale = cfg.IMAGE.WIDTH / w
    dims = (cfg.IMAGE.WIDTH, int(h * scale))

    img = cv2.resize(img, dims)
    return img, scale


def _detect_edges(edge_detection_cfg: CN, gray: np.ndarray) -> np.ndarray:
    if gray.dtype != np.uint8:
        gray = gray / gray.max() * 255
        gray = gray.astype(np.uint8)
    # Optionally use adaptive thresholding and morphology
    if getattr(edge_detection_cfg, 'USE_ADAPTIVE', False):
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        edges = cv2.Canny(closed,
                          edge_detection_cfg.LOW_THRESHOLD,
                          edge_detection_cfg.HIGH_THRESHOLD,
                          edge_detection_cfg.APERTURE)
    else:
        edges = cv2.Canny(gray,
                          edge_detection_cfg.LOW_THRESHOLD,
                          edge_detection_cfg.HIGH_THRESHOLD,
                          edge_detection_cfg.APERTURE)
    return edges


def _detect_lines(cfg: CN, edges: np.ndarray) -> np.ndarray:
    # array of [rho, theta]
    lines = cv2.HoughLines(edges, 1, np.pi/360, cfg.LINE_DETECTION.THRESHOLD)
    lines = lines.squeeze(axis=-2)
    lines = _fix_negative_rho_in_hesse_normal_form(lines)

    if cfg.LINE_DETECTION.DIAGONAL_LINE_ELIMINATION:
        threshold = np.deg2rad(
            cfg.LINE_DETECTION.DIAGONAL_LINE_ELIMINATION_THRESHOLD_DEGREES)
        vmask = np.abs(lines[:, 1]) < threshold
        hmask = np.abs(lines[:, 1] - np.pi / 2) < threshold
        mask = vmask | hmask
        lines = lines[mask]
    return lines


def _fix_negative_rho_in_hesse_normal_form(lines: np.ndarray) -> np.ndarray:
    lines = lines.copy()
    neg_rho_mask = lines[..., 0] < 0
    lines[neg_rho_mask, 0] = - \
        lines[neg_rho_mask, 0]
    lines[neg_rho_mask, 1] =  \
        lines[neg_rho_mask, 1] - np.pi
    return lines


def _absolute_angle_difference(x, y):
    diff = np.mod(np.abs(x - y), 2*np.pi)
    return np.min(np.stack([diff, np.pi - diff], axis=-1), axis=-1)


def _sort_lines(lines: np.ndarray) -> np.ndarray:
    if lines.ndim == 0 or lines.shape[-2] == 0:
        return lines
    rhos = lines[..., 0]
    sorted_indices = np.argsort(rhos)
    return lines[sorted_indices]


def _cluster_horizontal_and_vertical_lines(lines: np.ndarray):
    """Cluster lines into horizontal and vertical groups based on their angles.
    
    Args:
        lines (np.ndarray): Array of lines in Hesse normal form [rho, theta]
        
    Returns:
        tuple: (horizontal_lines, vertical_lines) arrays of lines
    """
    print("\nAPI Debug - Line Clustering:")
    print(f"Total lines before clustering: {len(lines)}")
    
    if lines.shape[0] == 0:
        print("No lines to cluster!")
        return np.array([]), np.array([])
        
    # Calculate angle differences from horizontal (0) and vertical (pi/2)
    angles = lines[:, 1]
    horizontal_diff = _absolute_angle_difference(angles, 0)
    vertical_diff = _absolute_angle_difference(angles, np.pi/2)
    
    # Classify lines based on which angle difference is smaller
    is_horizontal = horizontal_diff < vertical_diff
    horizontal_lines = lines[is_horizontal]
    vertical_lines = lines[~is_horizontal]
    
    print(f"Lines classified as horizontal: {len(horizontal_lines)}")
    print(f"Lines classified as vertical: {len(vertical_lines)}")
    
    if len(horizontal_lines) == 0:
        print("Warning: No horizontal lines found after clustering!")
    if len(vertical_lines) == 0:
        print("Warning: No vertical lines found after clustering!")
    
    return horizontal_lines, vertical_lines


def _eliminate_similar_lines(lines: np.ndarray, perpendicular_lines: np.ndarray) -> np.ndarray:
    perp_rho, perp_theta = perpendicular_lines.mean(axis=0)
    rho, theta = np.moveaxis(lines, -1, 0)
    intersection_points = get_intersection_point(
        rho, theta, perp_rho, perp_theta)
    intersection_points = np.stack(intersection_points, axis=-1)

    clustering = DBSCAN(eps=12, min_samples=1).fit(intersection_points)

    filtered_lines = []
    for c in range(clustering.labels_.max() + 1):
        lines_in_cluster = lines[clustering.labels_ == c]
        rho = lines_in_cluster[..., 0]
        median = np.argsort(rho)[len(rho)//2]
        filtered_lines.append(lines_in_cluster[median])
    return np.stack(filtered_lines)


def get_intersection_point(rho1: np.ndarray, theta1: np.ndarray, rho2: np.ndarray, theta2: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Obtain the intersection point of two lines in Hough space.

    This method can be batched

    Args:
        rho1 (np.ndarray): first line's rho
        theta1 (np.ndarray): first line's theta
        rho2 (np.ndarray): second lines's rho
        theta2 (np.ndarray): second line's theta

    Returns:
        typing.Tuple[np.ndarray, np.ndarray]: the x and y coordinates of the intersection point(s)
    """
    # rho1 = x cos(theta1) + y sin(theta1)
    # rho2 = x cos(theta2) + y sin(theta2)
    cos_t1 = np.cos(theta1)
    cos_t2 = np.cos(theta2)
    sin_t1 = np.sin(theta1)
    sin_t2 = np.sin(theta2)
    x = (sin_t1 * rho2 - sin_t2 * rho1) / (cos_t2 * sin_t1 - cos_t1 * sin_t2)
    y = (cos_t1 * rho2 - cos_t2 * rho1) / (sin_t2 * cos_t1 - sin_t1 * cos_t2)
    return x, y


def _choose_from_range(upper_bound: int, n: int = 2):
    return np.sort(np.random.choice(np.arange(upper_bound), (n,), replace=False), axis=-1)


def _get_intersection_points(horizontal_lines: np.ndarray, vertical_lines: np.ndarray) -> np.ndarray:
    rho1, theta1 = np.moveaxis(horizontal_lines, -1, 0)
    rho2, theta2 = np.moveaxis(vertical_lines, -1, 0)

    rho1, rho2 = np.meshgrid(rho1, rho2, indexing="ij")
    theta1, theta2 = np.meshgrid(theta1, theta2, indexing="ij")
    intersection_points = get_intersection_point(rho1, theta1, rho2, theta2)
    intersection_points = np.stack(intersection_points, axis=-1)
    return intersection_points


def compute_transformation_matrix(src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
    """Compute the transformation matrix based on source and destination points.

    Args:
        src_points (np.ndarray): the source points (shape: [..., 2])
        dst_points (np.ndarray): the source points (shape: [..., 2])

    Returns:
        np.ndarray: the transformation matrix
    """
    transformation_matrix, _ = cv2.findHomography(src_points.reshape(-1, 2),
                                                  dst_points.reshape(-1, 2))
    return transformation_matrix


def _compute_homography(intersection_points: np.ndarray, row1: int, row2: int, col1: int, col2: int):
    p1 = intersection_points[row1, col1]  # top left
    p2 = intersection_points[row1, col2]  # top right
    p3 = intersection_points[row2, col2]  # bottom right
    p4 = intersection_points[row2, col1]  # bottom left

    src_points = np.stack([p1, p2, p3, p4])
    dst_points = np.array([[0, 0],  # top left
                           [1, 0],  # top right
                           [1, 1],  # bottom right
                           [0, 1]])  # bottom left
    return compute_transformation_matrix(src_points, dst_points)


def _warp_points(transformation_matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = to_homogenous_coordinates(points)
    warped_points = points @ transformation_matrix.T
    return from_homogenous_coordinates(warped_points)


def _find_best_scale(cfg: CN, values: np.ndarray, scales: np.ndarray = np.arange(1, 9)):
    scales = np.sort(scales)
    scaled_values = np.expand_dims(values, axis=-1) * scales
    diff = np.abs(np.rint(scaled_values) - scaled_values)

    inlier_mask = diff < cfg.RANSAC.OFFSET_TOLERANCE / scales
    num_inliers = np.sum(inlier_mask, axis=tuple(range(inlier_mask.ndim - 1)))

    best_num_inliers = np.max(num_inliers)

    # We will choose a slightly worse scale if it is lower
    index = np.argmax(num_inliers > (
        1 - cfg.RANSAC.BEST_SOLUTION_TOLERANCE) * best_num_inliers)
    return scales[index], inlier_mask[..., index]


def _discard_outliers(cfg: CN, warped_points: np.ndarray, intersection_points: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, float, float]:
    horizontal_scale, horizontal_mask = _find_best_scale(
        cfg, warped_points[..., 0])
    vertical_scale, vertical_mask = _find_best_scale(
        cfg, warped_points[..., 1])
    mask = horizontal_mask & vertical_mask

    # Keep rows/cols that have more than 50% inliers
    num_rows_to_consider = np.any(mask, axis=-1).sum()
    num_cols_to_consider = np.any(mask, axis=-2).sum()
    rows_to_keep = mask.sum(axis=-1) / num_rows_to_consider > \
        cfg.MAX_OUTLIER_INTERSECTION_POINT_RATIO_PER_LINE
    cols_to_keep = mask.sum(axis=-2) / num_cols_to_consider > \
        cfg.MAX_OUTLIER_INTERSECTION_POINT_RATIO_PER_LINE

    warped_points = warped_points[rows_to_keep][:, cols_to_keep]
    intersection_points = intersection_points[rows_to_keep][:, cols_to_keep]
    return warped_points, intersection_points, horizontal_scale, vertical_scale


def _quantize_points(cfg: CN, warped_scaled_points: np.ndarray, intersection_points: np.ndarray) -> typing.Tuple[tuple, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean_col_xs = warped_scaled_points[..., 0].mean(axis=0)
    mean_row_ys = warped_scaled_points[..., 1].mean(axis=1)

    col_xs = np.rint(mean_col_xs).astype(np.int32)
    row_ys = np.rint(mean_row_ys).astype(np.int32)

    # Remove duplicates
    col_xs, col_indices = np.unique(col_xs, return_index=True)
    row_ys, row_indices = np.unique(row_ys, return_index=True)
    intersection_points = intersection_points[row_indices][:, col_indices]

    # Compute mins and maxs in warped space
    xmin = col_xs.min()
    xmax = col_xs.max()
    ymin = row_ys.min()
    ymax = row_ys.max()

    # Ensure we a have a maximum of 9 rows/cols
    while xmax - xmin > 8:
        xmax -= 1
        xmin += 1
    while ymax - ymin > 8:
        ymax -= 1
        ymin += 1
    col_mask = (col_xs >= xmin) & (col_xs <= xmax)
    row_mask = (row_ys >= xmin) & (row_ys <= xmax)

    # Discard
    col_xs = col_xs[col_mask]
    row_ys = row_ys[row_mask]
    intersection_points = intersection_points[row_mask][:, col_mask]

    # Create quantized points array
    quantized_points = np.stack(np.meshgrid(col_xs, row_ys), axis=-1)

    # Transform in warped space
    translation = -np.array([xmin, ymin]) + \
        cfg.BORDER_REFINEMENT.NUM_SURROUNDING_SQUARES_IN_WARPED_IMG
    scale = np.array(cfg.BORDER_REFINEMENT.WARPED_SQUARE_SIZE)

    scaled_quantized_points = (quantized_points + translation) * scale
    xmin, ymin = np.array((xmin, ymin)) + translation
    xmax, ymax = np.array((xmax, ymax)) + translation
    warped_img_size = (np.array((xmax, ymax)) +
                       cfg.BORDER_REFINEMENT.NUM_SURROUNDING_SQUARES_IN_WARPED_IMG) * scale

    return (xmin, xmax, ymin, ymax), scale, scaled_quantized_points, intersection_points, warped_img_size


def _compute_vertical_borders(cfg: CN, warped: np.ndarray, mask: np.ndarray, scale: np.ndarray, xmin: int, xmax: int) -> typing.Tuple[int, int]:
    G_x = np.abs(cv2.Sobel(warped, cv2.CV_64F, 1, 0,
                           ksize=cfg.BORDER_REFINEMENT.SOBEL_KERNEL_SIZE))
    G_x[~mask] = 0
    G_x = _detect_edges(cfg.BORDER_REFINEMENT.EDGE_DETECTION.VERTICAL, G_x)
    G_x[~mask] = 0

    def get_nonmax_supressed(x):
        x = (x * scale[0]).astype(np.int32)
        thresh = cfg.BORDER_REFINEMENT.LINE_WIDTH // 2
        return G_x[:, x-thresh:x+thresh+1].max(axis=1)

    while xmax - xmin < 8:
        top = get_nonmax_supressed(xmax + 1)
        bottom = get_nonmax_supressed(xmin - 1)

        if top.sum() > bottom.sum():
            xmax += 1
        else:
            xmin -= 1

    return xmin, xmax


def _compute_horizontal_borders(cfg: CN, warped: np.ndarray, mask: np.ndarray, scale: np.ndarray, ymin: int, ymax: int) -> typing.Tuple[int, int]:
    G_y = np.abs(cv2.Sobel(warped, cv2.CV_64F, 0, 1,
                           ksize=cfg.BORDER_REFINEMENT.SOBEL_KERNEL_SIZE))
    G_y[~mask] = 0
    G_y = _detect_edges(cfg.BORDER_REFINEMENT.EDGE_DETECTION.HORIZONTAL, G_y)
    G_y[~mask] = 0

    def get_nonmax_supressed(y):
        y = (y * scale[1]).astype(np.int32)
        thresh = cfg.BORDER_REFINEMENT.LINE_WIDTH // 2
        return G_y[y-thresh:y+thresh+1].max(axis=0)

    while ymax - ymin < 8:
        top = get_nonmax_supressed(ymax + 1)
        bottom = get_nonmax_supressed(ymin - 1)

        if top.sum() > bottom.sum():
            ymax += 1
        else:
            ymin -= 1
    return ymin, ymax


def _find_best_corners(intersections: np.ndarray) -> np.ndarray:
    """Find the best 4 corners of the chessboard using RANSAC.

    Args:
        intersections: The intersections of the horizontal and vertical lines

    Returns:
        The corners of the chessboard as a 4x2 array of (x, y) coordinates
    """
    print("\nAPI Debug - RANSAC Corner Detection:")
    print(f"Number of intersections to process: {len(intersections)}")
    
    if len(intersections) < 4:
        print("Not enough intersections for RANSAC")
        raise ChessboardNotLocatedException("not enough intersections for RANSAC")
    
    # Find the best 4 corners using RANSAC
    best_corners = None
    best_score = float('inf')
    
    for _ in range(1000):  # Try 1000 times
        # Randomly select 4 points
        indices = np.random.choice(len(intersections), 4, replace=False)
        corners = intersections[indices]
        
        # Calculate the score
        score = _calculate_corner_score(corners)
        
        if score < best_score:
            best_score = score
            best_corners = corners
            print(f"Found better corners with score: {score:.2f}")
    
    if best_corners is None:
        print("RANSAC failed to find viable corners")
        raise ChessboardNotLocatedException("RANSAC produced no viable results")
    
    print("\nFinal RANSAC Results:")
    print(f"Best score: {best_score:.2f}")
    print("Best corners:")
    print(best_corners)
    
    return best_corners


def _calculate_corner_score(corners: np.ndarray) -> float:
    """Calculate the score for a set of corners.

    Args:
        corners: The corners of the chessboard as a 4x2 array of (x, y) coordinates

    Returns:
        The score for the corners
    """
    # Calculate the distances between the corners
    distances = []
    for i in range(4):
        for j in range(i + 1, 4):
            distances.append(np.linalg.norm(corners[i] - corners[j]))
    
    # Calculate the score
    score = np.std(distances) / np.mean(distances)
    
    # Debug print for corner scoring
    print(f"\nCorner Score Calculation:")
    print(f"Corner coordinates:")
    print(corners)
    print(f"Distances between corners: {[f'{d:.2f}' for d in distances]}")
    print(f"Score: {score:.2f}")
    
    return score


def _find_horizontal_and_vertical_lines(lines: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Find the horizontal and vertical lines in the image.

    Args:
        lines: The lines detected in the image

    Returns:
        A tuple of (horizontal_lines, vertical_lines)
    """
    print("\nAPI Debug - Line Classification:")
    print(f"Total lines before classification: {len(lines)}")
    
    # Find the horizontal and vertical lines
    horizontal_lines = []
    vertical_lines = []
    
    for line in lines:
        rho, theta = line
        # Convert theta to degrees for easier debugging
        theta_deg = np.degrees(theta)
        print(f"Line: rho={rho:.2f}, theta={theta_deg:.2f}°")
        
        # Check if the line is horizontal
        if abs(theta - np.pi/2) < cfg.LINE_DETECTION.HORIZONTAL_THRESHOLD:
            horizontal_lines.append(line)
            print(f"  Classified as horizontal")
        # Check if the line is vertical
        elif abs(theta) < cfg.LINE_DETECTION.VERTICAL_THRESHOLD or abs(theta - np.pi) < cfg.LINE_DETECTION.VERTICAL_THRESHOLD:
            vertical_lines.append(line)
            print(f"  Classified as vertical")
        else:
            print(f"  Skipped (not horizontal or vertical)")
    
    horizontal_lines = np.array(horizontal_lines)
    vertical_lines = np.array(vertical_lines)
    
    print(f"\nClassification results:")
    print(f"Horizontal lines: {len(horizontal_lines)}")
    print(f"Vertical lines: {len(vertical_lines)}")
    
    return horizontal_lines, vertical_lines


def _find_intersections(horizontal_lines: np.ndarray, vertical_lines: np.ndarray) -> np.ndarray:
    """Find the intersections of the horizontal and vertical lines.

    Args:
        horizontal_lines: The horizontal lines
        vertical_lines: The vertical lines

    Returns:
        The intersections of the lines as a Nx2 array of (x, y) coordinates
    """
    print("\nAPI Debug - Intersection Detection:")
    print(f"Finding intersections between {len(horizontal_lines)} horizontal and {len(vertical_lines)} vertical lines")
    
    intersections = []
    for h_line in horizontal_lines:
        h_rho, h_theta = h_line
        for v_line in vertical_lines:
            v_rho, v_theta = v_line
            
            # Calculate the intersection
            A = np.array([
                [np.cos(h_theta), np.sin(h_theta)],
                [np.cos(v_theta), np.sin(v_theta)]
            ])
            b = np.array([h_rho, v_rho])
            
            try:
                x, y = np.linalg.solve(A, b)
                # Check if the intersection is within the image bounds
                if 0 <= x < cfg.IMAGE.WIDTH and 0 <= y < cfg.IMAGE.HEIGHT:
                    intersections.append([x, y])
                    print(f"Found intersection at ({x:.2f}, {y:.2f})")
            except np.linalg.LinAlgError:
                print(f"Failed to find intersection for lines:")
                print(f"  Horizontal: rho={h_rho:.2f}, theta={np.degrees(h_theta):.2f}°")
                print(f"  Vertical: rho={v_rho:.2f}, theta={np.degrees(v_theta):.2f}°")
    
    intersections = np.array(intersections)
    print(f"\nTotal intersections found: {len(intersections)}")
    if len(intersections) > 0:
        print("First few intersections:")
        print(intersections[:5])
    
    return intersections


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser(description="Chessboard corner detector.")
    parser.add_argument("file", type=str, help="URI of the input image file")
    parser.add_argument("--config", type=str, help="path to the config file",
                        default="config://corner_detection.yaml")
    args = parser.parse_args()

    cfg = CN.load_yaml_with_base(args.config)
    filename = URI(args.file)
    img = cv2.imread(str(filename))
    corners, debug_images = find_corners(cfg, img)

    fig = plt.figure()
    fig.canvas.set_window_title("Corner detection output")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.scatter(*corners.T, c="r")
    plt.axis("off")
    plt.show()
