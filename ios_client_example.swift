import Foundation
import UIKit

// MARK: - API Response Models
struct ChessRecognitionResponse: Codable {
    let fen: String
    let ascii: String
    let lichessUrl: String
    let legalPosition: Bool
    let debugImages: [String: String]
    let corners: [[Double]]?
    let processingTime: Double
    let imageInfo: ImageInfo
    let debugInfo: DebugInfo
    
    enum CodingKeys: String, CodingKey {
        case fen, ascii
        case lichessUrl = "lichess_url"
        case legalPosition = "legal_position"
        case debugImages = "debug_images"
        case corners, processingTime = "processing_time"
        case imageInfo = "image_info"
        case debugInfo = "debug_info"
    }
}

struct ImageInfo: Codable {
    let filename: String?
    let contentType: String
    let sizeBytes: Int
    let shape: [Int]
    
    enum CodingKeys: String, CodingKey {
        case filename
        case contentType = "content_type"
        case sizeBytes = "size_bytes"
        case shape
    }
}

struct DebugInfo: Codable {
    let cornerDetection: String
    let boardWarping: String
    let positionDetection: String
    let visualization: String
    
    enum CodingKeys: String, CodingKey {
        case cornerDetection = "corner_detection"
        case boardWarping = "board_warping"
        case positionDetection = "position_detection"
        case visualization
    }
}

struct CornerDetectionResponse: Codable {
    let corners: [[Double]]?
    let message: String
    let debugImages: [String: String]
    let processingTime: Double
    let imageInfo: ImageInfo
    
    enum CodingKeys: String, CodingKey {
        case corners, message
        case debugImages = "debug_images"
        case processingTime = "processing_time"
        case imageInfo = "image_info"
    }
}

// MARK: - API Client
class ChessPositionScannerAPI {
    private let baseURL: String
    
    init(baseURL: String = "https://api.chesspositionscanner.store") {
        self.baseURL = baseURL
    }
    
    // MARK: - Health Check
    func checkHealth() async throws -> Bool {
        let url = URL(string: "\(baseURL)/health")!
        let (data, response) = try await URLSession.shared.data(from: url)
        
        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.invalidResponse
        }
        
        if httpResponse.statusCode == 200 {
            let healthData = try JSONDecoder().decode([String: Any].self, from: data)
            return healthData["models_loaded"] as? Bool ?? false
        }
        
        return false
    }
    
    // MARK: - Recognize Chess Position
    func recognizeChessPosition(
        image: UIImage, 
        color: String = "white",
        debugImageWidth: Int = 800,
        debugImageHeight: Int = 600
    ) async throws -> ChessRecognitionResponse {
        let url = URL(string: "\(baseURL)/recognize_chess_position")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        
        // Create multipart form data
        let boundary = UUID().uuidString
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        
        var body = Data()
        
        // Add image data
        guard let imageData = image.jpegData(compressionQuality: 0.8) else {
            throw APIError.imageConversionFailed
        }
        
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"image\"; filename=\"image.jpg\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
        body.append(imageData)
        body.append("\r\n".data(using: .utf8)!)
        
        // Add color parameter
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"color\"\r\n\r\n".data(using: .utf8)!)
        body.append(color.data(using: .utf8)!)
        body.append("\r\n".data(using: .utf8)!)
        
        // Add debug image size parameters
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"debug_image_width\"\r\n\r\n".data(using: .utf8)!)
        body.append("\(debugImageWidth)".data(using: .utf8)!)
        body.append("\r\n".data(using: .utf8)!)
        
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"debug_image_height\"\r\n\r\n".data(using: .utf8)!)
        body.append("\(debugImageHeight)".data(using: .utf8)!)
        body.append("\r\n".data(using: .utf8)!)
        
        body.append("--\(boundary)--\r\n".data(using: .utf8)!)
        
        request.httpBody = body
        
        let (data, response) = try await URLSession.shared.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.invalidResponse
        }
        
        if httpResponse.statusCode == 200 {
            return try JSONDecoder().decode(ChessRecognitionResponse.self, from: data)
        } else {
            throw APIError.serverError(httpResponse.statusCode)
        }
    }
    
    // MARK: - Detect Corners
    func detectCorners(image: UIImage) async throws -> CornerDetectionResponse {
        let url = URL(string: "\(baseURL)/detect_corners")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        
        // Create multipart form data
        let boundary = UUID().uuidString
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        
        var body = Data()
        
        // Add image data
        guard let imageData = image.jpegData(compressionQuality: 0.8) else {
            throw APIError.imageConversionFailed
        }
        
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"image\"; filename=\"image.jpg\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
        body.append(imageData)
        body.append("\r\n".data(using: .utf8)!)
        
        body.append("--\(boundary)--\r\n".data(using: .utf8)!)
        
        request.httpBody = body
        
        let (data, response) = try await URLSession.shared.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.invalidResponse
        }
        
        if httpResponse.statusCode == 200 {
            return try JSONDecoder().decode(CornerDetectionResponse.self, from: data)
        } else {
            throw APIError.serverError(httpResponse.statusCode)
        }
    }
}

// MARK: - Debug Image Helper
extension ChessPositionScannerAPI {
    func decodeDebugImage(base64String: String) -> UIImage? {
        guard let data = Data(base64Encoded: base64String) else {
            return nil
        }
        return UIImage(data: data)
    }
}

// MARK: - Errors
enum APIError: Error, LocalizedError {
    case invalidResponse
    case imageConversionFailed
    case serverError(Int)
    case networkError(Error)
    
    var errorDescription: String? {
        switch self {
        case .invalidResponse:
            return "Invalid response from server"
        case .imageConversionFailed:
            return "Failed to convert image to JPEG data"
        case .serverError(let code):
            return "Server error with status code: \(code)"
        case .networkError(let error):
            return "Network error: \(error.localizedDescription)"
        }
    }
}

// MARK: - Debug Image Types
enum DebugImageType: String, CaseIterable {
    case resized = "resized"
    case edges = "edges"
    case lines = "lines"
    case filteredLines = "filtered_lines"
    case intersections = "intersections"
    case corners = "corners"
    case warpedBoard = "warped_board"
    case detectedPosition = "detected_position"
    case squareGrid = "square_grid"
    
    var displayName: String {
        switch self {
        case .resized: return "Resized Image"
        case .edges: return "Edge Detection"
        case .lines: return "Line Detection"
        case .filteredLines: return "Filtered Lines"
        case .intersections: return "Intersections"
        case .corners: return "Detected Corners"
        case .warpedBoard: return "Warped Board"
        case .detectedPosition: return "Detected Position"
        case .squareGrid: return "Square Grid"
        }
    }
    
    var description: String {
        switch self {
        case .resized: return "Input image resized for processing"
        case .edges: return "Edge detection results using Canny algorithm"
        case .lines: return "All detected lines using Hough transform"
        case .filteredLines: return "Filtered horizontal and vertical lines"
        case .intersections: return "Intersection points of lines"
        case .corners: return "Final detected chess board corners"
        case .warpedBoard: return "Chess board warped to regular grid"
        case .detectedPosition: return "Visual representation of detected position"
        case .squareGrid: return "Grid showing all 64 squares"
        }
    }
}

// MARK: - Usage Example
class ChessPositionViewController: UIViewController {
    private let api = ChessPositionScannerAPI()
    
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var resultLabel: UILabel!
    @IBOutlet weak var debugImageView: UIImageView!
    @IBOutlet weak var debugCollectionView: UICollectionView!
    
    private var debugImages: [DebugImageType: UIImage] = [:]
    private var currentDebugImageType: DebugImageType = .resized
    
    @IBAction func scanChessPosition(_ sender: UIButton) {
        guard let image = imageView.image else {
            resultLabel.text = "No image selected"
            return
        }
        
        Task {
            do {
                // Check API health first
                let isHealthy = try await api.checkHealth()
                if !isHealthy {
                    await MainActor.run {
                        resultLabel.text = "API is not healthy"
                    }
                    return
                }
                
                // Recognize chess position
                let response = try await api.recognizeChessPosition(
                    image: image,
                    debugImageWidth: 600,
                    debugImageHeight: 400
                )
                
                await MainActor.run {
                    // Display results
                    resultLabel.text = """
                    FEN: \(response.fen)
                    Legal: \(response.legalPosition)
                    Lichess: \(response.lichessUrl)
                    """
                    
                    // Process debug images
                    self.processDebugImages(response.debugImages)
                    
                    // Display first debug image
                    if let firstType = DebugImageType.allCases.first,
                       let image = self.debugImages[firstType] {
                        self.debugImageView.image = image
                        self.currentDebugImageType = firstType
                    }
                    
                    // Reload collection view
                    self.debugCollectionView.reloadData()
                }
                
            } catch {
                await MainActor.run {
                    resultLabel.text = "Error: \(error.localizedDescription)"
                }
            }
        }
    }
    
    @IBAction func detectCorners(_ sender: UIButton) {
        guard let image = imageView.image else {
            resultLabel.text = "No image selected"
            return
        }
        
        Task {
            do {
                let response = try await api.detectCorners(image: image)
                
                await MainActor.run {
                    resultLabel.text = "Corners detected: \(response.corners?.count ?? 0)"
                    
                    // Process debug images
                    self.processDebugImages(response.debugImages)
                    
                    // Display first debug image
                    if let firstType = DebugImageType.allCases.first,
                       let image = self.debugImages[firstType] {
                        self.debugImageView.image = image
                        self.currentDebugImageType = firstType
                    }
                    
                    // Reload collection view
                    self.debugCollectionView.reloadData()
                }
                
            } catch {
                await MainActor.run {
                    resultLabel.text = "Error: \(error.localizedDescription)"
                }
            }
        }
    }
    
    private func processDebugImages(_ debugImagesDict: [String: String]) {
        debugImages.removeAll()
        
        for (key, base64String) in debugImagesDict {
            if let debugType = DebugImageType(rawValue: key),
               let image = api.decodeDebugImage(base64String: base64String) {
                debugImages[debugType] = image
            }
        }
    }
    
    @IBAction func showDebugImageInfo(_ sender: UIButton) {
        let alert = UIAlertController(
            title: currentDebugImageType.displayName,
            message: currentDebugImageType.description,
            preferredStyle: .alert
        )
        alert.addAction(UIAlertAction(title: "OK", style: .default))
        present(alert, animated: true)
    }
}

// MARK: - Collection View for Debug Images
extension ChessPositionViewController: UICollectionViewDataSource, UICollectionViewDelegate {
    func collectionView(_ collectionView: UICollectionView, numberOfItemsInSection section: Int) -> Int {
        return debugImages.count
    }
    
    func collectionView(_ collectionView: UICollectionView, cellForItemAt indexPath: IndexPath) -> UICollectionViewCell {
        let cell = collectionView.dequeueReusableCell(withReuseIdentifier: "DebugImageCell", for: indexPath) as! DebugImageCell
        
        let debugTypes = Array(debugImages.keys)
        let debugType = debugTypes[indexPath.item]
        let image = debugImages[debugType]!
        
        cell.configure(with: image, title: debugType.displayName)
        return cell
    }
    
    func collectionView(_ collectionView: UICollectionView, didSelectItemAt indexPath: IndexPath) {
        let debugTypes = Array(debugImages.keys)
        let debugType = debugTypes[indexPath.item]
        let image = debugImages[debugType]!
        
        debugImageView.image = image
        currentDebugImageType = debugType
    }
}

// MARK: - Debug Image Cell
class DebugImageCell: UICollectionViewCell {
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var titleLabel: UILabel!
    
    func configure(with image: UIImage, title: String) {
        imageView.image = image
        titleLabel.text = title
        titleLabel.textAlignment = .center
        titleLabel.font = UIFont.systemFont(ofSize: 12)
    }
} 