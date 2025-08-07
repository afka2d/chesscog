import Foundation
import UIKit

// MARK: - Models
struct ManualCornersRequest {
    let image: UIImage
    let corners: [[Double]]  // [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    let color: String
    let debugImageWidth: Int
    let debugImageHeight: Int
}

struct ChessPositionResponse: Codable {
    let fen: String
    let ascii: String
    let lichessUrl: String
    let legalPosition: Bool
    let debugImages: [String: String]  // base64 encoded images
    let debugImagePaths: [String: String]
    let corners: [[Double]]
    let processingTime: Double
    let imageInfo: ImageInfo
    let debugInfo: DebugInfo
    
    enum CodingKeys: String, CodingKey {
        case fen, ascii
        case lichessUrl = "lichess_url"
        case legalPosition = "legal_position"
        case debugImages = "debug_images"
        case debugImagePaths = "debug_image_paths"
        case corners, processingTime, imageInfo, debugInfo
    }
}

struct ImageInfo: Codable {
    let filename: String
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

// MARK: - API Client
class ChessPositionAPI {
    private let baseURL = "http://localhost:8000"
    
    func recognizePositionWithManualCorners(
        request: ManualCornersRequest,
        completion: @escaping (Result<ChessPositionResponse, Error>) -> Void
    ) {
        guard let imageData = request.image.jpegData(compressionQuality: 0.8) else {
            completion(.failure(APIError.invalidImage))
            return
        }
        
        let url = URL(string: "\(baseURL)/recognize_chess_position_with_corners")!
        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = "POST"
        
        let boundary = UUID().uuidString
        urlRequest.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        
        var body = Data()
        
        // Add image
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"image\"; filename=\"chess_board.jpg\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
        body.append(imageData)
        body.append("\r\n".data(using: .utf8)!)
        
        // Add corners
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"corners\"\r\n\r\n".data(using: .utf8)!)
        let cornersJson = try! JSONSerialization.data(withJSONObject: request.corners)
        body.append(cornersJson)
        body.append("\r\n".data(using: .utf8)!)
        
        // Add color
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"color\"\r\n\r\n".data(using: .utf8)!)
        body.append(request.color.data(using: .utf8)!)
        body.append("\r\n".data(using: .utf8)!)
        
        // Add debug image width
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"debug_image_width\"\r\n\r\n".data(using: .utf8)!)
        body.append("\(request.debugImageWidth)".data(using: .utf8)!)
        body.append("\r\n".data(using: .utf8)!)
        
        // Add debug image height
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"debug_image_height\"\r\n\r\n".data(using: .utf8)!)
        body.append("\(request.debugImageHeight)".data(using: .utf8)!)
        body.append("\r\n".data(using: .utf8)!)
        
        body.append("--\(boundary)--\r\n".data(using: .utf8)!)
        
        urlRequest.httpBody = body
        
        URLSession.shared.dataTask(with: urlRequest) { data, response, error in
            DispatchQueue.main.async {
                if let error = error {
                    completion(.failure(error))
                    return
                }
                
                guard let data = data else {
                    completion(.failure(APIError.noData))
                    return
                }
                
                do {
                    let response = try JSONDecoder().decode(ChessPositionResponse.self, from: data)
                    completion(.success(response))
                } catch {
                    completion(.failure(error))
                }
            }
        }.resume()
    }
}

// MARK: - Errors
enum APIError: Error {
    case invalidImage
    case noData
    case invalidResponse
}

// MARK: - Usage Example
class ChessPositionViewController: UIViewController {
    private let api = ChessPositionAPI()
    
    func submitManualCorners() {
        // Example: User has corrected the corner coordinates in your UI
        let correctedCorners: [[Double]] = [
            [586.3321, 960.0475],   // Top-left
            [1109.8192, 978.328],   // Top-right
            [584.748, 899.5496],    // Bottom-left
            [1109.9733, 982.7372]   // Bottom-right
        ]
        
        // Get the chess board image from your UI
        guard let chessBoardImage = getChessBoardImage() else {
            print("No chess board image available")
            return
        }
        
        let request = ManualCornersRequest(
            image: chessBoardImage,
            corners: correctedCorners,
            color: "white",
            debugImageWidth: 800,
            debugImageHeight: 600
        )
        
        api.recognizePositionWithManualCorners(request: request) { result in
            switch result {
            case .success(let response):
                print("FEN: \(response.fen)")
                print("Legal Position: \(response.legalPosition)")
                print("Lichess URL: \(response.lichessUrl)")
                print("ASCII Board:\n\(response.ascii)")
                
                // Display debug images
                self.displayDebugImages(response.debugImages)
                
            case .failure(let error):
                print("Error: \(error)")
            }
        }
    }
    
    private func getChessBoardImage() -> UIImage? {
        // Return the chess board image from your UI
        // This would be the image the user captured or selected
        return nil // Replace with actual image
    }
    
    private func displayDebugImages(_ debugImages: [String: String]) {
        // Convert base64 strings back to UIImage and display them
        for (key, base64String) in debugImages {
            if let imageData = Data(base64Encoded: base64String),
               let image = UIImage(data: imageData) {
                // Display the image in your UI
                print("Debug image \(key) loaded: \(image.size)")
            }
        }
    }
}

// MARK: - Helper Extensions
extension String {
    func data(using encoding: String.Encoding) -> Data? {
        return self.data(using: encoding)
    }
}

extension Data {
    mutating func append(_ string: String) {
        if let data = string.data(using: .utf8) {
            append(data)
        }
    }
} 