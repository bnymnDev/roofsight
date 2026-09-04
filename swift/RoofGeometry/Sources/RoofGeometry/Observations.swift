import Foundation

/// The `edge_type` attribute of the dataset, verbatim.
public enum EdgeType: String, Sendable, Codable, CaseIterable {
    case eave, ridge, hip, valley, verge

    /// Angle in the horizontal plane between this edge and the eave, for slope edges.
    /// Verges run straight up the slope (90°); hips of a regular hip roof bisect the corner (45°).
    var planAngleToEaveDegrees: Double? {
        switch self {
        case .verge, .valley: return 90
        case .hip: return 45
        case .eave, .ridge: return nil
        }
    }

    var isHorizontal: Bool { self == .eave || self == .ridge }
}

/// One `roof_edge` instance as pixel samples along its mask (skeleton or contour points).
public struct EdgeObservation: Sendable, Hashable, Codable {
    public var type: EdgeType
    public var points: [Vec2]

    public init(type: EdgeType, points: [Vec2]) {
        self.type = type
        self.points = points
    }
}

/// Sparse metric depth (meters along the camera z axis) at pixel positions inside the plane
/// mask. From ARKit's `sceneDepth` when the device has LiDAR.
public struct DepthSamples: Sendable, Hashable, Codable {
    public var pixels: [Vec2]
    public var depths: [Double]

    public init(pixels: [Vec2], depths: [Double]) {
        precondition(pixels.count == depths.count)
        self.pixels = pixels
        self.depths = depths
    }
}

/// One `roof_plane` instance with the edges that touch it. Produced from the Core ML output.
public struct RoofPlaneObservation: Sendable, Hashable, Codable {
    public var id: Int
    /// A representative interior pixel: the mask centroid.
    public var centroid: Vec2
    public var edges: [EdgeObservation]
    public var depth: DepthSamples?

    public init(id: Int, centroid: Vec2, edges: [EdgeObservation], depth: DepthSamples? = nil) {
        self.id = id
        self.centroid = centroid
        self.edges = edges
        self.depth = depth
    }
}
