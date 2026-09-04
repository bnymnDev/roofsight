import Foundation

/// Pinhole intrinsics in pixels. The image origin is top-left, y down.
public struct CameraIntrinsics: Sendable, Hashable, Codable {
    public var fx: Double
    public var fy: Double
    public var cx: Double
    public var cy: Double
    public var width: Int
    public var height: Int

    public init(fx: Double, fy: Double, cx: Double, cy: Double, width: Int, height: Int) {
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
    }

    /// Unit ray through a pixel, in the camera frame (x right, y down, z forward).
    public func ray(through p: Vec2) -> Vec3 {
        Vec3((p.x - cx) / fx, (p.y - cy) / fy, 1).normalized
    }

    /// Project a camera-frame point with positive depth. Returns nil behind the camera.
    public func project(_ p: Vec3) -> Vec2? {
        guard p.z > 1e-9 else { return nil }
        return Vec2(fx * p.x / p.z + cx, fy * p.y / p.z + cy)
    }

    /// Camera-frame point for a pixel at a given metric depth (z).
    public func unproject(_ p: Vec2, depth: Double) -> Vec3 {
        Vec3((p.x - cx) / fx * depth, (p.y - cy) / fy * depth, depth)
    }
}

/// Where "down" and "north" are, expressed in the camera frame (x right, y down, z forward).
///
/// This is the whole world the estimator needs: gravity for the horizontal constraint on eaves,
/// north for azimuth. ARKit types are converted into this at the boundary (`ARKitBoundary.swift`).
public struct CameraOrientation: Sendable, Hashable, Codable {
    /// Unit vector pointing down (the direction gravity pulls), camera frame.
    public var gravity: Vec3
    /// Unit vector pointing to true north, camera frame. Horizontal (perpendicular to gravity).
    public var north: Vec3

    public init(gravity: Vec3, north: Vec3) {
        let g = gravity.normalized
        // remove any vertical component from north so the frame stays orthonormal
        let n = (north - g * north.dot(g)).normalized
        self.gravity = g
        self.north = n
    }

    public var up: Vec3 { -gravity }
    /// East = north × up (right-handed: x east, y up, z south).
    public var east: Vec3 { north.cross(up).normalized }

    /// Build from a camera-to-world rotation whose world frame is ARKit's gravity-and-heading
    /// aligned frame: y up, -z north, x east. `cameraToWorld` must already be in the OpenCV
    /// camera convention (x right, y down, z forward); see `ARKitBoundary.swift`.
    public init(cameraToWorld r: Mat3, headingDegrees: Double = 0) {
        let worldToCamera = r.transposed
        let g = worldToCamera * Vec3(0, -1, 0)
        // world -z is north when heading == 0; otherwise rotate about up by the heading
        let h = headingDegrees * .pi / 180
        let northWorld = Vec3(-sin(h), 0, -cos(h))
        self.init(gravity: g, north: worldToCamera * northWorld)
    }

    /// Pitch (angle to horizontal) and azimuth (0 = N, 90 = E, clockwise) of an upward normal.
    public func pitchAndAzimuth(ofNormal n: Vec3) -> (pitchDegrees: Double, azimuthDegrees: Double) {
        var normal = n.normalized
        if normal.dot(up) < 0 { normal = -normal }
        let pitch = normal.angleDegrees(to: up)
        let e = normal.dot(east)
        let no = normal.dot(north)
        var az = atan2(e, no) * 180 / .pi
        if az < 0 { az += 360 }
        if pitch < 1e-6 { az = 0 }  // flat roof: azimuth undefined
        return (pitch, az)
    }
}
