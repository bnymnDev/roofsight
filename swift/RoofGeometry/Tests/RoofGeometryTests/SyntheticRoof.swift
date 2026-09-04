import Foundation
@testable import RoofGeometry

/// A gable roof plane with known pitch and azimuth, rendered to 2D through a known camera.
///
/// World frame: x east, y up, z south (ARKit gravity-and-heading). All lengths in meters.
struct SyntheticRoof {
    let pitchDegrees: Double
    let azimuthDegrees: Double
    /// Lower-left corner of the plane (start of the eave), world frame.
    let corner: Vec3
    let eaveLength: Double
    let slopeLength: Double

    // world-frame geometry
    var facing: Vec3 {
        let a = azimuthDegrees * .pi / 180
        return Vec3(sin(a), 0, -cos(a))  // horizontal unit vector at the azimuth
    }
    var up: Vec3 { Vec3(0, 1, 0) }
    var eaveDir: Vec3 { facing.cross(up).normalized }  // horizontal, along the eave
    var slopeDir: Vec3 {
        let p = pitchDegrees * .pi / 180
        return (-facing * cos(p) + up * sin(p)).normalized  // up the slope, away from the viewer
    }
    var normal: Vec3 {
        let p = pitchDegrees * .pi / 180
        return (up * cos(p) + facing * sin(p)).normalized
    }

    var eave: (Vec3, Vec3) { (corner, corner + eaveDir * eaveLength) }
    var ridge: (Vec3, Vec3) { (corner + slopeDir * slopeLength, corner + slopeDir * slopeLength + eaveDir * eaveLength) }
    var leftVerge: (Vec3, Vec3) { (corner, corner + slopeDir * slopeLength) }
    var rightVerge: (Vec3, Vec3) { (corner + eaveDir * eaveLength, corner + eaveDir * eaveLength + slopeDir * slopeLength) }

    func point(u: Double, v: Double) -> Vec3 { corner + eaveDir * (u * eaveLength) + slopeDir * (v * slopeLength) }
}

/// A camera at `eye` looking at `target`, OpenCV convention, with intrinsics.
struct SyntheticCamera {
    let eye: Vec3
    let cameraToWorld: Mat3
    let intrinsics: CameraIntrinsics

    init(eye: Vec3, target: Vec3, intrinsics: CameraIntrinsics) {
        self.eye = eye
        self.intrinsics = intrinsics
        let forward = (target - eye).normalized
        let worldUp = Vec3(0, 1, 0)
        let right = forward.cross(worldUp).normalized
        let down = forward.cross(right).normalized
        cameraToWorld = Mat3(columns: right, down, forward)
    }

    var orientation: CameraOrientation { CameraOrientation(cameraToWorld: cameraToWorld) }

    func toCamera(_ p: Vec3) -> Vec3 { cameraToWorld.transposed * (p - eye) }

    func project(_ p: Vec3) -> Vec2 { intrinsics.project(toCamera(p))! }

    func samples(_ seg: (Vec3, Vec3), count: Int = 12, noisePx: Double = 0, seed: UInt64 = 1) -> [Vec2] {
        var rng = SplitMix64(seed: seed)
        return (0..<count).map { i in
            let t = Double(i) / Double(count - 1)
            var p = project(seg.0 + (seg.1 - seg.0) * t)
            if noisePx > 0 {
                p.x += (rng.nextDouble() * 2 - 1) * noisePx
                p.y += (rng.nextDouble() * 2 - 1) * noisePx
            }
            return p
        }
    }

    func observation(of roof: SyntheticRoof, id: Int = 1, edges: [EdgeType] = [.eave, .ridge, .verge],
                     noisePx: Double = 0, withDepth: Bool = false, depthNoiseM: Double = 0) -> RoofPlaneObservation {
        var obs: [EdgeObservation] = []
        var seed: UInt64 = 7
        for t in edges {
            seed += 1
            switch t {
            case .eave: obs.append(EdgeObservation(type: .eave, points: samples(roof.eave, noisePx: noisePx, seed: seed)))
            case .ridge: obs.append(EdgeObservation(type: .ridge, points: samples(roof.ridge, noisePx: noisePx, seed: seed)))
            case .verge:
                obs.append(EdgeObservation(type: .verge, points: samples(roof.leftVerge, noisePx: noisePx, seed: seed)))
                obs.append(EdgeObservation(type: .verge, points: samples(roof.rightVerge, noisePx: noisePx, seed: seed + 100)))
            default: break
            }
        }
        let centroid = project(roof.point(u: 0.5, v: 0.5))
        var depth: DepthSamples?
        if withDepth {
            var rng = SplitMix64(seed: 99)
            var px: [Vec2] = [], d: [Double] = []
            for i in 0..<10 {
                for j in 0..<10 {
                    let p = roof.point(u: (Double(i) + 0.5) / 10, v: (Double(j) + 0.5) / 10)
                    let c = toCamera(p)
                    px.append(intrinsics.project(c)!)
                    d.append(c.z + (rng.nextDouble() * 2 - 1) * depthNoiseM)
                }
            }
            depth = DepthSamples(pixels: px, depths: d)
        }
        return RoofPlaneObservation(id: id, centroid: centroid, edges: obs, depth: depth)
    }
}

struct SplitMix64 {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    mutating func nextDouble() -> Double { Double(next() >> 11) / Double(1 << 53) }
}

let phoneIntrinsics = CameraIntrinsics(fx: 1500, fy: 1500, cx: 960, cy: 720, width: 1920, height: 1440)

func azimuthDiff(_ a: Double, _ b: Double) -> Double {
    let d = abs((a - b).truncatingRemainder(dividingBy: 360))
    return min(d, 360 - d)
}
