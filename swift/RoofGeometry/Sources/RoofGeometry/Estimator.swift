import Foundation

/// The result per roof plane, as in SPEC.md.
public struct RoofPlaneEstimate: Sendable, Hashable, Codable {
    public let id: Int
    public let pitchDegrees: Double
    /// 0 = N, 90 = E, clockwise. The direction the roof plane faces (its normal, in plan).
    public let azimuthDegrees: Double
    /// 0…1. Depth fits score higher than single-view line geometry; long, straight edges
    /// score higher than short, noisy ones.
    public let confidence: Double
    /// Camera frame (x right, y down, z forward). Metric only if depth was available.
    public let eaveLine: Line3D?
    public let ridgeLine: Line3D?
    public let method: Method

    public enum Method: String, Sendable, Codable {
        /// Eave + slope edge + gravity (SPEC method v0.1, steps 1–3).
        case lines
        /// Plane fitted to LiDAR depth inside the mask (step 4).
        case depth
    }
}

public enum RoofGeometryError: Error, Sendable, Equatable {
    case noEave
    case noSlopeEdge
    case degenerateEdge(EdgeType)
    case tooFewDepthSamples(Int)
}

/// Pitch and azimuth per roof plane from masks, intrinsics and device orientation.
///
/// Pure and `Sendable`. Everything is in the camera frame; `CameraOrientation` supplies gravity
/// and north so no world pose is needed.
public struct RoofGeometryEstimator: Sendable {
    public var intrinsics: CameraIntrinsics
    public var orientation: CameraOrientation
    public var minDepthSamples: Int = 30
    /// Below this many pixels an edge is too short to trust.
    public var shortEdgePx: Double = 40

    public init(intrinsics: CameraIntrinsics, orientation: CameraOrientation) {
        self.intrinsics = intrinsics
        self.orientation = orientation
    }

    public func estimate(_ plane: RoofPlaneObservation) throws -> RoofPlaneEstimate {
        if let depth = plane.depth, depth.pixels.count >= minDepthSamples {
            return try estimateFromDepth(plane, depth)
        }
        return try estimateFromLines(plane)
    }

    public func estimate(_ planes: [RoofPlaneObservation]) -> [Result<RoofPlaneEstimate, RoofGeometryError>] {
        planes.map { p in
            do { return .success(try estimate(p)) }
            catch let e as RoofGeometryError { return .failure(e) }
            catch { return .failure(.noEave) }
        }
    }

    // MARK: - Lines (SPEC steps 1–3)

    /// Normal of the plane through the camera center and an image line (its "interpretation
    /// plane"). Every 3D line that projects onto this image line lies in that plane.
    func interpretationNormal(_ line: Line2D) -> Vec3 {
        let r1 = intrinsics.ray(through: line.start)
        let r2 = intrinsics.ray(through: line.end)
        return r1.cross(r2).normalized
    }

    /// 3D direction of a horizontal edge (eave/ridge): perpendicular to gravity and inside the
    /// interpretation plane. The sign is chosen so that the direction projects onto the image
    /// the same way the fitted 2D line runs (start → end).
    func horizontalDirection(_ line: Line2D) -> Vec3 {
        let d = interpretationNormal(line).cross(orientation.gravity).normalized
        let img = imageDirection(of: d, at: line.point)
        return (img.x * line.direction.x + img.y * line.direction.y) >= 0 ? d : -d
    }

    /// Image-plane direction in which a point moves when displaced along a 3D direction.
    func imageDirection(of d: Vec3, at p: Vec2) -> Vec2 {
        let r = intrinsics.ray(through: p)
        // derivative of the pinhole projection of r + t·d at t = 0
        let z = r.z
        return Vec2(intrinsics.fx * (d.x * z - r.x * d.z) / (z * z),
                    intrinsics.fy * (d.y * z - r.y * d.z) / (z * z))
    }

    /// Plan angle between a slope edge and the eave, measured from the eave's start→end
    /// direction. Verges are perpendicular either way. A hip bisects a corner: 45° when it
    /// leaves the eave's start corner heading along the eave, 135° from the far corner.
    func planAngle(of slope: Line2D, type: EdgeType, eave: Line2D) -> Double? {
        guard let base = type.planAngleToEaveDegrees else { return nil }
        if type != .hip { return base }
        let dStart = perpendicularDistance(slope.start, to: eave)
        let dEnd = perpendicularDistance(slope.end, to: eave)
        let corner = dStart <= dEnd ? slope.start : slope.end
        let far = dStart <= dEnd ? slope.end : slope.start
        let inward = far - corner
        let along = inward.x * eave.direction.x + inward.y * eave.direction.y
        return along >= 0 ? 45 : 135
    }

    func perpendicularDistance(_ p: Vec2, to line: Line2D) -> Double {
        let r = p - line.point
        return abs(r.x * line.direction.y - r.y * line.direction.x)
    }

    /// 3D direction of a slope edge whose plan angle to the eave is known: its horizontal part
    /// is fixed by the eave and the angle; its vertical part by the interpretation plane.
    func slopeDirection(_ line: Line2D, eave: Vec3, planAngleDegrees: Double, sign: Double) -> Vec3? {
        let g = orientation.gravity
        let a = planAngleDegrees * .pi / 180
        let perp = g.cross(eave).normalized * sign
        let h = (eave * cos(a) + perp * sin(a)).normalized
        let n = interpretationNormal(line)
        let gn = g.dot(n)
        guard abs(gn) > 1e-9 else { return nil }  // edge seen exactly edge-on to gravity
        let beta = -h.dot(n) / gn
        return (h + g * beta).normalized
    }

    func estimateFromLines(_ plane: RoofPlaneObservation) throws -> RoofPlaneEstimate {
        let fitted: [(EdgeType, Line2D)] = plane.edges.compactMap { e in
            guard let l = Line2D.fit(e.points) else { return nil }
            return (e.type, l)
        }
        guard let (_, eaveLine) = longest(fitted, where: { $0 == .eave })
            ?? longest(fitted, where: { $0 == .ridge })
        else { throw RoofGeometryError.noEave }
        guard eaveLine.length > 1 else { throw RoofGeometryError.degenerateEdge(.eave) }
        let eaveDir = horizontalDirection(eaveLine)

        guard let (slopeType, slopeLine) = longest(fitted, where: { $0.planAngleToEaveDegrees != nil }),
              let planAngle = planAngle(of: slopeLine, type: slopeType, eave: eaveLine)
        else { throw RoofGeometryError.noSlopeEdge }
        guard slopeLine.length > 1 else { throw RoofGeometryError.degenerateEdge(slopeType) }

        // Two mirror solutions (which side of the eave the plane rises on). The visible face's
        // normal points toward the camera: pick the sign whose normal opposes the view ray.
        let viewRay = intrinsics.ray(through: plane.centroid)
        var best: (normal: Vec3, score: Double)?
        for sign in [1.0, -1.0] {
            guard let s = slopeDirection(slopeLine, eave: eaveDir, planAngleDegrees: planAngle, sign: sign)
            else { continue }
            var n = eaveDir.cross(s).normalized
            if n.dot(orientation.up) < 0 { n = -n }
            let facing = -n.dot(viewRay)
            if best == nil || facing > best!.score { best = (n, facing) }
        }
        guard let normal = best?.normal else { throw RoofGeometryError.degenerateEdge(slopeType) }

        let (pitch, az) = orientation.pitchAndAzimuth(ofNormal: normal)
        let ridge = longest(fitted, where: { $0 == .ridge })?.1
        let confidence = lineConfidence(eave: eaveLine, slope: slopeLine)
        return RoofPlaneEstimate(
            id: plane.id, pitchDegrees: pitch, azimuthDegrees: az, confidence: confidence,
            eaveLine: Line3D(point: intrinsics.ray(through: eaveLine.point), direction: eaveDir, isMetric: false),
            ridgeLine: ridge.map { Line3D(point: intrinsics.ray(through: $0.point), direction: horizontalDirection($0), isMetric: false) },
            method: .lines
        )
    }

    func longest(_ fitted: [(EdgeType, Line2D)], where pred: (EdgeType) -> Bool) -> (EdgeType, Line2D)? {
        fitted.filter { pred($0.0) }.max { $0.1.length < $1.1.length }
    }

    func lineConfidence(eave: Line2D, slope: Line2D) -> Double {
        let lengthScore = min(1, min(eave.length, slope.length) / (2 * shortEdgePx))
        let residualScore = max(0, 1 - max(eave.residual, slope.residual) / 4)
        return max(0.05, min(0.85, lengthScore * residualScore))
    }

    // MARK: - Depth (SPEC step 4)

    func estimateFromDepth(_ plane: RoofPlaneObservation, _ depth: DepthSamples) throws -> RoofPlaneEstimate {
        let pts = zip(depth.pixels, depth.depths)
            .filter { $0.1.isFinite && $0.1 > 0 }
            .map { intrinsics.unproject($0.0, depth: $0.1) }
        guard pts.count >= minDepthSamples else { throw RoofGeometryError.tooFewDepthSamples(pts.count) }
        let n = Double(pts.count)
        let c = pts.reduce(Vec3.zero, +) * (1 / n)
        var xx = 0.0, xy = 0.0, xz = 0.0, yy = 0.0, yz = 0.0, zz = 0.0
        for p in pts {
            let d = p - c
            xx += d.x * d.x; xy += d.x * d.y; xz += d.x * d.z
            yy += d.y * d.y; yz += d.y * d.z; zz += d.z * d.z
        }
        let cov = Mat3(rows: Vec3(xx, xy, xz), Vec3(xy, yy, yz), Vec3(xz, yz, zz))
        var normal = SymmetricEigen3.smallestEigenvector(cov)
        if normal.dot(c) > 0 { normal = -normal }  // face the camera
        let (pitch, az) = orientation.pitchAndAzimuth(ofNormal: normal)

        // residual → confidence
        var sq = 0.0
        for p in pts { let d = (p - c).dot(normal); sq += d * d }
        let rms = (sq / n).squareRoot()
        let confidence = max(0.3, min(0.99, 1 - rms / 0.1))

        // metric eave/ridge lines if the mask came with edges: intersect their viewing planes
        // with the fitted plane
        let eave = metricLine(plane, type: .eave, planePoint: c, planeNormal: normal)
        let ridge = metricLine(plane, type: .ridge, planePoint: c, planeNormal: normal)
        return RoofPlaneEstimate(
            id: plane.id, pitchDegrees: pitch, azimuthDegrees: az, confidence: confidence,
            eaveLine: eave, ridgeLine: ridge, method: .depth
        )
    }

    func metricLine(_ plane: RoofPlaneObservation, type: EdgeType, planePoint: Vec3, planeNormal: Vec3) -> Line3D? {
        guard let e = plane.edges.first(where: { $0.type == type }), let l = Line2D.fit(e.points) else { return nil }
        let dir = interpretationNormal(l).cross(planeNormal).normalized
        let ray = intrinsics.ray(through: l.point)
        let denom = ray.dot(planeNormal)
        guard abs(denom) > 1e-9 else { return nil }
        let t = planePoint.dot(planeNormal) / denom
        guard t > 0 else { return nil }
        return Line3D(point: ray * t, direction: dir, isMetric: true)
    }
}
