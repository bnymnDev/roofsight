import Foundation

/// A fitted 2D line: a point on it and a unit direction, plus the extent of the samples.
public struct Line2D: Sendable, Hashable, Codable {
    public var point: Vec2
    public var direction: Vec2
    /// Endpoints of the samples projected onto the line, in sample order along the line.
    public var start: Vec2
    public var end: Vec2
    /// RMS perpendicular distance of the samples to the line, pixels.
    public var residual: Double

    public var length: Double { (end - start).length }

    /// Least-squares (total least squares / PCA) fit. Needs at least two distinct points.
    public static func fit(_ points: [Vec2]) -> Line2D? {
        guard points.count >= 2 else { return nil }
        let n = Double(points.count)
        let cx = points.reduce(0) { $0 + $1.x } / n
        let cy = points.reduce(0) { $0 + $1.y } / n
        var sxx = 0.0, sxy = 0.0, syy = 0.0
        for p in points {
            let dx = p.x - cx, dy = p.y - cy
            sxx += dx * dx
            sxy += dx * dy
            syy += dy * dy
        }
        // principal eigenvector of the 2×2 covariance
        let theta = 0.5 * atan2(2 * sxy, sxx - syy)
        let d = Vec2(cos(theta), sin(theta))
        guard sxx + syy > 1e-12 else { return nil }
        let c = Vec2(cx, cy)
        var tMin = Double.infinity, tMax = -Double.infinity
        var sq = 0.0
        for p in points {
            let r = p - c
            let t = r.x * d.x + r.y * d.y
            let perp = r.x * d.y - r.y * d.x
            sq += perp * perp
            tMin = min(tMin, t)
            tMax = max(tMax, t)
        }
        return Line2D(
            point: c, direction: d, start: c + d * tMin, end: c + d * tMax,
            residual: (sq / n).squareRoot()
        )
    }
}

/// A 3D line: a point and a unit direction. The frame is stated where it is produced.
public struct Line3D: Sendable, Hashable, Codable {
    public var point: Vec3
    public var direction: Vec3
    /// False when only the direction is metric and `point` is at an arbitrary distance along
    /// the viewing ray (no depth available).
    public var isMetric: Bool

    public init(point: Vec3, direction: Vec3, isMetric: Bool) {
        self.point = point
        self.direction = direction.normalized
        self.isMetric = isMetric
    }
}
