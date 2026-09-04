import Foundation

/// A 2D point or vector in image pixels. Origin top-left, x right, y down.
public struct Vec2: Sendable, Hashable, Codable {
    public var x: Double
    public var y: Double

    public init(_ x: Double, _ y: Double) {
        self.x = x
        self.y = y
    }

    public static func + (a: Vec2, b: Vec2) -> Vec2 { Vec2(a.x + b.x, a.y + b.y) }
    public static func - (a: Vec2, b: Vec2) -> Vec2 { Vec2(a.x - b.x, a.y - b.y) }
    public static func * (a: Vec2, s: Double) -> Vec2 { Vec2(a.x * s, a.y * s) }
    public var length: Double { (x * x + y * y).squareRoot() }
}

/// A 3D vector. The frame is stated wherever a `Vec3` is used.
public struct Vec3: Sendable, Hashable, Codable {
    public var x: Double
    public var y: Double
    public var z: Double

    public init(_ x: Double, _ y: Double, _ z: Double) {
        self.x = x
        self.y = y
        self.z = z
    }

    public static let zero = Vec3(0, 0, 0)

    public static func + (a: Vec3, b: Vec3) -> Vec3 { Vec3(a.x + b.x, a.y + b.y, a.z + b.z) }
    public static func - (a: Vec3, b: Vec3) -> Vec3 { Vec3(a.x - b.x, a.y - b.y, a.z - b.z) }
    public static func * (a: Vec3, s: Double) -> Vec3 { Vec3(a.x * s, a.y * s, a.z * s) }
    public static prefix func - (a: Vec3) -> Vec3 { Vec3(-a.x, -a.y, -a.z) }

    public func dot(_ b: Vec3) -> Double { x * b.x + y * b.y + z * b.z }

    public func cross(_ b: Vec3) -> Vec3 {
        Vec3(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x)
    }

    public var length: Double { dot(self).squareRoot() }

    public var normalized: Vec3 {
        let l = length
        return l > 0 ? self * (1 / l) : self
    }

    /// Angle to another vector in degrees, in `[0, 180]`.
    public func angleDegrees(to b: Vec3) -> Double {
        let c = max(-1, min(1, normalized.dot(b.normalized)))
        return acos(c) * 180 / .pi
    }
}

/// A 3×3 matrix stored row-major.
public struct Mat3: Sendable, Hashable, Codable {
    public var m: [Double]  // 9 entries, row-major

    public init(rows r0: Vec3, _ r1: Vec3, _ r2: Vec3) {
        m = [r0.x, r0.y, r0.z, r1.x, r1.y, r1.z, r2.x, r2.y, r2.z]
    }

    public init(columns c0: Vec3, _ c1: Vec3, _ c2: Vec3) {
        m = [c0.x, c1.x, c2.x, c0.y, c1.y, c2.y, c0.z, c1.z, c2.z]
    }

    public static let identity = Mat3(rows: Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1))

    public subscript(row: Int, col: Int) -> Double {
        get { m[row * 3 + col] }
        set { m[row * 3 + col] = newValue }
    }

    public var transposed: Mat3 {
        Mat3(rows: column(0), column(1), column(2))
    }

    public func column(_ c: Int) -> Vec3 { Vec3(m[c], m[3 + c], m[6 + c]) }
    public func row(_ r: Int) -> Vec3 { Vec3(m[r * 3], m[r * 3 + 1], m[r * 3 + 2]) }

    public static func * (a: Mat3, v: Vec3) -> Vec3 {
        Vec3(a.row(0).dot(v), a.row(1).dot(v), a.row(2).dot(v))
    }

    public static func * (a: Mat3, b: Mat3) -> Mat3 {
        Mat3(columns: a * b.column(0), a * b.column(1), a * b.column(2))
    }
}
