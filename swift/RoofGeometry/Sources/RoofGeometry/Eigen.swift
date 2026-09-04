import Foundation

/// Smallest-eigenvalue eigenvector of a symmetric 3×3 matrix (cyclic Jacobi).
/// Used for plane fitting to depth samples. Small, dependency-free, plenty accurate here.
enum SymmetricEigen3 {
    static func smallestEigenvector(_ a: Mat3) -> Vec3 {
        var m = a
        var v = Mat3.identity
        for _ in 0..<50 {
            // find largest off-diagonal element
            var p = 0, q = 1
            var maxOff = 0.0
            for i in 0..<3 {
                for j in (i + 1)..<3 where abs(m[i, j]) > maxOff {
                    maxOff = abs(m[i, j])
                    p = i
                    q = j
                }
            }
            if maxOff < 1e-15 { break }
            let theta = (m[q, q] - m[p, p]) / (2 * m[p, q])
            let t = (theta >= 0 ? 1.0 : -1.0) / (abs(theta) + (theta * theta + 1).squareRoot())
            let c = 1 / (t * t + 1).squareRoot()
            let s = t * c
            var j = Mat3.identity
            j[p, p] = c
            j[q, q] = c
            j[p, q] = s
            j[q, p] = -s
            m = j.transposed * m * j
            v = v * j
        }
        var best = 0
        for i in 1..<3 where m[i, i] < m[best, best] { best = i }
        return v.column(best).normalized
    }
}
