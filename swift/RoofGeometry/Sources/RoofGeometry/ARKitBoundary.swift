import Foundation

/// The only place that knows about ARKit conventions. Everything below it is plain structs.
///
/// ARKit camera frame: x right, y up, z backward (points in front have negative z).
/// RoofGeometry camera frame: x right, y down, z forward (the OpenCV/pinhole convention that
/// matches the pixel coordinates the model produces). Converting flips y and z.
public enum ARKitBoundary {
    /// Camera-to-world rotation of an ARKit camera transform (`ARCamera.transform`) as 16
    /// column-major floats, converted into the RoofGeometry camera convention.
    public static func cameraToWorld(columnMajor t: [Float]) -> Mat3 {
        precondition(t.count == 16, "expected a 4×4 column-major transform")
        // ARKit columns 0..2 are the camera axes in world coordinates
        let xAxis = Vec3(Double(t[0]), Double(t[1]), Double(t[2]))
        let yAxis = Vec3(Double(t[4]), Double(t[5]), Double(t[6]))
        let zAxis = Vec3(Double(t[8]), Double(t[9]), Double(t[10]))
        // OpenCV camera: x right (same), y down (-y), z forward (-z)
        return Mat3(columns: xAxis, -yAxis, -zAxis)
    }

    /// Orientation from an ARKit camera transform in a `.gravityAndHeading`-aligned session
    /// (world y up, -z true north). Pass `headingDegrees` from CoreLocation when the session
    /// is only `.gravity`-aligned.
    public static func orientation(cameraTransformColumnMajor t: [Float], headingDegrees: Double = 0) -> CameraOrientation {
        CameraOrientation(cameraToWorld: cameraToWorld(columnMajor: t), headingDegrees: headingDegrees)
    }

    /// Intrinsics from `ARCamera.intrinsics` (3×3 column-major) and `imageResolution`.
    public static func intrinsics(columnMajor k: [Float], width: Int, height: Int) -> CameraIntrinsics {
        precondition(k.count == 9, "expected a 3×3 column-major intrinsics matrix")
        return CameraIntrinsics(
            fx: Double(k[0]), fy: Double(k[4]), cx: Double(k[6]), cy: Double(k[7]),
            width: width, height: height
        )
    }
}

#if canImport(simd)
import simd

public extension ARKitBoundary {
    static func cameraToWorld(_ transform: simd_float4x4) -> Mat3 {
        let c = transform.columns
        return cameraToWorld(columnMajor: [
            c.0.x, c.0.y, c.0.z, c.0.w, c.1.x, c.1.y, c.1.z, c.1.w,
            c.2.x, c.2.y, c.2.z, c.2.w, c.3.x, c.3.y, c.3.z, c.3.w,
        ])
    }

    static func orientation(cameraTransform: simd_float4x4, headingDegrees: Double = 0) -> CameraOrientation {
        CameraOrientation(cameraToWorld: cameraToWorld(cameraTransform), headingDegrees: headingDegrees)
    }

    static func intrinsics(_ k: simd_float3x3, width: Int, height: Int) -> CameraIntrinsics {
        let c = k.columns
        return intrinsics(columnMajor: [c.0.x, c.0.y, c.0.z, c.1.x, c.1.y, c.1.z, c.2.x, c.2.y, c.2.z],
                          width: width, height: height)
    }
}
#endif
