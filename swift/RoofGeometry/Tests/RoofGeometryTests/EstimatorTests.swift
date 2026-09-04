import XCTest
@testable import RoofGeometry

final class EstimatorTests: XCTestCase {
    /// A camera on the street, 12 m in front of a house, looking up at the roof plane that faces it.
    func scene(pitch: Double, azimuth: Double, cameraOffsetDegrees: Double = 0, distance: Double = 12) -> (SyntheticRoof, SyntheticCamera) {
        let roof = SyntheticRoof(pitchDegrees: pitch, azimuthDegrees: azimuth, corner: Vec3(-4, 5, 0),
                                 eaveLength: 8, slopeLength: 5)
        // rotate the whole house so its plane faces `azimuth`: the corner is defined for facing
        // south (azimuth 180); place the camera on the facing side, slightly off-axis
        let center = roof.point(u: 0.5, v: 0.5)
        let a = (azimuth + cameraOffsetDegrees) * .pi / 180
        let eye = Vec3(center.x + sin(a) * distance, 1.6, center.z - cos(a) * distance)
        let cam = SyntheticCamera(eye: eye, target: center, intrinsics: phoneIntrinsics)
        return (roof, cam)
    }

    func testExactLinesRecoverPitchAndAzimuth() throws {
        for (pitch, az) in [(35.0, 180.0), (45.0, 90.0), (22.0, 270.0), (30.0, 15.0), (60.0, 200.0)] {
            let (roof, cam) = scene(pitch: pitch, azimuth: az, cameraOffsetDegrees: 20)
            let est = RoofGeometryEstimator(intrinsics: cam.intrinsics, orientation: cam.orientation)
            let r = try est.estimate(cam.observation(of: roof))
            XCTAssertEqual(r.pitchDegrees, pitch, accuracy: 0.05, "pitch @ \(pitch)/\(az)")
            XCTAssertLessThan(azimuthDiff(r.azimuthDegrees, az), 0.05, "azimuth @ \(pitch)/\(az): \(r.azimuthDegrees)")
            XCTAssertEqual(r.method, .lines)
            XCTAssertNotNil(r.eaveLine)
            XCTAssertNotNil(r.ridgeLine)
            XCTAssertFalse(r.eaveLine!.isMetric)
        }
    }

    func testNoisyLinesStayWithinTwoDegrees() throws {
        let (roof, cam) = scene(pitch: 38, azimuth: 160, cameraOffsetDegrees: 25)
        let est = RoofGeometryEstimator(intrinsics: cam.intrinsics, orientation: cam.orientation)
        let r = try est.estimate(cam.observation(of: roof, noisePx: 2))
        XCTAssertEqual(r.pitchDegrees, 38, accuracy: 2)
        XCTAssertLessThan(azimuthDiff(r.azimuthDegrees, 160), 2)
        XCTAssertGreaterThan(r.confidence, 0.3)
        XCTAssertLessThan(r.confidence, 0.9)
    }

    func testHipRoofFromEitherCorner() throws {
        // A regular hip roof: the hip leaves each eave corner at 45° in plan and climbs at
        // tan(pitch)/√2 per unit of plan length. Both corners must give the same plane.
        for (pitch, az, offset) in [(30.0, 180.0, 15.0), (42.0, 95.0, -20.0)] {
            let (roof, cam) = scene(pitch: pitch, azimuth: az, cameraOffsetDegrees: offset)
            let rise = tan(pitch * .pi / 180) / 2.0.squareRoot()
            let (c0, c1) = roof.eave
            let leftPlan = (-roof.facing + roof.eaveDir).normalized
            let rightPlan = (-roof.facing - roof.eaveDir).normalized
            let hips = [(c0, c0 + (leftPlan + roof.up * rise).normalized * 5),
                        (c1, c1 + (rightPlan + roof.up * rise).normalized * 5)]
            for hip in hips {
                let obs = RoofPlaneObservation(
                    id: 1, centroid: cam.project(roof.point(u: 0.5, v: 0.5)),
                    edges: [EdgeObservation(type: .eave, points: cam.samples(roof.eave)),
                            EdgeObservation(type: .hip, points: cam.samples(hip))]
                )
                let est = RoofGeometryEstimator(intrinsics: cam.intrinsics, orientation: cam.orientation)
                let r = try est.estimate(obs)
                XCTAssertEqual(r.pitchDegrees, pitch, accuracy: 0.1)
                XCTAssertLessThan(azimuthDiff(r.azimuthDegrees, az), 0.1)
            }
        }
    }

    func testRidgeSubstitutesForMissingEave() throws {
        let (roof, cam) = scene(pitch: 40, azimuth: 120, cameraOffsetDegrees: 10)
        let est = RoofGeometryEstimator(intrinsics: cam.intrinsics, orientation: cam.orientation)
        let r = try est.estimate(cam.observation(of: roof, edges: [.ridge, .verge]))
        XCTAssertEqual(r.pitchDegrees, 40, accuracy: 0.05)
        XCTAssertLessThan(azimuthDiff(r.azimuthDegrees, 120), 0.05)
    }

    func testErrorsAreTyped() {
        let (roof, cam) = scene(pitch: 40, azimuth: 120)
        let est = RoofGeometryEstimator(intrinsics: cam.intrinsics, orientation: cam.orientation)
        XCTAssertThrowsError(try est.estimate(cam.observation(of: roof, edges: [.verge]))) {
            XCTAssertEqual($0 as? RoofGeometryError, .noEave)
        }
        XCTAssertThrowsError(try est.estimate(cam.observation(of: roof, edges: [.eave]))) {
            XCTAssertEqual($0 as? RoofGeometryError, .noSlopeEdge)
        }
        let results = est.estimate([cam.observation(of: roof), cam.observation(of: roof, edges: [.eave])])
        XCTAssertEqual(results.count, 2)
        if case .failure(let e) = results[1] { XCTAssertEqual(e, .noSlopeEdge) } else { XCTFail() }
    }

    func testDepthFitBeatsLinesAndIsMetric() throws {
        let (roof, cam) = scene(pitch: 33, azimuth: 210, cameraOffsetDegrees: 30)
        let est = RoofGeometryEstimator(intrinsics: cam.intrinsics, orientation: cam.orientation)
        let r = try est.estimate(cam.observation(of: roof, noisePx: 3, withDepth: true, depthNoiseM: 0.01))
        XCTAssertEqual(r.method, .depth)
        XCTAssertEqual(r.pitchDegrees, 33, accuracy: 0.5)
        XCTAssertLessThan(azimuthDiff(r.azimuthDegrees, 210), 0.5)
        XCTAssertGreaterThan(r.confidence, 0.8)
        let eave = try XCTUnwrap(r.eaveLine)
        XCTAssertTrue(eave.isMetric)
        // the metric eave passes through the true eave midpoint (camera frame)
        let trueMid = cam.toCamera(roof.point(u: 0.5, v: 0))
        let offset = (trueMid - eave.point)
        let perp = (offset - eave.direction * offset.dot(eave.direction)).length
        XCTAssertLessThan(perp, 0.05)
        let trueDir = cam.cameraToWorld.transposed * roof.eaveDir
        let angle = eave.direction.angleDegrees(to: trueDir)
        XCTAssertLessThan(min(angle, 180 - angle), 0.5)
    }

    func testTooFewDepthSamplesFallsBackToLines() throws {
        let (roof, cam) = scene(pitch: 33, azimuth: 210)
        var est = RoofGeometryEstimator(intrinsics: cam.intrinsics, orientation: cam.orientation)
        est.minDepthSamples = 500
        let r = try est.estimate(cam.observation(of: roof, withDepth: true))
        XCTAssertEqual(r.method, .lines)
    }

    func testFlatRoofHasZeroPitch() {
        let o = CameraOrientation(gravity: Vec3(0, 1, 0), north: Vec3(0, 0, 1))
        let (p, a) = o.pitchAndAzimuth(ofNormal: Vec3(0, -1, 0))
        XCTAssertEqual(p, 0, accuracy: 1e-9)
        XCTAssertEqual(a, 0)
    }
}

final class LineFitTests: XCTestCase {
    func testFitRecoversDirectionAndExtent() throws {
        let pts = (0..<10).map { Vec2(Double($0) * 3, Double($0) * 4 + 1) }
        let l = try XCTUnwrap(Line2D.fit(pts))
        XCTAssertEqual(abs(l.direction.x), 0.6, accuracy: 1e-9)
        XCTAssertEqual(abs(l.direction.y), 0.8, accuracy: 1e-9)
        XCTAssertEqual(l.length, 45, accuracy: 1e-9)
        XCTAssertEqual(l.residual, 0, accuracy: 1e-9)
    }

    func testDegenerate() {
        XCTAssertNil(Line2D.fit([Vec2(1, 1)]))
        XCTAssertNil(Line2D.fit([Vec2(1, 1), Vec2(1, 1)]))
    }
}

final class BoundaryTests: XCTestCase {
    func testARKitTransformConversion() {
        // ARKit camera looking north (-z), level, y up: identity transform
        var t = [Float](repeating: 0, count: 16)
        t[0] = 1; t[5] = 1; t[10] = 1; t[15] = 1
        let o = ARKitBoundary.orientation(cameraTransformColumnMajor: t)
        // OpenCV camera: down is +y, forward (+z) is north
        XCTAssertEqual(o.gravity.y, 1, accuracy: 1e-9)
        XCTAssertEqual(o.north.z, 1, accuracy: 1e-9)
        XCTAssertEqual(o.east.x, 1, accuracy: 1e-9)
        let o90 = ARKitBoundary.orientation(cameraTransformColumnMajor: t, headingDegrees: 90)
        // heading 90 → north is 90° to the left of forward, i.e. -x in camera
        XCTAssertEqual(o90.north.x, -1, accuracy: 1e-9)
    }

    func testIntrinsicsConversion() {
        let k = ARKitBoundary.intrinsics(columnMajor: [1500, 0, 0, 0, 1500, 0, 960, 720, 1], width: 1920, height: 1440)
        XCTAssertEqual(k.fx, 1500)
        XCTAssertEqual(k.cx, 960)
        XCTAssertEqual(k.cy, 720)
        let p = Vec2(960, 720)
        XCTAssertEqual(k.ray(through: p).z, 1, accuracy: 1e-12)
        XCTAssertEqual(k.project(k.unproject(Vec2(100, 200), depth: 3))!.x, 100, accuracy: 1e-9)
    }

    func testEigenSmallest() {
        let m = Mat3(rows: Vec3(4, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 9))
        let v = SymmetricEigen3.smallestEigenvector(m)
        XCTAssertEqual(abs(v.y), 1, accuracy: 1e-9)
    }
}
