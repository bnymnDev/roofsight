# Roof geometry: pitch and azimuth from one photo

The `RoofGeometry` Swift package turns the model's masks plus the phone's pose into pitch and
azimuth per roof plane. Pure Swift, `Sendable`, no UIKit, no ARKit types below the boundary
file, so it is tested on synthetic roofs without a device.

## Input

| | |
|---|---|
| `RoofPlaneObservation` | plane id, mask centroid, the `roof_edge` instances touching it as pixel samples with their `edge_type`, optional `DepthSamples` |
| `CameraIntrinsics` | fx, fy, cx, cy in pixels, image size |
| `CameraOrientation` | gravity (down) and true north as unit vectors in the camera frame |

`ARKitBoundary` builds the last two from `ARCamera.intrinsics`, `ARCamera.transform` and a
heading. Camera frame inside the package: x right, y down, z forward (the pixel convention of
the model). ARKit's y-up, z-backward camera is flipped at the boundary.

## Output

```swift
struct RoofPlaneEstimate {
  let id: Int
  let pitchDegrees: Double
  let azimuthDegrees: Double        // 0 = N, 90 = E, the direction the plane faces
  let confidence: Double            // 0…1
  let eaveLine: Line3D?             // camera frame; metric only with depth
  let ridgeLine: Line3D?
  let method: Method                // .lines or .depth
}
```

## Method v0.1

1. **Line fits.** Total-least-squares line through each edge's pixel samples.
2. **Eave direction.** Every 3D line that projects onto an image line lies in the plane
   through the camera center and that line (its interpretation plane). An eave is horizontal,
   so its 3D direction is the one vector perpendicular to both gravity and that plane's normal.
   A ridge works the same way and substitutes when the eave is hidden.
3. **Slope edge and normal.** A verge runs straight up the slope, perpendicular to the eave
   in plan; a hip bisects the corner at 45°. That fixes the horizontal part of the slope
   direction; its vertical part is what makes it lie in its own interpretation plane. The
   plane normal is eave × slope. Of the two mirror solutions, the one whose normal faces the
   camera is the visible face. Pitch is the angle between the normal and up; azimuth is the
   compass bearing of the normal's horizontal part.
4. **Depth.** With LiDAR, a plane is fitted to the depth samples inside the mask
   (smallest eigenvector of the covariance) and step 3 is not needed. Eave and ridge become
   metric lines: their interpretation planes intersected with the fitted plane.

Confidence is a heuristic: depth fits score by their RMS residual; line fits by the shorter of
the two edges and the line-fit residual, capped at 0.85.

## Accuracy on synthetic roofs

`swift test` renders gable and hip roofs of known pitch and azimuth through a phone camera
(fx 1500 px, 1920×1440) and checks the estimates:

| Input | Pitch error | Azimuth error |
|---|---|---|
| exact edge samples | < 0.05° | < 0.05° |
| ± 2 px uniform noise on the samples | < 2° | < 2° |
| depth samples, 1 cm noise, ± 3 px on edges | < 0.5° | < 0.5° |

Real-world error will be dominated by mask quality and the heading sensor; that number is the
pitch/azimuth MAE on the own-photo subset in the [benchmark](benchmark.md).

## Using it

```swift
import RoofGeometry

let intrinsics = ARKitBoundary.intrinsics(frame.camera.intrinsics,
                                          width: Int(frame.camera.imageResolution.width),
                                          height: Int(frame.camera.imageResolution.height))
let orientation = ARKitBoundary.orientation(cameraTransform: frame.camera.transform)
let estimator = RoofGeometryEstimator(intrinsics: intrinsics, orientation: orientation)

let plane = RoofPlaneObservation(
    id: 1, centroid: maskCentroid,
    edges: [EdgeObservation(type: .eave, points: eavePixels),
            EdgeObservation(type: .verge, points: vergePixels)],
    depth: lidarSamples)
let estimate = try estimator.estimate(plane)
print(estimate.pitchDegrees, estimate.azimuthDegrees, estimate.confidence)
```

Not in v0.1: multi-view refinement, valleys as slope edges of the neighbouring plane, and roofs
without any straight edge (thatched, curved).
