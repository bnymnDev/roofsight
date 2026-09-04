// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "RoofGeometry",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "RoofGeometry", targets: ["RoofGeometry"]),
    ],
    targets: [
        .target(
            name: "RoofGeometry",
            swiftSettings: [.enableExperimentalFeature("StrictConcurrency")]
        ),
        .testTarget(name: "RoofGeometryTests", dependencies: ["RoofGeometry"]),
    ]
)
