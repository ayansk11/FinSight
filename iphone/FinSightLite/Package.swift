// swift-tools-version: 5.9
// Package.swift — FinSight Lite SPM dependencies

import PackageDescription

let package = Package(
    name: "FinSightLite",
    platforms: [
        .iOS(.v17),
    ],
    products: [
        .library(
            name: "FinSightLite",
            targets: ["FinSightLite"]
        ),
    ],
    dependencies: [
        // MLX Swift — Apple Silicon-optimized inference
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.18.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift-examples.git", from: "1.15.0"),
    ],
    targets: [
        .target(
            name: "FinSightLite",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-examples"),
                .product(name: "MLXLMCommon", package: "mlx-swift-examples"),
            ],
            path: "FinSightLite"
        ),
    ]
)
