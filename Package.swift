// swift-tools-version: 5.10

import PackageDescription

let package = Package(
    name: "OpenAIPrivacyFilterCoreML",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .library(
            name: "PrivacyFilterTokenizer",
            targets: ["PrivacyFilterTokenizer"]
        )
    ],
    targets: [
        .target(
            name: "PrivacyFilterTokenizer"
        ),
        .testTarget(
            name: "PrivacyFilterTokenizerTests",
            dependencies: ["PrivacyFilterTokenizer"],
            path: "tests/PrivacyFilterTokenizerTests",
            resources: [
                .process("Resources")
            ]
        )
    ]
)
