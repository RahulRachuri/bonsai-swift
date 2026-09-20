// swift-tools-version: 6.0
//
// bonsai-swift: a native Swift host for the Bonsai 2 27B (ternary Qwen3.8) Core AI port.
//
// Build (Xcode 27 toolchain, macOS 27 SDK):
//     swift build -c release
//
// `CoreAI` is a system framework in the OS 27 SDKs, so there is no package dependency
// to resolve for the runtime; the targets just link it. The kernels that make ternary
// weights usable never appear here: Core AI bakes custom Metal source into the exported
// bundle, so the Swift side only ever sees a bundle. The one dependency is the tokenizer:
// swift-transformers reads the bundle's `tokenizer/` directory and renders the chat
// template, the same files the Python reference used.
import PackageDescription

let package = Package(
    name: "bonsai-swift",
    platforms: [.macOS("27.0"), .iOS("27.0")],
    // `BonsaiKit` is exported so an app (coreai-bench, for one) can depend on this package
    // directly instead of building the CLI.
    products: [
        .library(name: "BonsaiKit", targets: ["BonsaiKit"]),
        .executable(name: "bonsai-swift", targets: ["bonsai-swift"]),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.3"),
    ],
    targets: [
        .target(
            name: "BonsaiKit",
            dependencies: [
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            linkerSettings: [.linkedFramework("CoreAI")]
        ),
        .executableTarget(
            name: "bonsai-swift",
            dependencies: ["BonsaiKit"],
            linkerSettings: [.linkedFramework("CoreAI")]
        ),
    ]
)
