import CoreAI
import Darwin
import Foundation

public enum BonsaiError: Error, CustomStringConvertible {
    case message(String)
    public var description: String {
        switch self { case .message(let m): return m }
    }
}

/// Seconds from a `Duration`, whole seconds included. Reading `.attoseconds` alone drops
/// everything past one second and once made a 3.6 s prefill look like 600 ms in the zoo.
@inline(__always)
func seconds(_ d: Duration) -> Double {
    let c = d.components
    return Double(c.seconds) + Double(c.attoseconds) / 1e18
}

// MARK: - NDArray helpers
//
// `NDArray` storage belongs to the runtime; every access goes through a short-lived view.
// The views are non-escapable, so each helper takes the view and finishes with it inside
// one call, which is also why none of these return a view.

func fillInt32(_ array: inout NDArray, _ values: some Sequence<Int32>) {
    let view = array.mutableView(as: Int32.self)
    view.withUnsafeMutablePointer { p, _, _ in
        var i = 0
        for v in values { p[i] = v; i += 1 }
    }
}

/// `position_ids` are the full `[0, total)` on every call: the graph indexes its caches
/// by position, so the prefix is always present, and only the tail beyond the processed
/// count is new work.
func fillPositions(_ array: inout NDArray, count: Int) {
    let view = array.mutableView(as: Int32.self)
    view.withUnsafeMutablePointer { p, _, _ in
        for i in 0..<count { p[i] = Int32(i) }
    }
}

func zeroFloat16(_ array: inout NDArray) {
    let count = array.shape.reduce(1, *)
    let view = array.mutableView(as: Float16.self)
    view.withUnsafeMutablePointer { p, _, _ in p.update(repeating: 0, count: count) }
}

/// First element of an int32 array (the graph's `next_token` is `[1,1]`).
func readInt32(_ array: NDArray) -> Int32 {
    array.view(as: Int32.self).withUnsafePointer { p, _, _ in p[0] }
}

/// One forward pass worth of fp16 logits, `[1, rows, vocab]`, still in the runtime's buffer.
///
/// The decode loop only ever needs the argmax of the last row, so nothing is widened or
/// copied unless a caller asks for `floats(row:)`. `top2` is what the parity gate reads:
/// the argmax plus its margin over the runner-up, which is how a miss gets classified as
/// a tie-band coin flip or a real divergence.
public struct Logits {
    let array: NDArray
    public let rows: Int
    public let vocab: Int

    public struct Top2 {
        public let argmax: Int32
        public let value: Float
        public let margin: Float
    }

    public func top2(row: Int) -> Top2 {
        precondition(row >= 0 && row < rows)
        let base = row * vocab
        let n = vocab
        return array.view(as: Float16.self).withUnsafePointer { p, _, _ in
            var best = 0
            var bestValue = Float(p[base])
            var second = -Float.infinity
            for i in 1..<n {
                let v = Float(p[base + i])
                if v > bestValue {
                    second = bestValue
                    bestValue = v
                    best = i
                } else if v > second {
                    second = v
                }
            }
            return Top2(argmax: Int32(best), value: bestValue, margin: bestValue - second)
        }
    }

    public func argmax(row: Int) -> Int32 { top2(row: row).argmax }

    public func floats(row: Int) -> [Float] {
        precondition(row >= 0 && row < rows)
        let base = row * vocab
        var out = [Float](repeating: 0, count: vocab)
        array.view(as: Float16.self).withUnsafePointer { p, _, _ in
            for i in 0..<vocab { out[i] = Float(p[base + i]) }
        }
        return out
    }

    /// Append one fp16 row to a scratch file without widening it or allocating a Swift
    /// array. Long self-check prompts have one 248,320-value row per token; retaining those
    /// rows as `[Float]` can consume about 1 MB per token and eventually force macOS to swap.
    func writeFloat16(row: Int, to file: Int32) throws {
        precondition(row >= 0 && row < rows)
        let base = row * vocab
        let byteCount = vocab * MemoryLayout<Float16>.stride
        var failure: Int32?
        array.view(as: Float16.self).withUnsafePointer { p, _, _ in
            let bytes = UnsafeRawPointer(p.advanced(by: base))
            var written = 0
            while written < byteCount {
                let n = Darwin.write(file, bytes.advanced(by: written), byteCount - written)
                if n < 0 {
                    if errno == EINTR { continue }
                    failure = errno
                    break
                }
                if n == 0 {
                    failure = EIO
                    break
                }
                written += n
            }
        }
        if let failure {
            throw BonsaiError.message("write self-check logits: \(String(cString: strerror(failure)))")
        }
    }

    /// Exact max absolute delta against one fp16 row in the self-check scratch file.
    /// `scratch` is reused for every position, keeping resident memory bounded by one row.
    func maxAbsDifference(row: Int, from file: Int32, offset: Int64,
                          scratch: inout [Float16]) throws -> Float {
        precondition(row >= 0 && row < rows)
        precondition(scratch.count == vocab)
        let byteCount = vocab * MemoryLayout<Float16>.stride
        var failure: Int32?
        scratch.withUnsafeMutableBytes { bytes in
            var read = 0
            while read < byteCount {
                let n = Darwin.pread(file, bytes.baseAddress!.advanced(by: read), byteCount - read,
                                     off_t(offset) + off_t(read))
                if n < 0 {
                    if errno == EINTR { continue }
                    failure = errno
                    break
                }
                if n == 0 {
                    failure = EIO
                    break
                }
                read += n
            }
        }
        if let failure {
            throw BonsaiError.message("read self-check logits: \(String(cString: strerror(failure)))")
        }

        let base = row * vocab
        return array.view(as: Float16.self).withUnsafePointer { p, _, _ in
            var worst: Float = 0
            for i in 0..<vocab {
                worst = max(worst, abs(Float(p[base + i]) - Float(scratch[i])))
            }
            return worst
        }
    }
}
