import AppKit
import Foundation
import Vision

struct FaceOut: Codable {
    let x: Double
    let y: Double
    let w: Double
    let h: Double
    let quality: Double
    let yaw: Double
    let emb_b64: String?
}

func computeEmbedding(cgImage: CGImage, x: Double, y: Double, w: Double, h: Double) -> String? {
    let cw = Double(cgImage.width)
    let ch = Double(cgImage.height)
    if cw <= 8 || ch <= 8 {
        return nil
    }
    let padX = 0.25 * w
    let padTop = 0.35 * h
    let padBot = 0.20 * h
    let cx0 = max(0.0, x - padX)
    let cy0 = max(0.0, y - padTop)
    let cx1 = min(1.0, x + w + padX)
    let cy1 = min(1.0, y + h + padBot)
    let rect = CGRect(
        x: cx0 * cw,
        y: cy0 * ch,
        width: max(8.0, (cx1 - cx0) * cw),
        height: max(8.0, (cy1 - cy0) * ch)
    ).integral
    guard let sub = cgImage.cropping(to: rect) else {
        return nil
    }
    let fpReq = VNGenerateImageFeaturePrintRequest()
    let handler = VNImageRequestHandler(cgImage: sub, options: [:])
    do {
        try handler.perform([fpReq])
        guard let fpObs = fpReq.results?.first as? VNFeaturePrintObservation else {
            return nil
        }
        let count = fpObs.elementCount
        if count <= 0 {
            return nil
        }
        var floats = [Float](repeating: 0, count: count)
        _ = floats.withUnsafeMutableBytes { fpObs.data.copyBytes(to: $0) }
        let sumSq = floats.reduce(0.0) { $0 + Double($1) * Double($1) }
        let norm = Float(sqrt(sumSq))
        if norm > 1e-6 {
            for idx in 0..<floats.count {
                floats[idx] /= norm
            }
        }
        let data = floats.withUnsafeBufferPointer { Data(buffer: $0) }
        return data.base64EncodedString()
    } catch {
        return nil
    }
}

var resultsMap: [String: [FaceOut]] = [:]

let paths = Array(CommandLine.arguments.dropFirst())
for path in paths {
    let url = URL(fileURLWithPath: path)
    let cgImg = NSImage(contentsOfFile: path)?.cgImage(forProposedRect: nil, context: nil, hints: nil)
    let req = VNDetectFaceCaptureQualityRequest()
    let handler = VNImageRequestHandler(url: url, options: [:])
    var boxes: [FaceOut] = []
    do {
        try handler.perform([req])
        for obs in req.results ?? [] {
            let b = obs.boundingBox
            let x = max(0.0, min(1.0, Double(b.origin.x)))
            let y = max(0.0, min(1.0, 1.0 - Double(b.origin.y + b.size.height)))
            let w = max(0.0, min(1.0 - x, Double(b.size.width)))
            let h = max(0.0, min(1.0 - y, Double(b.size.height)))
            let q = Double(obs.faceCaptureQuality ?? 0.5)
            let yaw = obs.yaw?.doubleValue ?? 0.0
            var emb: String? = nil
            if h >= 0.14 && w >= 0.07, let cg = cgImg {
                emb = computeEmbedding(cgImage: cg, x: x, y: y, w: w, h: h)
            }
            boxes.append(FaceOut(x: x, y: y, w: w, h: h, quality: q, yaw: yaw, emb_b64: emb))
        }
    } catch {
        boxes = []
    }
    resultsMap[path] = boxes
}

let encoder = JSONEncoder()
if let data = try? encoder.encode(resultsMap),
   let jsonStr = String(data: data, encoding: .utf8) {
    print(jsonStr)
} else {
    print("{}")
}
