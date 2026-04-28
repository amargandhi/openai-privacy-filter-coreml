import Foundation
import PrivacyFilterTokenizer
import XCTest

final class PrivacyFilterTokenizerTests: XCTestCase {
    func testTokenIDsAndOffsetsMatchPythonFixture() throws {
        let tokenizer = try PrivacyFilterBPETokenizer(tokenizerJSONURL: tokenizerJSONURL())
        let fixture = try loadFixture()

        for sample in fixture.samples {
            let encoding = try tokenizer.encode(
                sample.text,
                maxLength: fixture.maxLength,
                padding: true,
                truncation: true
            )
            XCTAssertEqual(encoding.inputIDs, sample.inputIDs, sample.id)
            XCTAssertEqual(encoding.attentionMask, sample.attentionMask, sample.id)
            XCTAssertEqual(encoding.offsets, sample.offsets, sample.id)
        }
    }

    private func loadFixture() throws -> TokenizerFixture {
        let url = Bundle.module.url(
            forResource: "tokenizer_baseline_128",
            withExtension: "json"
        )!
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(TokenizerFixture.self, from: data)
    }

    private func tokenizerJSONURL() throws -> URL {
        if let rawPath = ProcessInfo.processInfo.environment["OPENAI_PRIVACY_FILTER_TOKENIZER_JSON"] {
            let url = URL(fileURLWithPath: rawPath)
            if FileManager.default.fileExists(atPath: url.path) {
                return url
            }
        }

        let defaultPath = NSHomeDirectory()
            + "/.cache/huggingface/hub/models--openai--privacy-filter"
            + "/snapshots/7ffa9a043d54d1be65afb281eddf0ffbe629385b/tokenizer.json"
        if FileManager.default.fileExists(atPath: defaultPath) {
            return URL(fileURLWithPath: defaultPath)
        }

        throw XCTSkip(
            "Set OPENAI_PRIVACY_FILTER_TOKENIZER_JSON to the Hugging Face tokenizer.json path."
        )
    }
}

private struct TokenizerFixture: Decodable {
    let maxLength: Int
    let samples: [TokenizerSample]

    enum CodingKeys: String, CodingKey {
        case maxLength = "max_length"
        case samples
    }
}

private struct TokenizerSample: Decodable {
    let id: String
    let text: String
    let inputIDs: [Int]
    let attentionMask: [Int]
    let offsets: [TokenOffset]

    enum CodingKeys: String, CodingKey {
        case id
        case text
        case inputIDs = "input_ids"
        case attentionMask = "attention_mask"
        case offsets
    }
}
