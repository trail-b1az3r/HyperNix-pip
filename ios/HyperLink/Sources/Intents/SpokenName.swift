//  SpokenName.swift
//  HyperLink — the name Siri should listen for.
//
//  Siri matches a spoken sentence against the titles an app publishes
//  for its entities. A model's id is a file name --
//  `blazeindustries/gemma-4-e2b-it-Q4_K_M.gguf` -- and nobody says
//  "gemma dash four dash e two b dash it dash Q four underscore K
//  underscore M". Publishing that as the title meant "Load Gemma 4 E2B
//  in HyperLink" had nothing to match, and Siri answered that HyperLink
//  had not added support for it.
//
//  So the title is the name as it is said: the quantisation and file
//  format dropped (they describe the file, not the model), the tuning
//  marker dropped ("it", "instruct" and "chat" are left out when people
//  say a name), and separators turned into spaces. The fuller forms go
//  in as synonyms, so either is heard.

import Foundation

enum SpokenName {
    /// A quantisation or format at the end of a file name: `-Q4_K_M`,
    /// `.IQ2_XS`, `-mxfp4`, `-f16`, `-GGUF`, `.gguf`.
    private static let fileSuffix = try! NSRegularExpression(
        pattern: "[-_. ](?:i?q[0-9][a-z0-9_]*|mxfp4|nvfp4|bf16|fp16|fp32|f16|f32|gguf)$",
        options: [.caseInsensitive]
    )

    /// Tuning markers people leave out when they say a model's name.
    private static let tuning: Set<String> = ["it", "instruct", "chat"]

    /// "blazeindustries/gemma-4-e2b-it-Q4_K_M.gguf" -> "Gemma 4 E2B".
    static func model(_ id: String) -> String {
        var spoken = words(id)
        while spoken.count > 1, let last = spoken.last, tuning.contains(last.lowercased()) {
            spoken.removeLast()
        }
        return spoken.map(capitalise).joined(separator: " ")
    }

    /// The other ways the same model is said: with its tuning marker,
    /// and as the file's own short name.
    static func synonyms(_ id: String) -> [String] {
        let title = model(id)
        var found: [String] = []
        for candidate in [words(id).map(capitalise).joined(separator: " "), shortModelName(id)] {
            if !candidate.isEmpty && candidate != title && !found.contains(candidate) {
                found.append(candidate)
            }
        }
        return found
    }

    /// The name's words, with the file markers taken off the end.
    static func words(_ id: String) -> [String] {
        var name = shortModelName(id)
        // Repeatedly: "-Q4_K_M.gguf" is two markers, and so is "-GGUF-f16".
        for _ in 0..<4 {
            let whole = NSRange(name.startIndex..., in: name)
            guard let match = fileSuffix.firstMatch(in: name, range: whole),
                  let found = Range(match.range, in: name) else { break }
            name.removeSubrange(found)
        }
        return name
            .split(whereSeparator: { $0 == "-" || $0 == "_" || $0 == " " })
            .map(String.init)
    }

    /// "gemma" -> "Gemma", "e2b" -> "E2B", "qwen3.5" -> "Qwen3.5".
    /// A short word with a digit in it is a size or a variant and is
    /// said letter by letter, so it is written in capitals.
    static func capitalise(_ word: String) -> String {
        if word.count <= 4 && word.contains(where: { $0.isNumber }) {
            return word.uppercased()
        }
        guard let first = word.first else { return word }
        return first.uppercased() + String(word.dropFirst())
    }
}
