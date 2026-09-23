//  OnDeviceSettingsView.swift
//  The knobs for models on this iPhone, and the Hugging Face token.
//
//  The token is what gated models (Llama, Gemma and friends) need, and
//  the download screen tells people to add one here — so "here" has to
//  exist. It is written to the Keychain by OnDeviceSettings and handed
//  to both the search and the download in one step, because two copies
//  drifted: a token that reached the search and not the download listed
//  a gated model fine and then failed it with a 401.

import SwiftUI

struct OnDeviceSettingsView: View {
    @EnvironmentObject private var hub: OnDeviceHub
    @State private var token = ""
    @State private var saved = false

    var body: some View {
        Form {
            Section {
                SecureField("hf_…", text: $token)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                Button(saved ? "Saved" : "Save token") {
                    let trimmed = token.trimmingCharacters(in: .whitespacesAndNewlines)
                    hub.settings.huggingFaceToken = trimmed.isEmpty ? nil : trimmed
                    hub.applyToken()
                    saved = true
                }
                .disabled(saved)
            } header: {
                Text("Hugging Face token")
            } footer: {
                Text("Needed only for gated models. A read-only token is enough. Kept in the Keychain on this device.")
            }

            Section("Running") {
                Stepper("Context: \(hub.settings.contextLength) tokens",
                        value: Binding(get: { hub.settings.contextLength },
                                       set: { hub.settings.contextLength = $0 }),
                        in: 512...32768, step: 512)
                Picker("Compute", selection: Binding(get: { hub.settings.backend },
                                                     set: { hub.settings.backend = $0 })) {
                    ForEach(ComputeBackend.allCases, id: \.self) { backend in
                        Text(backend.rawValue.capitalized).tag(backend)
                    }
                }
                Toggle("Refuse models that will not fit",
                       isOn: Binding(get: { hub.settings.enforceMemoryCheck },
                                     set: { hub.settings.enforceMemoryCheck = $0 }))
                Toggle("Stop generating in the background",
                       isOn: Binding(get: { hub.settings.pauseInBackground },
                                     set: { hub.settings.pauseInBackground = $0 }))
            }
        }
        .navigationTitle("On-device settings")
        .onAppear {
            token = hub.settings.huggingFaceToken ?? ""
            saved = !token.isEmpty
        }
        .onChange(of: token) { _, _ in saved = false }
    }
}
