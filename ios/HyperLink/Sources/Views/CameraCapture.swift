//  CameraCapture.swift
//  The camera, as one SwiftUI view.
//
//  `UIImagePickerController` rather than a `PHPickerViewController` or a
//  hand-rolled `AVCaptureSession`: the picker is the only one of the
//  three that gives a *camera* with one line of configuration, and a
//  capture session here would mean writing a preview layer, a shutter
//  button, orientation handling and a flash toggle to end up in the same
//  place. It is soft-deprecated for the photo *library* — `PhotosPicker`
//  is the replacement, and ChatView uses it — and remains the supported
//  way to take a picture.
//
//  `NSCameraUsageDescription` has to be in Info.plist or this crashes on
//  presentation rather than failing. `ios/project.yml` declares it, under
//  the target's `info.properties`.

#if canImport(UIKit)
import SwiftUI
import UIKit

struct CameraCapture: UIViewControllerRepresentable {
    /// Called with the captured image. Not called on cancel.
    let onCapture: (UIImage) -> Void

    @Environment(\.dismiss) private var dismiss

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        // Falls back to the library on a simulator, which has no camera.
        // Without this the presentation is a black screen nobody can get
        // out of.
        picker.sourceType =
            UIImagePickerController.isSourceTypeAvailable(.camera) ? .camera : .photoLibrary
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ picker: UIImagePickerController, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(onCapture: onCapture, dismiss: { dismiss() })
    }

    final class Coordinator: NSObject, UIImagePickerControllerDelegate,
                             UINavigationControllerDelegate {
        private let onCapture: (UIImage) -> Void
        private let dismiss: () -> Void

        init(onCapture: @escaping (UIImage) -> Void, dismiss: @escaping () -> Void) {
            self.onCapture = onCapture
            self.dismiss = dismiss
        }

        func imagePickerController(
            _ picker: UIImagePickerController,
            didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]
        ) {
            // `.editedImage` first: if the user cropped, the crop is what
            // they meant to send.
            let image = (info[.editedImage] as? UIImage)
                ?? (info[.originalImage] as? UIImage)
            if let image { onCapture(image) }
            dismiss()
        }

        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            dismiss()
        }
    }
}
#endif
