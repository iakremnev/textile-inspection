import argparse
import os.path

import cv2
import matplotlib.pyplot as plt
from anomalib.data.utils import read_image
from anomalib.deploy import OpenVINOInferencer


def infer_image(inferencer: OpenVINOInferencer, image_path: str, vis_all_predictions: bool = False) -> None:
    image = read_image(image_path)
    save_to = os.path.join("predictions", os.path.basename(image_path))

    predictions = inferencer.predict(image=image)
    
    if vis_all_predictions:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes[0,0].imshow(predictions.image)
        axes[0,1].imshow(predictions.segmentations)
        axes[1,0].imshow(predictions.anomaly_map)
        axes[1,1].imshow(predictions.heat_map)
        axes[1,2].imshow(predictions.pred_mask)

        axes[0,0].set_title("image")
        axes[0,1].set_title("segmentations")
        axes[1,0].set_title("anomaly_map")
        axes[1,1].set_title("heat_map")
        axes[1,2].set_title("pred_mask")

        plt.savefig(save_to)
    
    else:
        output = predictions.segmentations
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_to, output)


def infer_camera_stream(inferencer: OpenVINOInferencer, out_video_file: str) -> None:
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out_cap = cv2.VideoWriter(out_video_file, cv2.VideoWriter_fourcc("M", "J", "P", "G"), 15, (frame_width, frame_height))
    
    try:
        while cap.isOpened():
            success, frame = cap.read()

            if success:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                predictions = inferencer.predict(frame)
                output = predictions.segmentations
                print(i, predictions.pred_score)
                text_col = (255, 0, 0) if predictions.pred_label else (255, 0, 0)
                cv2.putText(output, f"Anom={predictions.pred_label}", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, text_col, 2)
                cv2.putText(output, f"Score={predictions.pred_score:.4f}", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, text_col, 2)
                # cv2.imshow("Camera_inference", output)
                output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
                out_cap.write(output)

            else:
                break
    except KeyboardInterrupt:
        print("Stopping video stream")
    
    cap.release()
    out_cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="*")
    parser.add_argument("-m", "--model", help="Model dir with model.bin and metadata.json")
    parser.add_argument("-a", "--all", action="store_true", help="Show all types of predictions")
    parser.add_argument("-c", "--camera", action="store_true", help="Infer from webcam stream rather then image files")

    args = parser.parse_args()

    openvino_model_path = os.path.join(args.model, "model.bin")
    metadata_path = os.path.join(args.model, "metadata.json")

    inferencer = OpenVINOInferencer(
        path=openvino_model_path,
        metadata_path=metadata_path,
        device="CPU",
    )

    if args.camera:
        # For some unfortunate reason, cv2.imshow freezes when the OpenVINOInferencer is imported
        # so the video must be recorded on disk
        infer_camera_stream(inferencer, out_video_file="out.avi")
    else:
        image_paths = args.image
        os.makedirs("predictions", exist_ok=True)
        for image in image_paths:
            infer_image(inferencer, image, vis_all_predictions=args.all)
