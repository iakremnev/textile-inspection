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
        fig, axes = plt.subplots(2, 3)
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
        cv2.imwrite(save_to, predictions.segmentations)


def infer_camera_stream(inferencer: OpenVINOInferencer) -> None:
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()

        if success:
            predictions = inferencer.predict(frame)
            output = predictions.segmentations
            text_col = (0, 255, 0) if predictions.pred_label else (0, 0, 255)
            cv2.putText(output, f"Anom={predictions.pred_label}\nScore={predictions.pred_score:.4f}", (5, 5), cv2.FONT_HERSHEY_PLAIN, 2, text_col, 2)
            cv2.imshow("Camera inference", output)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        else:
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="*")
    parser.add_argument("-m", "--model", help="Model dir with model.bin and metadata.json")
    parser.add_argument("-a", "--all", help="Show all types of predictions")
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
        infer_camera_stream(inferencer)
    else:
        image_paths = args.image
        os.makedirs("predictions")
        for image in image_paths:
            infer_image(inferencer, image)
    