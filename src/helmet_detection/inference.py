"""Run helmet-detection inference on a single image."""

import argparse
from pathlib import Path
from typing import Optional, Sequence, Union


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _class_name(names: Union[dict, list, tuple], class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, f"class {class_id}"))
    if 0 <= class_id < len(names):
        return str(names[class_id])
    return f"class {class_id}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect helmet use and other trained classes in a rider image."
    )
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Path to the input image (for example: media/helmet.jpg).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "weights" / "best.pt",
        help="Path to YOLO weights (default: weights/best.pt).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/inference"),
        help="Directory for the annotated image (default: runs/inference).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Minimum detection confidence from 0 to 1 (default: 0.25).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size in pixels (default: 640).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the annotated image in an OpenCV window.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not 0 <= args.conf <= 1:
        parser.error("--conf must be between 0 and 1.")
    if args.imgsz <= 0:
        parser.error("--imgsz must be a positive integer.")
    if not args.source.is_file():
        parser.error(f"Input image does not exist: {args.source}")
    if not args.model.is_file():
        parser.error(f"Model weights do not exist: {args.model}")

    import cv2
    from ultralytics import YOLO

    image = cv2.imread(str(args.source))
    if image is None:
        parser.error(f"Could not read input image: {args.source}")

    model = YOLO(str(args.model))
    predictions = model.predict(
        source=image,
        conf=args.conf,
        imgsz=args.imgsz,
        verbose=False,
    )
    if not predictions:
        parser.error("The model returned no prediction results.")

    prediction = predictions[0]
    annotated_image = prediction.plot()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.source.stem}_predictions.jpg"
    if not cv2.imwrite(str(output_path), annotated_image):
        raise OSError(f"Could not save annotated image to {output_path}")

    boxes = prediction.boxes
    detection_count = len(boxes) if boxes is not None else 0
    print(f"Detected {detection_count} object(s).")
    if boxes is not None:
        for box in boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            print(
                f"- {_class_name(model.names, class_id)} "
                f"({confidence:.2f} confidence)"
            )
    print(f"Annotated image saved to: {output_path}")

    if args.show:
        cv2.imshow("Helmet Detection", annotated_image)
        try:
            cv2.waitKey(0)
        finally:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
