import os.path
import numpy as np
import cv2
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import yolov5
import torch


def rotate_image(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    :param image:
    :param angle:
    :return:
    """
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def create_image_mask(gray: np.ndarray):
    """
    Creates a mask for the image
    :param gray:
    :return:
    """
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 12)
    return mask


def get_model_and_processor():
    """
    Returns the model and the processor for the OCR
    :return:
    """
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
    return model, processor


def make_prediction(image, model, processor):
    """
    Makes a prediction using the OCR model
    :param image:
    :param model:
    :param processor:
    :return:
    """
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text


def find_plate_lines(mask: np.ndarray):
    """
    Finds the lines of the plate. Finds the longest horizontal and vertical lines.
    Uses Canny edge detection and Hough transform.
    :param mask:
    :return:
    """
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)
    smallest_side = min(mask.shape)
    # Apply Hough transform to detect lines in the edge image

    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=50,
        minLineLength=smallest_side // 2, maxLineGap=smallest_side // 20
    )
    if lines is None or len(lines) < 2:
        return None

    horizontal_lines = []
    vertical_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            vector = np.array([(x2 - x1), (y2 - y1)])
            length = np.linalg.norm(vector)
            angle = np.arctan2(vector[1], vector[0]) * 180 / np.pi

            if -45 < angle < 45:
                horizontal_lines.append((length, angle, (x1, y1), (x2, y2)))
            else:
                vertical_lines.append((length, angle, (x1, y1), (x2, y2)))
    if len(horizontal_lines) < 1 or len(vertical_lines) < 1:
        return None
    horizontal_lines = sorted(horizontal_lines, reverse=True)
    vertical_lines = sorted(vertical_lines, reverse=True)
    # if we have several lines with similar lengths we need to pick the one that is closest to the border

    (h_length, h_angle, h_1, h_2) = horizontal_lines[0]
    (v_length, v_angle, v_1, v_2) = vertical_lines[0]
    if h_1[0] > h_2[0]:
        horizontal_line = (h_2, h_1)
    else:
        horizontal_line = (h_1, h_2)
    if v_1[1] < v_2[1]:
        vertical_line = (v_2, v_1)
    else:
        vertical_line = (v_1, v_2)
    # if the horizontal line is too short, we should set it to None
    if h_length < mask.shape[1] * 0.5:
        horizontal_line = None
    if v_length < mask.shape[0] * 0.5:
        vertical_line = None
    if not horizontal_line and not vertical_line:
        return None

    return horizontal_line, vertical_line


def show_image(image: np.ndarray, name: str):
    """
    Helper function to show an image
    :param image:
    :param name:
    :return:
    """
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def find_points(h_1, h_2, v_1, v_2, original_shape):
    """
    Find the corners of the plate.
    Uses the longest horizontal and vertical lines.
    Persumes that the plate is a parallelogram.
    :param h_1:
    :param h_2:
    :param v_1:
    :param v_2:
    :param original_shape:
    :return:
    """
    side_h = np.array(h_2) - np.array(h_1)
    side_v = np.array(v_2) - np.array(v_1)
    h_1, h_2 = np.array(h_1), np.array(h_2)
    v_1, v_2 = np.array(v_1), np.array(v_2)

    minimum_distance_between_points = min(
        np.linalg.norm(h_1 - v_1), np.linalg.norm(h_1 - v_2),
        np.linalg.norm(h_2 - v_1), np.linalg.norm(h_2 - v_2),
    )
    points = dict()
    if np.linalg.norm(h_1 - v_1) == minimum_distance_between_points:
        # left bottom
        point = (h_1 + v_1) / 2
        points = {
            "left_bottom": point,
            "left_top": point + side_v,
            "right_bottom": point + side_h,
            "right_upper": point + side_v + side_h,
        }
    elif np.linalg.norm(h_1 - v_2) == minimum_distance_between_points:
        # left top
        point = (h_1 + v_2) / 2
        points = {
            "left_bottom": point - side_v,
            "left_top": point,
            "right_bottom": point + side_h,
            "right_upper": point - side_v + side_h,
        }
    elif np.linalg.norm(h_2 - v_2) == minimum_distance_between_points:
        # right top
        point = (h_2 + v_2) / 2
        points = {
            "left_bottom": point - side_v - side_h,
            "left_top": point - side_h,
            "right_bottom": point - side_v,
            "right_upper": point,
        }
    else:
        # right bottom
        point = (h_2 + v_1) / 2
        points = {
            "left_bottom": point - side_h,
            "left_top": point - side_h + side_v,
            "right_bottom": point,
            "right_upper": point + side_v,
        }

    print("Points", points)
    points["left_bottom"] = np.array([np.clip(points["left_bottom"][0], 0, original_shape[1]),
                                      np.clip(points["left_bottom"][1], 0, original_shape[0])], dtype=int)
    points["left_top"] = np.array(
        [np.clip(points["left_top"][0], 0, original_shape[1]), np.clip(points["left_top"][1], 0, original_shape[0])],
        dtype=int)
    points["right_bottom"] = np.array([np.clip(points["right_bottom"][0], 0, original_shape[1]),
                                       np.clip(points["right_bottom"][1], 0, original_shape[0])], dtype=int)
    points["right_upper"] = np.array([np.clip(points["right_upper"][0], 0, original_shape[1]),
                                      np.clip(points["right_upper"][1], 0, original_shape[0])], dtype=int)
    print("Fixed", points)
    return points


def warp_rectangle(points: dict, image: np.ndarray):
    """
    Warp the plate to a rectangle, restoring the front view of the plate.
    Uses the points found by find_points.
    :param points:
    :param image:
    :return:
    """
    width = int(np.linalg.norm(points["left_bottom"] - points["right_bottom"]))
    height = int(np.linalg.norm(points["left_top"] - points["left_bottom"]))
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)

    # Compute the perspective transform matrix
    src = np.array([points["left_top"], points["right_upper"], points["right_bottom"], points["left_bottom"]],
                   dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    # Apply the perspective transform to the image
    rect_img = cv2.warpPerspective(image, M, (width, height))
    return rect_img


def load_yolo_model():
    """
    Load the YOLOv5 model for plate detection.
    :return:
    """
    model = yolov5.load('keremberke/yolov5m-license-plate', device=torch.device('cpu'))
    model.conf = 0.25  # NMS confidence threshold
    model.iou = 0.45  # NMS IoU threshold
    model.agnostic = False  # NMS class-agnostic
    model.multi_label = False  # NMS multiple labels per box
    model.max_det = 5  # maximum number of detections per image
    return model


def extract_plate_with_yolo(yolo_model, input_image: np.ndarray, scale=1.0, probability_threshold=0.6):
    """
    Extract the plate from the image using YOLOv5.
    :param yolo_model:
    :param input_image:
    :param scale:
    :return:
    """
    results = yolo_model(input_image, size=640).pred
    images = []
    for prediction in results:
        for index in range(prediction.shape[0]):
            x1, y1, x2, y2 = prediction[index, :4]
            proba = prediction[index, 4]
            if proba <= probability_threshold:
                print("Too low probability, skipping", proba)
                continue
            center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            top = int(center[1] - (height * scale / 2))
            bottom = int(center[1] + (height * scale / 2))
            left = int(center[0] - (width * scale / 2))
            right = int(center[0] + (width * scale / 2))
            top = max(0, top)
            bottom = min(input_image.shape[0] - 1, bottom)
            left = max(0, left)
            right = min(input_image.shape[1] - 1, right)
            images.append(input_image[top: bottom + 1, left: right + 1])
    return images


def extract_text(img: np.ndarray, model, processor, verbose=False, ):
    """
    Extract the text from the plate using the OCR model.
    :param img:
    :param model:
    :param processor:
    :param verbose:
    :return:
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = create_image_mask(gray)
    if verbose:
        show_image(mask, "gray")
    if mask.shape[1] / mask.shape[0] > 3:
        print("Image is too wide, assuming one line plate")
        return make_prediction(img, model, processor), ""
    print("Image shape", mask.shape, "ratio", mask.shape[1] / mask.shape[0])

    lines = find_plate_lines(mask)
    if not lines:
        print("No lines found")
        rect_img = img
    else:
        horizontal_line, vertical_line = lines
        lines_image = np.copy(img)
        if horizontal_line is None:
            # We only have a vertical line
            # We find the angle of the vertical line and rotate the image base on that
            angle = np.arctan2(vertical_line[1][1] - vertical_line[0][1], vertical_line[1][0] - vertical_line[0][0])
            rect_img = rotate_image(img, -angle)
            print("Found only vertical line")
            print("Rotating image by", -angle)
        elif vertical_line is None:
            # We only have a horizontal line
            # We find the angle of the horizontal line and rotate the image base on that
            angle = np.arctan2(horizontal_line[1][1] - horizontal_line[0][1],
                               horizontal_line[1][0] - horizontal_line[0][0])
            rect_img = rotate_image(img, -angle)
            print("Found only horizontal line")
            print("Rotating image by", -angle)

        else:
            # We have both lines
            if verbose:

                cv2.line(lines_image, horizontal_line[0], horizontal_line[1], (0, 0, 255), 3)
                cv2.line(lines_image, vertical_line[0], vertical_line[1], (0, 255, 0), 3)
                show_image(lines_image, "Lines")

            points = find_points(horizontal_line[0], horizontal_line[1], vertical_line[0], vertical_line[1], gray.shape)

            if verbose:
                cv2.circle(lines_image, points["left_bottom"], 4, (0, 0, 255), 3)
                cv2.circle(lines_image, points["left_top"], 4, (0, 255, 0), 3)
                cv2.circle(lines_image, points["right_bottom"], 4, (255, 0, 0), 3)
                cv2.circle(lines_image, points["right_upper"], 4, (255, 0, 255), 3)

                show_image(lines_image, "Points")

            rect_img = warp_rectangle(points, img)

    if verbose:
        show_image(rect_img, "Fixed image")

    division_line = find_division_line(rect_img)
    if not division_line:
        print("Could not divide the image")
        return make_prediction(rect_img, model, processor), ""
    upper_image = rect_img[: division_line]
    lower_image = rect_img[division_line:]
    if verbose:
        show_image(upper_image, "Upper image")
        show_image(lower_image, "Lower image")
    # Don't recognize the text if the image is too small
    if upper_image.shape[0] < 10 or upper_image.shape[1] < 10:
        print("Upper image is too small")
        upper_text = ""
    else:
        upper_text = make_prediction(upper_image, model, processor)
    if lower_image.shape[0] < 10 or lower_image.shape[1] < 10:
        print("Lower image is too small")
        lower_text = ""
    else:
        lower_text = make_prediction(lower_image, model, processor)
    return upper_text, lower_text


def find_division_line(image_mask: np.ndarray):
    """
    Finds the largest horizontal white rectangle on the image.
    It is used to split the plate image into upper and lower parts, if the plate has two lines of text.
    it iteratively checks every y coordinate and computes the average portion of white pixels on the line.
    It rounds the portion to the 1 digit after the decimal point.
    It tries to collect lines into sequences. Then it tries to find the closest sequence to the center of the image

    :param image_mask: A gray and white image mask after the thresholding.
    :return division_height: A y coordinate of the division line. Or None, if the image doesn't contain a white rectangle.
    """
    white_portions = []
    for y in range(image_mask.shape[0]):
        white_portions.append(np.round(np.mean(image_mask[y, :]) / 255, 1))
    sequences = []
    current_sequence = []
    for index, portion in enumerate(white_portions):
        if portion >= 0.8:
            current_sequence.append(index)
        else:
            if len(current_sequence) > 0:
                sequences.append(current_sequence)
                current_sequence = []
    if len(current_sequence) > 0:
        sequences.append(current_sequence)
    print("Sequences", sequences)
    if len(sequences) == 0:
        return None
    # Find the closest sequence to the center of the image
    center = image_mask.shape[0] // 2
    closest = sorted(sequences, key=lambda x: abs(np.mean(x) - center))[0]
    # Find the middle of the sequence
    division_height = int(np.mean(closest))
    return division_height


def recognize_plate(image_path: str, verbose=False, scale=1.0):
    """
    Full pipeline for plate recognition.
    Uses YOLOv5 for plate detection and OCR for text recognition.
    :param image_path:
    :param verbose:
    :param scale:
    :return:
    """
    img = cv2.imread(image_path)
    if verbose:
        show_image(img, "Original")

    yolo_model = load_yolo_model()
    plate_images = extract_plate_with_yolo(yolo_model, img, scale=scale)
    if len(plate_images) == 0:
        print("No plates found")
        return None
    model, processor = get_model_and_processor()
    texts = []
    for i, image in enumerate(plate_images):
        upper_text, lower_text = extract_text(image, model, processor, verbose=verbose)
        print("Plate", i, "Upper text:", upper_text, "Lower text:", lower_text)
        texts.append((upper_text, lower_text))
    return texts


if __name__ == "__main__":
    import time

    start = time.time()
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    # texts = recognize_plate(os.path.join(project_root, "data/content/1IAAAgDB0-A-960.jpg"), verbose=True, scale=1.1) # Good image with lines
    # texts = recognize_plate(os.path.join(project_root, "data/content/0KAAAgAXgeA-960.jpg"), verbose=True, scale=1.1) # No lines
    # texts = recognize_plate(os.path.join(project_root, "data/content/1yqdergyycdci_fls6w8.jpeg"), verbose=True, scale=1.1) # Only one Line
    # print("Plates:", texts)
    # texts = recognize_plate(os.path.join(project_root, "data/content/A car with single line.jpg"), verbose=True, scale=1.1) # Single text string plate
    # texts = recognize_plate(os.path.join(project_root, "data/G6honJPmMM4.jpg"), verbose=True, scale=1.1) # Single text string plate
    texts = recognize_plate(os.path.join(project_root, "data/1.jpg"), verbose=True, scale=1.1) # Single text string plate
    print(texts)


    print("Time spent", time.time() - start)

