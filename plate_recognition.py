import cv2
import numpy as np
import argparse
import onnxruntime as ort
import re
from collections import OrderedDict, namedtuple

from utils import correct_skew, resize_img
import easyocr

class PlateRecognition():
    INDONESIAN_PLATE_PREFIXES = {
        "A", "AA", "AB", "AD", "AE", "AG",
        "B", "BA", "BB", "BD", "BE", "BG", "BH", "BK", "BL", "BM", "BN", "BP",
        "D", "DA", "DB", "DC", "DD", "DE", "DG", "DH", "DK", "DL", "DM", "DN", "DR", "DS", "DT", "DW", "DZ",
        "E", "EA", "EB", "ED",
        "F", "G", "H", "K",
        "KB", "KH", "KT", "KU",
        "L", "M", "N", "P",
        "PA", "PB",
        "R", "S", "T", "W", "Z",
    }

    def __init__(self, model_path, enhancer, cuda=False):
        self.model_path = model_path
        self.cuda = cuda
        
        self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(self.model_path, providers=self.providers)
        print("Onnx runtime running with plate detector model...")
        
        self.reader = easyocr.Reader(['en'], gpu=cuda, quantize=True, verbose=False)
        self.enhancer = enhancer
    
    
    def extract_text(self, img, ocr_mode="accurate"):
        allowlist = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-./'
        if ocr_mode == "fast":
            try:
                return self.reader.readtext(
                    img,
                    allowlist=allowlist,
                    decoder='greedy',
                    batch_size=1,
                    detail=1,
                    paragraph=False,
                    text_threshold=0.55,
                    low_text=0.35,
                    link_threshold=0.35,
                    mag_ratio=1.0,
                )
            except TypeError:
                return self.reader.readtext(img, allowlist=allowlist)

        beam_width = 5 if ocr_mode == "balanced" else 10
        mag_ratio = 1.2 if ocr_mode == "balanced" else 1.5
        try:
            return self.reader.readtext(
                img,
                allowlist=allowlist,
                decoder='beamsearch',
                beamWidth=beam_width,
                batch_size=1,
                detail=1,
                paragraph=False,
                contrast_ths=0.05,
                adjust_contrast=0.7,
                text_threshold=0.4,
                low_text=0.2,
                link_threshold=0.2,
                width_ths=1.0,
                mag_ratio=mag_ratio,
            )
        except TypeError:
            return self.reader.readtext(img, allowlist=allowlist)

    
    def split_plate_and_date(self, text):
        plate_number, plate_date, _score = self.parse_ocr_result(text)
        return plate_number, plate_date

    
    def parse_ocr_result(self, text):
        tokens = self.normalize_ocr_tokens(text)
        plate_date = self.choose_plate_date(tokens)
        candidates = self.build_plate_candidates(tokens)

        if not candidates:
            return "", plate_date, 0

        best = max(candidates, key=lambda item: item["score"])
        return best["plate"], plate_date, best["score"] + (1 if plate_date else 0)

    
    def normalize_ocr_tokens(self, text):
        tokens = []
        for item in text:
            raw_value = str(item[1]).upper().strip()
            if not raw_value:
                continue

            confidence = float(item[2]) if len(item) > 2 else 0.5
            box = item[0] if item else []
            xs = [point[0] for point in box] if box else [0]
            ys = [point[1] for point in box] if box else [0]
            clean = re.sub(r'[^A-Z0-9]', '', raw_value)
            date_match = re.search(r'\d{1,2}[-./]\d{1,2}', raw_value)
            tokens.append({
                "raw": raw_value,
                "clean": clean,
                "confidence": confidence,
                "x": sum(xs) / len(xs),
                "y": sum(ys) / len(ys),
                "height": max(ys) - min(ys) if ys else 0,
                "is_date": bool(date_match),
                "date": date_match.group(0).replace(".", "-").replace("/", "-") if date_match else "",
            })

        return sorted(tokens, key=lambda item: (item["y"], item["x"]))

    
    def choose_plate_date(self, tokens):
        dates = [token["date"] for token in tokens if token.get("date")]
        return dates[0] if dates else ""

    
    def build_plate_candidates(self, tokens):
        usable_tokens = [
            token for token in tokens
            if token["clean"] and not token["is_date"] and not self.is_noise_token(token["clean"])
        ]
        usable_tokens = sorted(usable_tokens, key=lambda item: item["x"])
        candidates = {}

        for start in range(len(usable_tokens)):
            combined = ""
            confidence = 0.0
            max_height = 0

            for end in range(start, min(len(usable_tokens), start + 4)):
                token = usable_tokens[end]
                combined += token["clean"]
                confidence += token["confidence"]
                max_height = max(max_height, token["height"])
                self.add_candidate(candidates, combined, confidence, end - start + 1, max_height)

        if usable_tokens:
            combined = "".join(token["clean"] for token in usable_tokens)
            confidence = sum(token["confidence"] for token in usable_tokens)
            max_height = max((token["height"] for token in usable_tokens), default=0)
            self.add_candidate(candidates, combined, confidence, len(usable_tokens), max_height)

        return list(candidates.values())

    
    def add_candidate(self, candidates, raw_value, confidence, token_count, height):
        formatted = self.format_plate_number(raw_value)
        if not self.is_valid_indonesian_plate(formatted):
            return

        compact_raw = re.sub(r'[^A-Z0-9]', '', raw_value.upper())
        compact_plate = re.sub(r'[^A-Z0-9]', '', formatted.upper())
        dropped_chars = max(len(compact_raw) - len(compact_plate), 0)
        match = re.match(r'^([A-Z]{1,2})(\d{1,4})([A-Z]{1,3})$', compact_plate)
        prefix, numbers, suffix = match.groups()

        score = self.score_plate_candidate(formatted, "", confidence)
        score += min(len(numbers), 4) * 0.8
        score += min(len(suffix), 3) * 0.6
        score += min(token_count, 3) * 0.4
        score += min(height, 80) / 80
        score -= dropped_chars * 2.0

        if compact_raw and compact_raw[0].isdigit():
            score -= 6
        if prefix == "B":
            score += 0.5
        if len(numbers) >= 3:
            score += 0.8

        key = compact_plate
        if key not in candidates or candidates[key]["score"] < score:
            candidates[key] = {
                "plate": formatted,
                "score": score,
            }

    
    def is_noise_token(self, value):
        if len(value) <= 1 and not value.isalpha():
            return True
        if value in {"CNN", "CWN", "OCW", "CCW", "CW", "WWW"}:
            return True
        return False

    
    def format_plate_number(self, plate_number):
        compact = re.sub(r'[^A-Z0-9]', '', plate_number.upper())
        direct_match = re.match(r'^([A-Z]{1,2})(\d{1,4})([A-Z]{1,3})$', compact)
        if direct_match and direct_match.group(1) in self.INDONESIAN_PLATE_PREFIXES:
            return " ".join(direct_match.groups())

        compact = self.correct_plate_compact(compact)
        match = re.match(r'^([A-Z]{1,2})(\d{1,4})([A-Z]{1,3})$', compact)
        if match and match.group(1) in self.INDONESIAN_PLATE_PREFIXES:
            return " ".join(match.groups())
        return plate_number

    
    def correct_plate_compact(self, compact):
        best_value = compact
        best_score = -1

        for start in range(len(compact)):
            for end in range(start + 5, min(len(compact), start + 8) + 1):
                segment = compact[start:end]
                dropped_chars = len(compact) - len(segment)

                for prefix_len in (1, 2):
                    for suffix_len in range(1, 4):
                        number_len = len(segment) - prefix_len - suffix_len
                        if number_len < 1 or number_len > 4:
                            continue

                        raw_prefix = segment[:prefix_len]
                        raw_number = segment[prefix_len:prefix_len + number_len]
                        raw_suffix = segment[prefix_len + number_len:]
                        if not any(char.isalpha() for char in raw_prefix):
                            continue

                        prefix = self.correct_expected_letters(raw_prefix)
                        number = self.correct_expected_digits(raw_number)
                        suffix = self.correct_expected_letters(raw_suffix)
                        candidate = f"{prefix}{number}{suffix}"

                        if not re.match(r'^[A-Z]{1,2}\d{1,4}[A-Z]{1,3}$', candidate):
                            continue
                        if prefix not in self.INDONESIAN_PLATE_PREFIXES:
                            continue

                        score = 0
                        score += 4 if prefix_len == 1 else 3
                        score += min(number_len, 4)
                        score += 3 if suffix_len in (2, 3) else 2
                        score += sum(char.isalpha() for char in raw_prefix)
                        score += sum(char.isdigit() for char in raw_number)
                        score += sum(char.isalpha() for char in raw_suffix)
                        score -= dropped_chars * 1.5
                        if number_len >= 3:
                            score += 2

                        if score > best_score:
                            best_value = candidate
                            best_score = score

        return best_value

    
    def correct_expected_digits(self, value):
        replacements = {
            'O': '0', 'Q': '0', 'D': '0',
            'I': '1', 'L': '1',
            'Z': '2',
            'A': '4',
            'S': '5',
            'G': '6',
            'T': '7',
            'B': '8',
        }
        return "".join(replacements.get(char, char) for char in value)

    
    def correct_expected_letters(self, value):
        replacements = {
            '0': 'O',
            '1': 'I',
            '2': 'Z',
            '4': 'A',
            '5': 'S',
            '6': 'G',
            '7': 'T',
            '8': 'B',
        }
        return "".join(replacements.get(char, char) for char in value)

    
    def score_plate_candidate(self, plate_number, plate_date, confidence):
        compact = re.sub(r'[^A-Z0-9]', '', plate_number.upper())
        score = confidence

        if self.is_valid_indonesian_plate(plate_number):
            score += 10
        if 5 <= len(compact) <= 10:
            score += 2
        if plate_date:
            score += 1
        return score

    
    def is_valid_indonesian_plate(self, plate_number):
        compact = re.sub(r'[^A-Z0-9]', '', (plate_number or "").upper())
        match = re.match(r'^([A-Z]{1,2})(\d{1,4})([A-Z]{1,3})$', compact)
        if not match:
            return False
        prefix, numbers, suffix = match.groups()
        if prefix not in self.INDONESIAN_PLATE_PREFIXES:
            return False
        if len(numbers) > 4 or int(numbers) == 0:
            return False
        return bool(suffix)

    
    def choose_best_ocr_result(self, images, ocr_mode="accurate"):
        candidates = {}

        for image in images:
            if image is None or not image.size:
                continue

            text = self.extract_text(image, ocr_mode)
            if not text:
                continue

            plate_number, plate_date, score = self.parse_ocr_result(text)
            if not plate_number and not plate_date:
                continue

            if self.is_confident_ocr(plate_number, plate_date, score, ocr_mode):
                return plate_number, plate_date, score

            key = re.sub(r'[^A-Z0-9]', '', plate_number.upper())
            if not key:
                key = plate_date

            if key not in candidates:
                candidates[key] = {
                    "plate": plate_number,
                    "date": plate_date,
                    "count": 0,
                    "score": 0.0,
                    "best_score": -1,
                }

            candidates[key]["count"] += 1
            candidates[key]["score"] += score
            candidates[key]["best_score"] = max(candidates[key]["best_score"], score)
            if plate_date:
                candidates[key]["date"] = plate_date

        if not candidates:
            return "", "", 0

        best = max(
            candidates.values(),
            key=lambda item: (
                item["count"] * 8 + item["score"] + item["best_score"],
                bool(item["date"]),
                len(re.sub(r'[^A-Z0-9]', '', item["plate"])),
            ),
        )

        if self.is_confident_ocr(best["plate"], best["date"], best["best_score"], ocr_mode):
            return best["plate"], best["date"], best["best_score"]

        return "", best["date"] if best["date"] else "", best["best_score"]

    
    def is_confident_ocr(self, plate_number, plate_date, score, ocr_mode="accurate"):
        if not self.is_valid_indonesian_plate(plate_number):
            return False

        threshold = {
            "fast": 13.2,
            "balanced": 13.0,
            "accurate": 13.0,
        }.get(ocr_mode, 14.0)
        return score >= threshold or (score >= threshold - 1.0 and bool(plate_date))

    
    def build_ocr_images(self, image, ocr_mode="accurate"):
        base = self.resize_for_ocr(image, max_size=900)
        if base.shape[0] < 80 or base.shape[1] < 240:
            scale = 1.8 if ocr_mode == "fast" else 2.5
            base = cv2.resize(base, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, 7, 45, 45)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(denoised)
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(clahe, -1, sharpen_kernel)
        otsu = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        if ocr_mode == "fast":
            return [
                base,
                sharpened,
                otsu,
            ]

        adaptive = cv2.adaptiveThreshold(
            sharpened,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            9,
        )

        if ocr_mode == "balanced":
            return [
                base,
                clahe,
                sharpened,
                otsu,
                adaptive,
            ]

        return [
            base,
            gray,
            denoised,
            clahe,
            sharpened,
            otsu,
            255 - otsu,
            adaptive,
        ]

    
    def resize_for_ocr(self, image, max_size=1000):
        height, width = image.shape[:2]
        largest_side = max(height, width)
        if largest_side <= max_size:
            return image

        scale = max_size / largest_side
        new_size = (int(width * scale), int(height * scale))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    
    def fallback_recognition(self, img, ocr_mode="accurate"):
        height, width = img.shape[:2]
        candidates = [
            img,
            img[int(height * 0.2):height, :],
            img[int(height * 0.25):height, int(width * 0.05):int(width * 0.95)],
        ]
        if ocr_mode == "fast":
            candidates = candidates[:1]
        elif ocr_mode == "balanced":
            candidates = candidates[:2]

        ocr_images = []
        for candidate in candidates:
            if candidate is None or not candidate.size:
                continue
            ocr_images.extend(self.build_ocr_images(candidate, ocr_mode))

        plate_number, plate_date, _score = self.choose_best_ocr_result(ocr_images, ocr_mode)
        return plate_number, plate_date
    
    
    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, r, (dw, dh)
    
    
    def plate_detector(self, img):
        names = ['license']
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = img.copy()
        image, ratio, dwdh = self.letterbox(image, auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)

        im = image.astype(np.float32)
        im /= 255
        im.shape

        outname = [i.name for i in self.session.get_outputs()]
        outname

        inname = [i.name for i in self.session.get_inputs()]
        inname

        inp = {inname[0]:im}

        outputs = self.session.run(outname, inp)[0]
        return outputs, ratio, dwdh
    
    
    def plat_recognition(self, img, box, ocr_mode="accurate"):
        height, width = img.shape[:2]
        x0 = max(box[0] - 6, 0)
        y0 = max(box[1] - 6, 0)
        x1 = min(box[2] + 6, width)
        y1 = min(box[3] + 6, height)

        crop_img = img[y0:y1, x0:x1]
        hr_img = self.enhancer.enhance_image(crop_img)

        if hr_img.shape[0] > 400 or hr_img.shape[1] > 400:
            hr_img = resize_img(hr_img)
        skewness, thresh_skew = correct_skew(hr_img)

        inv = 255 - thresh_skew
        kernel = np.ones((2, 2), np.uint8)
        dilate = cv2.dilate(inv, kernel)

        ocr_images = [
            dilate,
            skewness,
            thresh_skew,
            crop_img,
            hr_img,
        ]
        if ocr_mode == "fast":
            ocr_images = [skewness, dilate, crop_img]
        elif ocr_mode == "balanced":
            ocr_images = [dilate, skewness, crop_img, hr_img]
        ocr_images.extend(self.build_ocr_images(hr_img, ocr_mode))
        plate_number, plate_date, _score = self.choose_best_ocr_result(ocr_images, ocr_mode)
        if plate_number or plate_date:
            return plate_number, plate_date

        return "not detected", ""
    
    
    def anpr(self, img, threshold, ocr_mode="accurate"):
        # plate detector
        outputs, ratio, dwdh = self.plate_detector(img)

        result = [img.copy()]
        license_num = ""
        plate_date = ""

        for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(outputs):
            image = result[int(batch_id)]
            box = np.array([x0,y0,x1,y1])
            box -= np.array(dwdh*2)
            box /= ratio
            box = box.round().astype(np.int32).tolist()

            if score >= threshold:
                # plate recognition
                license_num, plate_date = self.plat_recognition(image, box, ocr_mode)

                color = [0, 255, 0]
                if license_num == "not detected":
                    color = [0, 0, 255]

                license_num = license_num.strip()

                cv2.rectangle(image, box[:2], box[2:], color, 2)
                cv2.putText(image, license_num, (box[0], box[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, thickness=2)
                if plate_date:
                    cv2.putText(image, plate_date, (box[0], box[3] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, thickness=2)

        # result = cv2.cvtColor(result[0], cv2.COLOR_BGR2RGB)
        return result[0], license_num, plate_date
