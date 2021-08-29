import json
import os.path
import random

import numpy as np
from PIL import Image, ImageDraw
import pyny3d.geoms as geometry


def get_patient_file(idx: int):
    assert 0 <= idx <= 4
    base_path = os.path.join(os.path.dirname(__file__), "../asset")
    patient_file = os.path.join(base_path, "patients", str(idx) + ".png")
    return patient_file


def patch_str_2_idx(name: str):
    return {
        "arm": 1,
        "knee": 2,
        "shoulder": 3,
        "leg": 4
    }.get(name, -1)


def wound_idx_2_str(idx: int):
    return {
        0: "big",
        1: "bruise",
        2: "infected",
        3: "scalpel",
        4: "scar",
        5: "small",
        6: "stitched",
        7: "swollen",
        8: "swollen_infected"
    }.get(idx, "")


def wound_str_2_idx(name: str):
    return {
        "big": 0,
        "bruise": 1,
        "infected": 2,
        "scalpel": 3,
        "scar": 4,
        "small": 5,
        "stitched": 6,
        "swollen": 7,
        "swollen_infected": 8
    }.get(name, -1)


def convert_scale(src_n):
    return {
        "infected.png": 0.617801047,
        "small.png": 0.478991597,
        "scalpel.png": 0.471204188,
    }.get(src_n, 1)


def get_patch_file(idx: int):
    assert 0 <= idx <= 4
    base_path = os.path.join(os.path.dirname(__file__), "../asset", "patients", "patches_v2")
    patch_file = sorted(os.listdir(base_path))[idx]
    return os.path.join(base_path, patch_file)


def get_body_part_canvas(idx: int):
    assert 0 <= idx <= 4
    base_path = os.path.join(os.path.dirname(__file__), "../asset", "body_parts")
    patch_canvas = sorted(os.listdir(base_path))[idx]
    return os.path.join(base_path, patch_canvas)


def get_wound_patch(idx: int):
    assert 0 <= idx <= 7
    base_path = os.path.join(os.path.dirname(__file__), "../asset", "wounds")
    return os.path.join(base_path, wound_idx_2_str(idx) + ".png")


class Patient:
    def __init__(self, patient=None, patches=None):
        assert patches is None or 0 in patches, "Patch must at least contain 0 (Torso)"
        if patient is None:
            patient = random.randint(0, 4)
        self.patient = patient
        self.patient_canvas_file = get_patient_file(self.patient)

        if patches is None:
            patches = [0]
            patches.extend(random.sample([1, 2, 3, 4], random.randint(1, 4)))
        self.patches_idx = patches
        print(f"Selected patient {self.patient}")
        self.patches_files = [get_patch_file(i) for i in self.patches_idx]
        print(f"Selected body parts for surgery: {[os.path.split(t)[1] for t in self.patches_files]}")
        self.body_parts = [BodyPart(i) for i in self.patches_idx]
        self.bbox = []  # [x, y, w, h]
        for f_ in self.patches_files:
            _, name = os.path.split(f_)
            name = name.split(".")[0].split("_")
            self.bbox.append([int(name[2]), int(name[3]), int(name[4]), int(name[5])])

    def to_json(self):
        return {
            "patient": self.patient,
            "body_parts": {
                os.path.split(t)[1]: p.to_json() for t, p in zip(self.patches_files, self.body_parts)
            }
        }

    def render(self):
        canvas = Image.open(self.patient_canvas_file)
        print(f"Loaded canvas file '{os.path.split(self.patient_canvas_file)[1]}'")
        draw = ImageDraw.Draw(canvas)
        for f_ in self.patches_files:
            patch = Image.open(f_)
            _, name = os.path.split(f_)
            name = name.split(".")[0].split("_")
            coords = (int(name[2]), int(name[3]))
            canvas.paste(patch, coords)
            print(f"Pasted patch '{os.path.split(f_)[1]}' onto canvas at position {coords}")

        for bp, f_ in zip(self.body_parts, self.patches_files):
            _, name = os.path.split(f_)
            name = name.split(".")[0].split("_")
            if bp.wound_files:
                draw.rounded_rectangle([(int(name[2]), int(name[3])),
                                        (int(name[2]) + 74, int(name[3]) + 74)],
                                       13, fill=None, outline=(214, 241, 139), width=3)
        return canvas

    def get_canvas_mask(self):
        canvas_g = Image.new('L', Image.open(self.patient_canvas_file).size)
        draw = ImageDraw.Draw(canvas_g)
        for bp, box in zip(self.body_parts, self.bbox):
            draw.rectangle([(box[0], box[1]), (box[0] + box[2], box[1] + box[3])], fill=255)
        return canvas_g

    def canvas_mask_to_ndarray(self):
        ...

    def ndarray_mask_to_rgb(self):
        ...


class BodyPart:
    polygons = {
        0: [(529, 66), (152, 86), (156, 432), (526, 432)],
        1: [(558, 259), (133, 159), (158, 323), (543, 369)],
        2: [(540, 116), (375, 136), (153, 105), (156, 423), (544, 364)],
        3: [(521, 69), (250, 69), (160, 271), (156, 425), (521, 425)],
        4: [(526, 220), (156, 170), (158, 387), (526, 320)]
    }
    color_map = {
        10: np.array([255, 255, 255], dtype=np.uint8),
        9: np.array([0, 0, 0], dtype=np.uint8),
        8: np.array([63, 81, 181], dtype=np.uint8),
        7: np.array([3, 169, 244], dtype=np.uint8),
        6: np.array([169, 3, 244], dtype=np.uint8),
        5: np.array([0, 150, 136], dtype=np.uint8),
        4: np.array([139, 195, 74], dtype=np.uint8),
        3: np.array([255, 235, 59], dtype=np.uint8),
        2: np.array([255, 152, 0], dtype=np.uint8),
        1: np.array([121, 85, 72], dtype=np.uint8),
        0: np.array([96, 125, 139], dtype=np.uint8),
    }
    wound_progress = {
        "big.png": "stitched",
        "stitched.png": "scar",
        "bruise.png": None,
        "infected.png": "big",
        "scalpel.png": "big",
        "scar.png": None,
        "small.png": "scar",
        "swollen.png": "small",
        "swollen_infected.png": "swollen"
    }

    def __init__(self, idx: int, wounds: list = None, num_wounds=None):
        assert wounds is None or len(wounds)
        if num_wounds is None:
            num_wounds = random.randint(1, 4)
        self.idx = idx
        self.body_part_canvas = get_body_part_canvas(idx)
        self.wounds = [random.randint(1, 7) for _ in range(num_wounds)]
        if wounds is None:
            wounds = [get_wound_patch(idx) for idx in self.wounds]
        self.wound_files = wounds
        print(f"    Initializing body part {idx}. "
              f"Selected {num_wounds} wounds {[os.path.split(t)[1] for t in self.wound_files]}")
        self.wound_data = []

        m_canvas = Image.new('L', (702, 510), color=10)  # Is Outside of the canvas
        drawer = ImageDraw.Draw(m_canvas)
        drawer.polygon(self.polygons[idx], fill=9)  # Is skin and inside of canvas
        self.m_canvas_np = np.array(m_canvas)

        rmv_idx = []
        for i, wound_file in enumerate(self.wound_files):
            places, rot, scale = self._find_places_for_wound(wound_file)
            if places is None:
                rmv_idx.append(i)
                print(f"        Unable to find configuration for wound {os.path.split(wound_file)[1]} - Removing.")
                continue
            self.wound_data.append({})
            self.wound_data[-1]["coords"] = places
            self.wound_data[-1]["rotation"] = rot
            self.wound_data[-1]["scale"] = scale
            print(f"        Found configuration for wound {os.path.split(wound_file)[1]}: {self.wound_data[-1]}")

        for idx in rmv_idx:
            self.wounds.pop(idx)
            self.wound_files.pop(idx)

    def render(self):
        canvas = Image.open(self.body_part_canvas)
        for wf, wd in zip(self.wound_files, self.wound_data):
            img = Image.open(wf)
            if "scar.png" in wf:
                img = img.resize((int(img.width * wd["scale"]), int(img.height)))
            else:
                img = img.resize((int(img.width * wd["scale"]), int(img.height * wd["scale"])))
            img = img.rotate(wd["rotation"], expand=True)
            # Coords are in [y, x, height, width] order
            img_blend = Image.new('RGBA', canvas.size)
            img_blend.paste(img, (wd["coords"][1], wd["coords"][0]), img.split()[-1])
            canvas = Image.alpha_composite(canvas, img_blend)

        return canvas

    def get_canvas_mask(self):

        mask = self.m_canvas_np.copy()
        mask_rgb = np.zeros([*mask.shape, 3], dtype=np.uint8)
        for key in self.color_map:
            mask_rgb[mask == key] = self.color_map[key]

        return Image.fromarray(mask_rgb)

    def canvas_mask_to_ndarray(self):
        mask_nd = np.zeros([*self.m_canvas_np.shape, 11], dtype=np.uint8)
        for key in self.color_map:
            mask_nd[self.m_canvas_np == key, key] = 1
        return mask_nd

    def ndarray_mask_to_rgb(self, arr: np.ndarray):
        mask_rgb = np.zeros([*self.m_canvas_np.shape, 3], dtype=np.uint8)
        for key in self.color_map:
            mask_rgb[arr[:, :, key] == 1] = self.color_map[key]
        return Image.fromarray(mask_rgb)

    def to_json(self):
        return {
            "id": self.idx,
            "wounds": {
                os.path.split(wf)[1]: data for wf, data in zip(self.wound_files, self.wound_data)
            }
        }

    def advance_wound(self, idx):
        if not len(self.wound_files):
            return True
        progress = self.wound_progress[os.path.split(self.wound_files[idx])[1]]
        if progress is None:
            self.wound_files.pop(idx)
            self.wound_data.pop(idx)
            if not len(self.wound_files):
                return True
            return False
        next_idx = wound_str_2_idx(progress)
        next_wound = get_wound_patch(next_idx)
        self.wound_data[idx]["scale"] *= convert_scale(os.path.split(self.wound_files[idx])[1])
        old_bbox = self.wound_data[idx]["coords"]  # y, x, h, w
        img = Image.open(next_wound)
        if "scar.png" in next_wound or "stitched.png" in next_wound:
            img = img.resize((int(img.width * self.wound_data[idx]["scale"]), int(img.height)))
        else:
            img = img.resize(
                (int(img.width * self.wound_data[idx]["scale"]), int(img.height * self.wound_data[idx]["scale"])))
        img = img.rotate(self.wound_data[idx]["rotation"], expand=True)
        new_bbox = [old_bbox[0] + (old_bbox[2] - img.height) // 2,
                    old_bbox[1] + (old_bbox[3] - img.width) // 2,
                    img.height, img.width]
        self.wound_files[idx] = next_wound
        self.wound_data[idx]["coords"] = new_bbox
        return False

    def _find_places_for_wound(self, f_wound):
        w_img = Image.open(f_wound)

        coords = []
        rotation = 0
        scale = 1.0
        for iteration in range(1000):
            rotation = angle = random.randint(0, 360)
            if os.path.split(f_wound)[1] in ["big.png", "infected.png"]:
                # Scale is allowed
                scale = random.uniform(0.5, 1.5)
                img = w_img.copy().resize((int(w_img.width * scale), int(w_img.height * scale)))
            elif os.path.split(f_wound)[1] in ["scar.png", "stitched.png"]:
                scale = random.uniform(0.5, 1.5)
                img = w_img.copy().resize((int(w_img.width * scale), int(w_img.height)))
            else:
                img = w_img.copy()
            img = img.rotate(angle, expand=True)

            img = np.array(img.split()[-1])
            # img[img > 0] = wound_str_2_idx(os.path.split(f_wound)[1].split(".")[0])
            THRESH = 50
            img[img < THRESH] = 0
            img[img >= THRESH] = 1
            for y in range(0, self.m_canvas_np.shape[0] - img.shape[0], 20):
                for x in range(0, self.m_canvas_np.shape[1] - img.shape[1], 20):
                    if np.any(self.m_canvas_np[y:y + img.shape[0], x:x + img.shape[1]][img > 0] != 9):
                        continue
                    coords.append((y, x))
            if len(coords):
                break

        if not len(coords):
            return None, None, None

        coords = random.choice(coords)
        # TODO Change [y, x, h, w] to [x, y, w, h]
        coords = [coords[0], coords[1], img.shape[0], img.shape[1]]
        self.m_canvas_np[coords[0]:coords[0] + coords[2], coords[1]:coords[1] + coords[3]] \
            [img > 0] = wound_str_2_idx(os.path.split(f_wound)[1].split(".")[0])
        return coords, rotation, scale


class Generator:
    def __init__(self):
        self.p = Patient()

    def save(self, path: str):
        pass
