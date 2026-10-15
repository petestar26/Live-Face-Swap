import ast
import pathlib
import unittest
from types import SimpleNamespace

import cv2
import numpy as np
from modules.face_geometry import crop_support, face_outline


class BlendGeometryTests(unittest.TestCase):
    def test_crop_edges_are_transparent(self):
        for size in (128, 256, 512):
            mask = crop_support((size, size))
            self.assertEqual(float(mask[0].max()), 0)
            self.assertEqual(float(mask[-1].max()), 0)
            self.assertEqual(float(mask[:, 0].max()), 0)
            self.assertEqual(float(mask[:, -1].max()), 0)
            self.assertEqual(float(mask[size // 2, size // 2]), 1)

    def test_oversized_mask_cannot_paint_outside_crop(self):
        # Exercise the actual paste function without loading GPU models/UI.
        path = pathlib.Path(__file__).resolve().parents[1] / 'modules/processors/frame/face_swapper.py'
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_landmark_paste_back')
        namespace = dict(np=np, cv2=cv2, Frame=np.ndarray)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
        original = np.full((200, 200, 3), 80, np.uint8)
        fake = np.full((128, 128, 3), 200, np.uint8)
        mask = np.full((200, 200), 255, np.uint8)
        result = namespace['_landmark_paste_back'](original.copy(), fake,
                    np.array([[1., 0., -30.], [0., 1., -30.]]), mask)
        np.testing.assert_array_equal(result[:30], original[:30])
        np.testing.assert_array_equal(result[158:], original[158:])
        self.assertTrue(np.all(result[80, 80] == 200))

    def test_bad_landmarks_rejected(self):
        face = SimpleNamespace(landmark_2d_106=np.full((106, 2), np.nan),
                               kps=np.zeros((5, 2)), bbox=np.array([0, 0, 100, 100]))
        self.assertIsNone(face_outline(face))


if __name__ == '__main__':
    unittest.main()
