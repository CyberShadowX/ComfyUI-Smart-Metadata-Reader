import numpy as np
from PIL import Image

from smart_metadata_reader import image_io


class FakeTensor:
    def __init__(self, array):
        self.array = np.array(array)

    @property
    def shape(self):
        return self.array.shape

    def reshape(self, *shape):
        return FakeTensor(self.array.reshape(*shape))

    def float(self):
        return FakeTensor(self.array.astype(np.float32))

    def __truediv__(self, value):
        return FakeTensor(self.array / value)

    def tolist(self):
        return self.array.tolist()


class FakeTorch:
    float32 = np.float32

    @staticmethod
    def tensor(data, dtype=None):
        return FakeTensor(np.array(data, dtype=dtype))

    @staticmethod
    def from_numpy(array):
        return FakeTensor(array)


def use_fake_torch(monkeypatch):
    monkeypatch.setattr(image_io, "_torch_module", lambda: FakeTorch)


def test_rgb_png_outputs_image_and_empty_mask(tmp_path, monkeypatch):
    use_fake_torch(monkeypatch)
    path = tmp_path / "rgb.png"
    Image.new("RGB", (2, 1), (128, 64, 32)).save(path)

    image, mask, width, height = image_io.load_image_and_mask(path)

    assert (width, height) == (2, 1)
    assert image.shape == (1, 1, 2, 3)
    assert mask.shape == (1, 1, 2)
    assert np.allclose(mask.array, 0.0)
    assert np.allclose(image.array[0, 0, 0], [128 / 255.0, 64 / 255.0, 32 / 255.0])


def test_rgba_png_outputs_alpha_mask(tmp_path, monkeypatch):
    use_fake_torch(monkeypatch)
    path = tmp_path / "rgba.png"
    image = Image.new("RGBA", (3, 1))
    image.putdata([(255, 0, 0, 255), (0, 255, 0, 0), (0, 0, 255, 128)])
    image.save(path)

    loaded, mask, width, height = image_io.load_image_and_mask(path)

    assert (width, height) == (3, 1)
    assert loaded.shape == (1, 1, 3, 3)
    assert mask.shape == (1, 1, 3)
    assert np.allclose(mask.array[0, 0], [0.0, 1.0, 1.0 - (128 / 255.0)])


def test_large_image_conversion_keeps_expected_shape(tmp_path, monkeypatch):
    use_fake_torch(monkeypatch)
    path = tmp_path / "large.png"
    Image.new("RGB", (640, 512), (10, 20, 30)).save(path)

    image, mask, width, height = image_io.load_image_and_mask(path)

    assert (width, height) == (640, 512)
    assert image.shape == (1, 512, 640, 3)
    assert mask.shape == (1, 512, 640)


def test_numpy_path_does_not_use_getdata(tmp_path, monkeypatch):
    use_fake_torch(monkeypatch)
    path = tmp_path / "numpy.png"
    Image.new("RGBA", (2, 2), (10, 20, 30, 128)).save(path)

    def broken_getdata(self):
        raise AssertionError("numpy path should not call getdata")

    monkeypatch.setattr(Image.Image, "getdata", broken_getdata)

    image, mask, width, height = image_io.load_image_and_mask(path)

    assert (width, height) == (2, 2)
    assert image.shape == (1, 2, 2, 3)
    assert mask.shape == (1, 2, 2)


def test_numpy_unavailable_fallback_shape_matches_numpy_path(tmp_path, monkeypatch):
    path = tmp_path / "fallback.png"
    Image.new("RGBA", (4, 3), (10, 20, 30, 128)).save(path)

    use_fake_torch(monkeypatch)
    numpy_image, numpy_mask, width, height = image_io.load_image_and_mask(path)

    monkeypatch.setattr(image_io, "_numpy_module", lambda: None, raising=False)
    fallback_image, fallback_mask, fallback_width, fallback_height = image_io.load_image_and_mask(path)

    assert (width, height) == (fallback_width, fallback_height)
    assert fallback_image.shape == numpy_image.shape
    assert fallback_mask.shape == numpy_mask.shape
    assert np.allclose(fallback_image.array, numpy_image.array)
    assert np.allclose(fallback_mask.array, numpy_mask.array)
