from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


OUT = Path(__file__).resolve().parents[1] / "sample_data"
OUT.mkdir(exist_ok=True)
W, H = 720, 480


def optical(extra_buildings=False):
    image = Image.new("RGB", (W, H), "#91aa78")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, W, 190), fill="#648c4f")
    for x in range(20, 700, 90):
        draw.rectangle((x, 35, x + 54, 165), fill="#79a85c", outline="#536f43", width=2)
    draw.polygon([(0, 285), (145, 235), (270, 265), (410, 220), (720, 280), (720, 480), (0, 480)], fill="#397d9e")
    draw.line([(0, 330), (250, 300), (500, 335), (720, 310)], fill="#65a9c6", width=30)
    for x, y in [(390,75),(470,105),(535,65),(610,130),(320,145)]:
        draw.rectangle((x,y,x+46,y+34),fill="#c8c3b5",outline="#66645e",width=2)
    if extra_buildings:
        for x, y in [(250,60),(270,110),(330,40),(555,165)]:
            draw.rectangle((x,y,x+52,y+38),fill="#d9d4c7",outline="#5f5d58",width=2)
    return image.filter(ImageFilter.GaussianBlur(.35))


before = optical(False)
after = optical(True)
before.save(OUT / "before.png")
after.save(OUT / "after.png")
before.save(OUT / "optical.png")

rng = np.random.default_rng(42)
gray = np.asarray(before.convert("L"), dtype=np.float32)
noise = rng.normal(0, 24, gray.shape)
sar = np.clip(gray * .65 + noise, 0, 255).astype(np.uint8)
sar[285:, :] = np.clip(sar[285:, :] * .35, 0, 255)
Image.fromarray(sar, mode="L").convert("RGB").save(OUT / "sar.png")

rgb_test = Image.new("RGB", (720, 360), "#20242b")
rgb_draw = ImageDraw.Draw(rgb_test)
rgb_draw.rectangle((30, 45, 220, 315), fill="#e53935")
rgb_draw.ellipse((265, 45, 455, 315), fill="#43a047")
rgb_draw.polygon([(595, 35), (690, 315), (500, 315)], fill="#1e88e5")
rgb_test.save(OUT / "rgb_test.png")
print(f"Demo images written to {OUT}")
