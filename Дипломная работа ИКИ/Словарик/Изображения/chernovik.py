import base64, pathlib

src = pathlib.Path("solar.jpg")          # твой присланный снимок
b64 = "data:image/jpeg;base64," + base64.b64encode(src.read_bytes()).decode()

svg_path = pathlib.Path("tile_rice_pipeline.svg")
svg = svg_path.read_text(encoding="utf-8")
svg = svg.replace("SOLAR_IN", b64).replace("SOLAR_OUT", b64)
svg_path.write_text(svg, encoding="utf-8")
print("готово:", svg_path)