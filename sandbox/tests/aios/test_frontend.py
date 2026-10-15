import re
from pathlib import Path

from fastapi.testclient import TestClient

from p17sim.aios.api import STATIC, create_app


def norm(path: str) -> str:
    path = path.split("?")[0]
    path = re.sub(r"\$\{[^}]+\}", "{p}", path)
    return re.sub(r"\{[a-z_]+\}", "{p}", path)


def test_page_and_assets_are_served(v2ds, tmp_path):
    c = TestClient(create_app(v2ds, store_dir=tmp_path / "ws", results=tmp_path / "r"))
    html = c.get("/").text
    assert "<title>Personal AI OS</title>" in html and "/static/app.js" in html
    assert c.get("/static/app.js").status_code == 200 and c.get("/static/app.css").status_code == 200


def test_every_api_call_in_the_page_has_a_route(v2ds, tmp_path):
    app = create_app(v2ds, store_dir=tmp_path / "ws", results=tmp_path / "r")
    routes = {norm(r.path) for r in app.routes if hasattr(r, "path")}
    js = (STATIC / "app.js").read_text()
    calls = {norm(m) for m in re.findall(r"api\(\s*[`\"](/api/[^`\"]+)[`\"]", js)}
    calls |= {norm(m) for m in re.findall(r"src=\"(/api/[^\"]+)\"", js)}
    assert calls, "no api calls found"
    missing = sorted(c for c in calls if c not in routes)
    assert not missing, missing


def test_every_element_the_script_uses_exists():
    html = (STATIC / "index.html").read_text()
    js = (STATIC / "app.js").read_text()
    ids = set(re.findall(r'\$\("#([A-Za-z0-9_-]+)"\)', js))
    missing = sorted(i for i in ids if f'id="{i}"' not in html and f'id="{i}"' not in js)  # or built by the script
    assert not missing, missing
