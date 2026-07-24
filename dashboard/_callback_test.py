import json
import sys
import time
import urllib.request

slug = "book-curve-tens-10s30s"
base = "http://127.0.0.1:8099"


def post(payload):
    req = urllib.request.Request(
        f"{base}/_dash-update-component",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def sim(changed_id, changed_prop, changed_value, window="1Y", page=0):
    outputs = [
        {"id": f"card-body-{slug}", "property": "children"},
        {"id": f"trades-page-{slug}", "property": "data"},
    ]
    inputs = [
        {"id": f"pull-{slug}", "property": "n_clicks", "value": 0},
        {"id": f"run-{slug}", "property": "n_clicks", "value": 0},
        {"id": f"window-{slug}", "property": "value", "value": window},
        {"id": f"trades-prev-{slug}", "property": "n_clicks", "value": 0},
        {"id": f"trades-next-{slug}", "property": "n_clicks", "value": 0},
    ]
    for inp in inputs:
        if inp["id"] == changed_id:
            inp["value"] = changed_value
    payload = {
        "output": "..%s.." % "...".join(f"{o['id']}.{o['property']}" for o in outputs),
        "outputs": outputs,
        "inputs": inputs,
        "state": [
            {"id": f"window-{slug}", "property": "value", "value": window},
            {"id": f"trades-page-{slug}", "property": "data", "value": page},
        ],
        "changedPropIds": [f"{changed_id}.{changed_prop}"],
    }
    return post(payload)


print("== window change to 1Y ==")
status, body = sim(f"window-{slug}", "value", "1Y", window="1Y", page=0)
print("status", status)
if status != 200:
    print(body)
    sys.exit(1)
parsed = json.loads(body)
new_page = parsed["response"][f"trades-page-{slug}"]["data"]
print("new page after window change:", new_page)
assert new_page == 0

print("== trades-next click ==")
status, body = sim(f"trades-next-{slug}", "n_clicks", 1, window="1Y", page=0)
print("status", status)
parsed = json.loads(body)
new_page = parsed["response"][f"trades-page-{slug}"]["data"]
print("new page after next click:", new_page)
assert new_page == 5

print("== trades-prev click from page 5 ==")
status, body = sim(f"trades-prev-{slug}", "n_clicks", 1, window="1Y", page=5)
print("status", status)
parsed = json.loads(body)
new_page = parsed["response"][f"trades-page-{slug}"]["data"]
print("new page after prev click:", new_page)
assert new_page == 0

print("ALL CALLBACK CHECKS PASSED")
