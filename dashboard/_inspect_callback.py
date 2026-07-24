from dashboard.app import build_app

app = build_app()
for k in app.callback_map:
    if "book-curve-tens-10s30s" in k:
        print(repr(k))
