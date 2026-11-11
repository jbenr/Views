const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");
const source = fs.readFileSync(path.join(__dirname, "../utils/assets/dashboard_theme.js"), "utf8");

function load(saved = null, dashboard = true, blocked = false) {
  const root = { dataset: {} };
  const button = { textContent: "Dark mode", setAttribute(k, v) { this[k] = v; } };
  const image = { dataset: { chartLight: "light-png", chartDark: "dark-png" }, src: "light-png",
    getAttribute(k) { return this[k]; }, setAttribute(k, v) { this[k] = v; } };
  let click;
  const storage = { value: saved, getItem() { if (blocked) throw Error(); return this.value; },
    setItem(k, v) { if (blocked) throw Error(); this.value = v; } };
  vm.runInNewContext(source, {
    document: { documentElement: root, querySelector: () => dashboard ? { querySelectorAll: () => [image] } : null,
      getElementById: () => button, addEventListener: (_, callback) => { click = callback; } },
    localStorage: storage, MutationObserver: class { observe() {} },
  });
  return { root, button, storage, image, toggle: () => click({ target: { closest: () => button } }) };
}
test("theme defaults light, toggles both ways and remembers the choice", () => {
  const app = load();
  assert.equal(app.root.dataset.sdTheme, "light");
  app.toggle();
  assert.equal(app.root.dataset.sdTheme, "dark");
  assert.equal(app.button.textContent, "Light mode");
  assert.equal(app.button["aria-pressed"], "true");
  assert.equal(app.image.src, "dark-png");
  assert.equal(load(app.storage.value).root.dataset.sdTheme, "dark");
  app.toggle();
  assert.equal(app.root.dataset.sdTheme, "light");
  assert.equal(app.image.src, "light-png");
});
test("theme leaves other apps alone and works when storage is unavailable", () => {
  assert.equal(load("dark", false).root.dataset.sdTheme, undefined);
  const app = load(null, true, true);
  app.toggle();
  assert.equal(app.root.dataset.sdTheme, "dark");
});
