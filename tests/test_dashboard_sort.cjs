const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");

test("Deep Dive switches views within its card", () => {
  const handlers = [];
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, "../utils/assets/dashboard.js"), "utf8"), {
    document: { addEventListener: (_event, handler) => handlers.push(handler) },
  });
  const card = { dataset: { deepTab: "signal" }, querySelectorAll: () => tabs };
  const tabs = ["signal", "performance", "details"].map((view) => ({
    dataset: { deepView: view }, closest: () => card,
    setAttribute(name, value) { this[name] = value; },
  }));
  handlers[0]({ target: { closest: () => tabs[1] } });
  assert.equal(card.dataset.deepTab, "performance");
  assert.equal(tabs[1]["aria-pressed"], "true");
  assert.equal(tabs[0]["aria-pressed"], "false");
  handlers[0]({ target: { closest: () => tabs[2] } });
  assert.equal(card.dataset.deepTab, "details");
});

test("overview sorts numbers and text, keeps missing values last and preserves expanded rows", () => {
  let click;
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, "../utils/assets/dashboard.js"), "utf8"), {
    document: { addEventListener: (_event, handler) => { click = handler; } },
  });
  const makeRow = (values, open = false) => ({
    dataset: { sortValues: JSON.stringify(values) }, style: {}, open,
    matches: (selector) => selector === ".sd-row",
  });
  const rows = [makeRow(["B", 934], true), makeRow(["A", -1200]), makeRow(["C", null])];
  const headers = [0, 1].map((column) => {
    const arrow = { textContent: "" };
    return {
      dataset: { sortColumn: String(column) },
      parentElement: { setAttribute(name, value) { this[name] = value; } },
      querySelector: () => arrow,
      closest: () => sheet,
    };
  });
  const sheet = { children: rows, querySelectorAll: () => headers };
  const press = (column) => click({ target: { closest: () => headers[column] } });
  const visible = () => [...rows].sort((a, b) => a.style.order - b.style.order);
  press(1);
  assert.deepEqual(visible(), [rows[1], rows[0], rows[2]]);
  assert.equal(headers[1].querySelector().textContent, " ▲");
  press(1);
  assert.deepEqual(visible(), [rows[0], rows[1], rows[2]]);
  assert.equal(headers[1].parentElement["aria-sort"], "descending");
  assert.equal(headers[1].querySelector().textContent, " ▼");
  press(1);
  assert.ok(rows.every((row) => row.style.order === ""));
  assert.equal(headers[1].querySelector().textContent, "");
  assert.equal(headers[1].parentElement["aria-sort"], "none");
  press(0);
  assert.deepEqual(visible(), [rows[1], rows[0], rows[2]]);
  assert.equal(headers[1].querySelector().textContent, "");
  assert.equal(rows[0].open, true);
  assert.equal(sheet.children, rows);
});
