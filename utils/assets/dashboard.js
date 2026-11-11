/* Sort whole expandable rows using CSS order, preserving React's DOM and open state. */
(function () {
  document.addEventListener("click", (event) => {
    const button = event.target.closest(".sd-deep-tab");
    if (!button) return;
    const card = button.closest(".sd-card");
    card.dataset.deepTab = button.dataset.deepView;
    card.querySelectorAll(".sd-deep-tab").forEach((tab) => {
      tab.setAttribute("aria-pressed", String(tab === button));
    });
  });

  document.addEventListener("click", (event) => {
    const button = event.target.closest(".sd-close-all");
    if (!button) return;
    button.closest(".sd-page").querySelectorAll(".sd-row[open]").forEach((row) => {
      row.open = false;
    });
  });

  document.addEventListener("click", (event) => {
    const button = event.target.closest(".sd-sort");
    if (!button) return;
    const sheet = button.closest(".sd-sheet");
    if (!sheet) return;
    const column = Number(button.dataset.sortColumn);
    const reset = button.dataset.direction === "desc";
    const ascending = button.dataset.direction !== "asc";
    sheet.querySelectorAll(".sd-sort").forEach((header) => {
      delete header.dataset.direction;
      header.parentElement.setAttribute("aria-sort", "none");
      header.querySelector(".viz-sort-arrow").textContent = "";
    });
    if (reset) {
      Array.from(sheet.children).filter((row) => row.matches(".sd-row")).forEach((row) => {
        row.style.order = "";
      });
      return;
    }
    button.dataset.direction = ascending ? "asc" : "desc";
    button.parentElement.setAttribute("aria-sort", ascending ? "ascending" : "descending");
    button.querySelector(".viz-sort-arrow").textContent = ascending ? " ▲" : " ▼";
    const rows = Array.from(sheet.children).filter((row) => row.matches(".sd-row"));
    const value = (row) => JSON.parse(row.dataset.sortValues)[column];
    const missing = (v) => v === null || v === "" || v === "—";
    rows.sort((a, b) => {
      const av = value(a), bv = value(b);
      if (missing(av) || missing(bv)) return Number(missing(av)) - Number(missing(bv));
      const comparison = typeof av === "number" && typeof bv === "number"
        ? av - bv : String(av).localeCompare(String(bv));
      return ascending ? comparison : -comparison;
    });
    rows.forEach((row, index) => { row.style.order = index; });
  });

  /* On a phone, size each signal column to its widest cell so nothing is cut off and
     no space is wasted. Each row is its own grid, so the widths are measured across
     every row and section and set once, keeping the columns lined up. */
  const phone = window.matchMedia("(max-width: 700px)");
  const MARKER = "14px";  // the +/- column: a pseudo-element, so it can't be measured
  let pending = false;
  function fitColumns() {
    pending = false;
    const sections = document.querySelectorAll(".sd-signal-section");
    if (!phone.matches) {
      sections.forEach((s) => s.style.removeProperty("--sd-columns"));
      return;
    }
    if (!sections.length) return;
    sections.forEach((s) => s.style.setProperty("--sd-columns", `${MARKER} repeat(7, max-content)`));
    const widest = new Array(7).fill(0);
    document.querySelectorAll(".sd-signal-section :is(.sd-row-head, .sd-row-legend)").forEach((row) => {
      const cells = Array.from(row.children);
      const offset = row.matches(".sd-row-legend") ? 1 : 0;  // the legend has a real marker cell
      cells.slice(offset, offset + 7).forEach((cell, i) => {
        widest[i] = Math.max(widest[i], Math.ceil(cell.getBoundingClientRect().width));
      });
    });
    const columns = `${MARKER} ${widest.map((w) => `${w}px`).join(" ")}`;
    sections.forEach((s) => s.style.setProperty("--sd-columns", columns));
  }
  function scheduleFit() {
    if (pending) return;
    pending = true;
    requestAnimationFrame(fitColumns);
  }
  function startFitting() {
    new MutationObserver(scheduleFit).observe(document.body, { childList: true, subtree: true, characterData: true });
    phone.addEventListener("change", scheduleFit);
    if (document.fonts) document.fonts.ready.then(scheduleFit);
    scheduleFit();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", startFitting);
  else startFitting();
}());
