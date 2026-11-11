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
}());
