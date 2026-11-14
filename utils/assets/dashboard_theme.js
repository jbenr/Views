/* Dashboard-only theme, saved per browser; light is the initial default. */
(function () {
  const key = "signal-dashboard-theme";
  let theme = "light";
  try { theme = localStorage.getItem(key) === "dark" ? "dark" : "light"; } catch (_) {}
  function apply() {
    const app = document.querySelector(".sd-app");
    if (!app) return;
    document.documentElement.dataset.sdTheme = theme;
    // Supporting mobile browsers use this for the address/tab bar and edge chrome.
    let browserTheme = document.querySelector('meta[name="theme-color"]');
    if (!browserTheme) {
      browserTheme = document.createElement("meta");
      browserTheme.name = "theme-color";
      document.head.appendChild(browserTheme);
    }
    const browserColor = theme === "dark" ? "#17191d" : "#ffffff";
    if (browserTheme.content !== browserColor) browserTheme.content = browserColor;
    app.querySelectorAll("img[data-chart-dark]").forEach((image) => {
      const source = theme === "dark" ? image.dataset.chartDark : image.dataset.chartLight;
      if (image.getAttribute("src") !== source) image.setAttribute("src", source);
    });
    const button = document.getElementById("sd-theme-toggle");
    if (button) {
      const label = theme === "dark" ? "Light mode" : "Dark mode";
      if (button.textContent !== label) button.textContent = label;
      button.setAttribute("aria-pressed", String(theme === "dark"));
    }
  }
  document.addEventListener("click", (event) => {
    if (!event.target.closest("#sd-theme-toggle")) return;
    theme = theme === "dark" ? "light" : "dark";
    try { localStorage.setItem(key, theme); } catch (_) {}
    apply();
  });
  new MutationObserver(apply).observe(document.documentElement, {
    childList: true, subtree: true, attributes: true, attributeFilter: ["src", "data-chart-dark"],
  });
  apply();
}());
