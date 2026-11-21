/* Light/dark toggle for every app built with make_app(theme=True), saved per browser;
   light is the initial default. Each app is its own origin, so each remembers its own. */
(function () {
  const key = "views-theme";
  let theme = "light";
  try {
    // the signal dashboard's key before the theme was shared
    const saved = localStorage.getItem(key) || localStorage.getItem("signal-dashboard-theme");
    theme = saved === "dark" ? "dark" : "light";
  } catch (_) {}
  function apply() {
    const app = document.querySelector(".themed");
    if (!app) return;
    document.documentElement.dataset.theme = theme;
    // Supporting mobile browsers use this for the address/tab bar and edge chrome.
    let browserTheme = document.querySelector('meta[name="theme-color"]');
    if (!browserTheme) {
      browserTheme = document.createElement("meta");
      browserTheme.name = "theme-color";
      document.head.appendChild(browserTheme);
    }
    const page = getComputedStyle(document.documentElement).getPropertyValue("--theme-page").trim();
    const browserColor = theme === "dark" && page ? page : "#ffffff";
    if (browserTheme.content !== browserColor) browserTheme.content = browserColor;
    app.querySelectorAll("img[data-chart-dark]").forEach((image) => {
      const source = theme === "dark" ? image.dataset.chartDark : image.dataset.chartLight;
      if (image.getAttribute("src") !== source) image.setAttribute("src", source);
    });
    const button = document.getElementById("theme-toggle");
    if (button) {
      const label = theme === "dark" ? "Light mode" : "Dark mode";
      if (button.textContent !== label) button.textContent = label;
      button.setAttribute("aria-pressed", String(theme === "dark"));
    }
  }
  document.addEventListener("click", (event) => {
    if (!event.target.closest("#theme-toggle")) return;
    theme = theme === "dark" ? "light" : "dark";
    try { localStorage.setItem(key, theme); } catch (_) {}
    apply();
  });
  new MutationObserver(apply).observe(document.documentElement, {
    childList: true, subtree: true, attributes: true, attributeFilter: ["src", "data-chart-dark"],
  });
  apply();
}());
