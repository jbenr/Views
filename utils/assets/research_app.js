/* Copy a dashboard card's current chart grid as one clipboard-ready PNG.
   This uses event delegation because the chart/card contents are replaced by
   Dash callbacks. A direct listener on the button would disappear on refresh. */
(function () {
  const buttonPrefix = "copy-charts-";
  const statusPrefix = "copy-status-";
  const chartPrefix = "card-charts-";

  function imageReady(image) {
    if (image.complete && image.naturalWidth) return Promise.resolve();
    return new Promise((resolve, reject) => {
      image.addEventListener("load", resolve, { once: true });
      image.addEventListener("error", reject, { once: true });
    });
  }

  function pngBlob(canvas) {
    return new Promise((resolve, reject) => {
      canvas.toBlob((blob) => {
        if (blob) resolve(blob);
        else reject(new Error("The chart image could not be encoded."));
      }, "image/png");
    });
  }

  document.addEventListener("click", async (event) => {
    const button = event.target.closest(".copy-charts-btn");
    if (!button || !button.id.startsWith(buttonPrefix)) return;

    const suffix = button.id.slice(buttonPrefix.length);
    const status = document.getElementById(statusPrefix + suffix);
    const charts = document.getElementById(chartPrefix + suffix);
    const images = charts ? Array.from(charts.querySelectorAll("img")) : [];
    if (!images.length) {
      if (status) status.textContent = "No charts";
      return;
    }

    button.disabled = true;
    if (status) status.textContent = "Copying…";
    try {
      await Promise.all(images.map(imageReady));
      const padding = 20;
      const columns = 2;
      const cellWidth = Math.max(...images.map((image) => image.naturalWidth));
      const rows = [];
      for (let i = 0; i < images.length; i += columns) {
        rows.push(images.slice(i, i + columns));
      }
      const rowHeights = rows.map((row) => Math.max(
        ...row.map((image) => image.naturalHeight),
      ));
      const canvas = document.createElement("canvas");
      canvas.width = columns * cellWidth + (columns + 1) * padding;
      canvas.height = rowHeights.reduce((total, height) => total + height, 0)
        + (rows.length + 1) * padding;
      const context = canvas.getContext("2d");
      context.fillStyle = "#ffffff";
      context.fillRect(0, 0, canvas.width, canvas.height);

      let y = padding;
      rows.forEach((row, rowIndex) => {
        row.forEach((image, columnIndex) => {
          const x = padding + columnIndex * (cellWidth + padding);
          context.drawImage(image, x, y, image.naturalWidth, image.naturalHeight);
        });
        y += rowHeights[rowIndex] + padding;
      });

      const blob = await pngBlob(canvas);
      await navigator.clipboard.write([
        new ClipboardItem({ "image/png": blob }),
      ]);
      if (status) status.textContent = "Copied";
    } catch (error) {
      console.error("Could not copy dashboard charts", error);
      if (status) status.textContent = "Copy failed";
    } finally {
      button.disabled = false;
    }
  });
}());
