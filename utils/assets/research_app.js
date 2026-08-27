/* Copy a dashboard card's current chart grid as one clipboard-ready PNG.
   This uses event delegation because the chart/card contents are replaced by
   Dash callbacks. A direct listener on the button would disappear on refresh. */
(function () {
  const buttonPrefix = "copy-charts-";
  const statusPrefix = "copy-status-";
  const chartPrefix = "card-charts-";

  function pngBlobNow(canvas) {
    // toBlob() is async. Some browsers revoke the trusted-click permission
    // before its callback runs, then reject navigator.clipboard.write().
    // Data URLs are synchronous, so this keeps the clipboard call inside the
    // original button event.
    const encoded = canvas.toDataURL("image/png").split(",", 2)[1];
    const binary = atob(encoded);
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index += 1) {
      bytes[index] = binary.charCodeAt(index);
    }
    return new Blob([bytes], { type: "image/png" });
  }

  function legacyCopyImage(canvas) {
    // Firefox and a few embedded browsers still lack ClipboardItem for image
    // data. Their legacy copy implementation can copy an image selected from
    // an editable DOM fragment, provided it runs in the original click event.
    const holder = document.createElement("div");
    holder.contentEditable = "true";
    holder.style.cssText = "position:fixed;left:-10000px;top:0;width:1px;height:1px;overflow:hidden;";
    const image = document.createElement("img");
    image.src = canvas.toDataURL("image/png");
    holder.appendChild(image);
    document.body.appendChild(holder);

    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(holder);
    selection.removeAllRanges();
    selection.addRange(range);
    const copied = document.execCommand("copy");
    selection.removeAllRanges();
    holder.remove();
    return copied;
  }

  document.addEventListener("click", (event) => {
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

    if (!images.every((image) => image.complete && image.naturalWidth)) {
      if (status) status.textContent = "Charts still loading";
      return;
    }

    button.disabled = true;
    if (status) status.textContent = "Copying…";
    try {
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

      const blob = pngBlobNow(canvas);
      // Do not await before this line: Chromium/Firefox require a direct
      // user gesture for an image clipboard write.
      if (navigator.clipboard && window.ClipboardItem) {
        navigator.clipboard.write([
          new ClipboardItem({ "image/png": blob }),
        ]).then(() => {
          if (status) status.textContent = "Copied";
        }).catch((error) => {
          console.error("Could not copy dashboard charts", error);
          if (status) {
            status.textContent = error.name === "NotAllowedError"
              ? "Allow clipboard"
              : "Copy failed";
          }
        }).finally(() => {
          button.disabled = false;
        });
      } else {
        if (status) status.textContent = legacyCopyImage(canvas)
          ? "Copied"
          : "Copy unsupported";
        button.disabled = false;
      }
    } catch (error) {
      console.error("Could not copy dashboard charts", error);
      if (status) status.textContent = "Copy failed";
      button.disabled = false;
    }
  });
}());
