/* Follow new console output unless the user has scrolled back to read it. */
(function () {
  const following = new WeakMap();
  document.addEventListener("scroll", function (event) {
    const node = event.target;
    if (node.id === "market-bbg-output") {
      following.set(node, node.scrollHeight - node.scrollTop - node.clientHeight < 30);
    }
  }, true);
  new MutationObserver(function (changes) {
    const node = document.getElementById("market-bbg-output");
    if (!node || following.get(node) === false) return;
    const changed = changes.some(function (change) {
      return node.contains(change.target) || Array.from(change.addedNodes).some(function (added) {
        return added === node || (added.contains && added.contains(node));
      });
    });
    if (changed) node.scrollTop = node.scrollHeight;
  }).observe(document.documentElement, {childList: true, characterData: true, subtree: true});
})();
