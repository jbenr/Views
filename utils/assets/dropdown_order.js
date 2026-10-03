/* Dash 4 groups selected options first. Keep numeric menus ascending without
   moving React-owned nodes; match keyboard navigation to the visual order. */
(function () {
  const selector = '.dash-dropdown-options';
  function numericRows(list) {
    const rows = Array.from(list.children);
    const value = row => row.querySelector('input')?.value;
    if (!rows.length || !rows.every(row => value(row)?.trim() && Number.isFinite(Number(value(row))))) {
      return null;
    }
    return rows.sort((a, b) => Number(value(a)) - Number(value(b)));
  }
  function sortMenus() {
    document.querySelectorAll(selector).forEach(list => {
      const rows = numericRows(list);
      list.classList.toggle('viz-numeric-options', !!rows);
      Array.from(list.children).forEach(row => { row.style.order = ''; });
      if (rows) rows.forEach((row, index) => { row.style.order = index; });
    });
  }
  function start() {
    new MutationObserver(sortMenus).observe(document.body, {
      childList: true, subtree: true, attributes: true, attributeFilter: ['value']
    });
    sortMenus();
  }
  document.addEventListener('keydown', event => {
    const list = event.target.closest('.viz-numeric-options');
    if (!list || !['ArrowDown', 'ArrowUp', 'Home', 'End', 'Tab'].includes(event.key)) return;
    const inputs = numericRows(list).map(row => row.querySelector('input')).filter(input => !input.disabled);
    const current = inputs.indexOf(event.target);
    if (current < 0) return;
    let next = current + (event.key === 'ArrowUp' || (event.key === 'Tab' && event.shiftKey) ? -1 : 1);
    if (event.key === 'Home') next = 0;
    if (event.key === 'End') next = inputs.length - 1;
    if (event.key === 'Tab' && (next < 0 || next >= inputs.length)) {
      // Leave from the corresponding DOM edge so native Tab exits the list.
      const domInputs = Array.from(list.querySelectorAll('input:not(:disabled)'));
      domInputs[next < 0 ? 0 : domInputs.length - 1].focus();
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    inputs[(next + inputs.length) % inputs.length].focus();
  }, true);
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
  else start();
})();
