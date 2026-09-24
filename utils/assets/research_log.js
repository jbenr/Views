/* Preserve the reader's message anchor; follow output only at the bottom. */
(() => {
    const states = new Map();
    const selector = '.research-work-log[data-log-key]';

    function remember(log, state) {
        state.follow = log.scrollHeight - log.clientHeight - log.scrollTop <= 2;
        state.top = log.scrollTop;
        const top = log.getBoundingClientRect().top;
        const first = Array.from(log.children).find(line => line.getBoundingClientRect().bottom > top);
        state.anchor = first ? first.dataset.lineId : null;
        state.offset = first ? first.getBoundingClientRect().top - top : 0;
    }

    function sync(log) {
        const key = log.dataset.logKey;
        let state = states.get(key);
        if (!state || state.run !== log.dataset.runId) {
            state = {run: log.dataset.runId, follow: true, top: 0, anchor: null, offset: 0};
            states.set(key, state);
        }
        const signature = [log.firstElementChild?.dataset.lineId,
                           log.lastElementChild?.dataset.lineId, log.children.length].join(':');
        if (state.node === log && state.signature === signature) return;
        if (state.node !== log) {
            log.addEventListener('scroll', () => {
                // Ignore detached nodes and stale listeners from an earlier run.
                if (state.node === log && states.get(key) === state) remember(log, state);
            }, {passive: true});
        }
        state.node = log;
        state.signature = signature;
        if (state.follow) {
            log.scrollTop = log.scrollHeight;
        } else {
            const anchor = Array.from(log.children).find(line => line.dataset.lineId === state.anchor);
            if (anchor) {
                log.scrollTop += anchor.getBoundingClientRect().top - log.getBoundingClientRect().top - state.offset;
            } else {
                // The reading position aged out of the bounded 200-message log.
                log.scrollTop = 0;
            }
        }
        remember(log, state);
    }

    function scan() {
        document.querySelectorAll(selector).forEach(sync);
    }
    function start() {
        new MutationObserver(scan).observe(document.body, {childList: true, subtree: true, characterData: true});
        scan();
    }
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start, {once: true});
    else start();
})();
