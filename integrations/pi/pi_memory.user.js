// ==UserScript==
// @name         Pi → Memory (auto-save)
// @namespace    agentic-memory
// @version      1.0.0
// @description  Automatically saves Pi (pi.ai) conversations to the agentic memory system
// @author       agentic-memory
// @match        https://pi.ai/*
// @grant        none
// @run-at       document-idle
// ==/UserScript==

// HOW TO INSTALL:
// 1. Install Tampermonkey: https://www.tampermonkey.net
//    (available for Chrome, Firefox, Safari, Edge)
// 2. Open Tampermonkey > Dashboard > + (new script)
// 3. Paste the entire contents of this file and save (Ctrl+S)
// 4. Start the ingest server once:
//      python3 cli.py ingest-server start
//    After that, the launchd agent keeps it running automatically if you ran install.sh.
//
// WHAT HAPPENS:
// - On load: reads any messages already on the page (if you refreshed mid-conversation).
// - As you chat: a MutationObserver watches for new messages appearing in the DOM.
// - Auto-save: 15 seconds after the last new message, the current conversation
//   is saved to memory. This gives Pi time to finish streaming its response.
// - On page unload: saves immediately (catches the session when you close the tab).
//
// Each Pi conversation is tracked by its URL path (pi.ai/talk/abc123 → session_id
// "pi-talk-abc123"). Returning to the same URL updates the existing memory session
// rather than creating a duplicate.

(function () {
  'use strict';

  // ── Configuration ───────────────────────────────────────────────────────────
  var INGEST_URL        = 'http://localhost:7747/ingest';
  var AUTO_SAVE_DELAY   = 15000;  // ms to wait after last DOM change before saving
  var MIN_TURNS_TO_SAVE = 2;      // don't save a session with only 1 message

  // ── Session ID ──────────────────────────────────────────────────────────────
  // Derive a stable ID from the page URL so re-visits update the same row.
  function makeSessionId() {
    var path = location.pathname.replace(/^\/+|\/+$/g, '').replace(/\//g, '-');
    return 'pi-' + (path || 'unknown') + '-' + location.hostname;
  }

  // ── Selector strategies ─────────────────────────────────────────────────────
  // Same strategies as the bookmarklet — try each in order until one finds messages.
  var STRATEGIES = [
    { user: '[data-sender="human"], [data-role="user"]',
      pi:   '[data-sender="ai"],    [data-role="assistant"]' },

    { user: '[class*="user" i], [class*="human" i]',
      pi:   '[class*="pi" i],   [class*="bot" i], [class*="assistant" i]' },

    { user: '[aria-label*="You" i], [aria-label*="your" i]',
      pi:   '[aria-label*="Pi" i],  [aria-label*="assistant" i]' },

    { user: 'POSITION_RIGHT', pi: 'POSITION_LEFT' },
  ];

  // ── DOM scraping ─────────────────────────────────────────────────────────────
  function scrapeByPosition() {
    var midX   = document.documentElement.scrollWidth / 2;
    var result = [];

    Array.from(document.querySelectorAll('p, div, span')).forEach(function (el) {
      var text = (el.innerText || '').trim();
      if (text.length < 5) return;

      // Skip container elements (those that have block children).
      var hasBlockChild = Array.from(el.children).some(function (c) {
        return ['DIV', 'P', 'SECTION', 'ARTICLE'].indexOf(c.tagName) !== -1;
      });
      if (hasBlockChild) return;

      var rect = el.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) return;

      result.push({
        role:    rect.left + rect.width / 2 > midX ? 'user' : 'assistant',
        content: text,
        top:     rect.top + window.scrollY,
      });
    });

    result.sort(function (a, b) { return a.top - b.top; });
    return result.map(function (t) { return { role: t.role, content: t.content }; });
  }

  function scrape() {
    for (var i = 0; i < STRATEGIES.length; i++) {
      var strat  = STRATEGIES[i];
      var turns  = [];

      if (strat.user === 'POSITION_RIGHT') {
        turns = scrapeByPosition();
      } else {
        var userEls = Array.from(document.querySelectorAll(strat.user));
        var piEls   = Array.from(document.querySelectorAll(strat.pi));

        if (userEls.length === 0 && piEls.length === 0) continue;

        var tagged = [];
        userEls.forEach(function (el) { tagged.push({ role: 'user', el: el }); });
        piEls.forEach(function (el)   { tagged.push({ role: 'assistant', el: el }); });

        tagged.sort(function (a, b) {
          return (a.el.getBoundingClientRect().top + window.scrollY) -
                 (b.el.getBoundingClientRect().top + window.scrollY);
        });

        tagged.forEach(function (item) {
          var text = (item.el.innerText || item.el.textContent || '').trim();
          if (text) turns.push({ role: item.role, content: text });
        });
      }

      if (turns.length > 0) return turns;
    }

    return [];
  }

  // ── Ingest ───────────────────────────────────────────────────────────────────
  // lastSavedTurnCount tracks how many turns we have already sent to the server.
  // We skip saving if nothing new has arrived since the last save.
  var lastSavedTurnCount = 0;

  function save(reason) {
    var turns = scrape();

    // Nothing new to save.
    if (turns.length <= lastSavedTurnCount) return;
    if (turns.length < MIN_TURNS_TO_SAVE) return;

    var payload = {
      session_id: makeSessionId(),
      agent:      'pi',
      turns:      turns,
      started_at: new Date().toISOString(),
      metadata:   { source_url: location.href, save_reason: reason },
    };

    // Use sendBeacon for page-unload saves — it queues the request even as the
    // page is closing, which regular fetch cannot guarantee.
    if (reason === 'unload' && navigator.sendBeacon) {
      var blob = new Blob([JSON.stringify(payload)], { type: 'application/json' });
      var sent = navigator.sendBeacon(INGEST_URL, blob);
      if (sent) {
        lastSavedTurnCount = turns.length;
        console.log('[pi-memory] beacon sent on unload —', turns.length, 'turns');
      }
      return;
    }

    // Normal save: use fetch.
    fetch(INGEST_URL, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
    })
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (data.ok) {
          lastSavedTurnCount = turns.length;
          console.log('[pi-memory] saved', turns.length, 'turns (', reason, ')');
        }
      })
      .catch(function (err) {
        // Silent — we don't alert mid-conversation. Check the browser console.
        console.warn('[pi-memory] save failed (', reason, '):', err.message,
          '— is the ingest server running? python3 cli.py ingest-server start');
      });
  }

  // ── Auto-save via MutationObserver ────────────────────────────────────────────
  // We watch the entire document for changes. When Pi streams a new response,
  // many small DOM mutations happen in quick succession. We debounce them:
  // each mutation resets a timer, and we only save after the timer fires.
  // This means "15 seconds of silence after the last message fragment = save."
  var saveTimer = null;

  var observer = new MutationObserver(function () {
    // Reset the debounce timer each time something changes.
    clearTimeout(saveTimer);
    saveTimer = setTimeout(function () {
      save('auto');
    }, AUTO_SAVE_DELAY);
  });

  // Start observing — childList catches new message nodes, subtree catches text
  // being streamed into existing nodes.
  observer.observe(document.body, { childList: true, subtree: true });

  // ── Save on tab close ─────────────────────────────────────────────────────────
  window.addEventListener('beforeunload', function () {
    clearTimeout(saveTimer);
    save('unload');
  });

  // ── Initial scrape ────────────────────────────────────────────────────────────
  // If the user returns to an existing conversation (browser back, page refresh),
  // record the baseline so we know how many turns were already saved, and save
  // any that weren't (e.g. if the server was down last time).
  setTimeout(function () {
    var existing = scrape();
    if (existing.length > 0) {
      console.log('[pi-memory] found', existing.length, 'existing turns on load — saving if new');
      save('page-load');
    }
  }, 3000);  // 3-second delay gives the React app time to finish rendering

})();
