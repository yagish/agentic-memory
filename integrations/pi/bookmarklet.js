// pi_bookmarklet.js — click to save the current Pi conversation to your memory system.
//
// HOW TO INSTALL (one-time setup):
// 1. Start the ingest server:
//      python3 cli.py ingest-server start
// 2. Create a new bookmark in your browser (any folder, any name — e.g. "Save to Memory").
// 3. Paste the ENTIRE contents of bookmarklet_url.txt as the bookmark URL.
// 4. Navigate to pi.ai, have a conversation, then click the bookmark.
//
// HOW IT WORKS:
// The bookmarklet runs this code in the context of the Pi page, reads the
// conversation from the DOM, and POSTs it to the local ingest server at
// http://localhost:7747/ingest. The session then appears in your memory and
// will be surfaced in future Claude sessions when the topic is relevant.
//
// TROUBLESHOOTING:
// If nothing is saved, open the browser console (F12 > Console tab) after
// clicking the bookmark — it will print which selector strategy it tried.
// See "Selector strategies" below if you need to update the selectors.

(function () {
  'use strict';

  // ── Where the ingest server listens ────────────────────────────────────────
  var INGEST_URL = 'http://localhost:7747/ingest';

  // ── Selector strategies ────────────────────────────────────────────────────
  // Pi.ai is a React app. React often generates unpredictable CSS class names,
  // so we try several approaches in order and use the first one that finds messages.
  //
  // If all strategies fail, open the browser console (F12) on a Pi conversation
  // page and run: document.querySelectorAll('[class*="message"]')
  // Look at the elements it returns and find the CSS class or data attribute
  // that distinguishes user messages from Pi's responses. Then add a new entry
  // to the STRATEGIES array below.
  var STRATEGIES = [
    // Strategy 1 — stable data attributes (most reliable if Pi uses them)
    { user: '[data-sender="human"], [data-role="user"]',
      pi:   '[data-sender="ai"], [data-role="assistant"]' },

    // Strategy 2 — common class name fragments (React class names vary, but
    // the substring often stays the same even when the hash changes)
    { user: '[class*="user" i], [class*="human" i], [class*="you" i]',
      pi:   '[class*="pi" i], [class*="bot" i], [class*="assistant" i]' },

    // Strategy 3 — ARIA labels (accessibility-first apps often add these)
    { user: '[aria-label*="You" i], [aria-label*="your" i]',
      pi:   '[aria-label*="Pi" i], [aria-label*="assistant" i]' },

    // Strategy 4 — visual position fallback.
    // In most chat UIs, user messages are right-aligned and assistant messages
    // are left-aligned. We detect this by comparing each element's horizontal
    // midpoint to the page midpoint.
    // This strategy uses 'ALL_MESSAGES' as a special key — see code below.
    { user: 'POSITION_RIGHT', pi: 'POSITION_LEFT' },
  ];

  // ── DOM scraping ────────────────────────────────────────────────────────────
  function scrape() {
    // Try each selector strategy.
    for (var i = 0; i < STRATEGIES.length; i++) {
      var strat = STRATEGIES[i];
      var turns = [];

      if (strat.user === 'POSITION_RIGHT') {
        // Position-based fallback: find all leaf text elements in the main
        // scrollable area and classify them by horizontal position.
        turns = scrapeByPosition();
      } else {
        var userEls = Array.from(document.querySelectorAll(strat.user));
        var piEls   = Array.from(document.querySelectorAll(strat.pi));

        if (userEls.length === 0 && piEls.length === 0) {
          console.log('[pi-memory] strategy', i + 1, ': no matches for', strat.user, 'or', strat.pi);
          continue;
        }

        // Tag each element with its role and vertical position, then sort so
        // the turns come out in conversation order (top-to-bottom on the page).
        var tagged = [];
        userEls.forEach(function (el) {
          tagged.push({ role: 'user', el: el });
        });
        piEls.forEach(function (el) {
          tagged.push({ role: 'assistant', el: el });
        });
        tagged.sort(function (a, b) {
          // scrollTop + getBoundingClientRect().top gives the absolute page position.
          return (a.el.getBoundingClientRect().top + window.scrollY) -
                 (b.el.getBoundingClientRect().top + window.scrollY);
        });

        tagged.forEach(function (item) {
          var text = (item.el.innerText || item.el.textContent || '').trim();
          if (text) {
            turns.push({ role: item.role, content: text });
          }
        });
      }

      if (turns.length > 0) {
        console.log('[pi-memory] strategy', i + 1, ': found', turns.length, 'turns');
        return turns;
      }
    }

    return [];
  }

  // Position-based fallback: walk the DOM looking for text blocks and classify
  // them as user (right-aligned) or Pi (left-aligned) by horizontal position.
  function scrapeByPosition() {
    var pageWidth = document.documentElement.scrollWidth;
    var midX = pageWidth / 2;
    var turns = [];

    // Collect all paragraph-like elements with meaningful text.
    var candidates = Array.from(document.querySelectorAll('p, div, span'))
      .filter(function (el) {
        // Skip tiny or invisible elements.
        var text = (el.innerText || '').trim();
        if (text.length < 5) return false;
        // Skip elements that contain child block elements — we want leaf nodes.
        var hasBlockChild = Array.from(el.children).some(function (c) {
          var tag = c.tagName;
          return tag === 'DIV' || tag === 'P' || tag === 'SECTION';
        });
        return !hasBlockChild;
      });

    candidates.forEach(function (el) {
      var rect = el.getBoundingClientRect();
      // Skip off-screen elements (e.g. lazy-loaded or hidden).
      if (rect.width === 0 || rect.height === 0) return;

      var centerX = rect.left + rect.width / 2;
      var role = centerX > midX ? 'user' : 'assistant';
      var text = (el.innerText || el.textContent || '').trim();
      if (text) {
        turns.push({
          role: role,
          content: text,
          top: rect.top + window.scrollY,
        });
      }
    });

    // Sort by vertical page position.
    turns.sort(function (a, b) { return a.top - b.top; });

    // Remove the 'top' field — the ingest server doesn't need it.
    return turns.map(function (t) { return { role: t.role, content: t.content }; });
  }

  // ── Generate a stable session ID from the page URL ─────────────────────────
  // If Pi puts the conversation ID in the URL (e.g. pi.ai/talk/abc123), we use
  // that so the same conversation always maps to the same memory session.
  // Otherwise we fall back to a timestamp-based ID.
  function makeSessionId() {
    var path = location.pathname.replace(/^\/+|\/+$/g, '').replace(/\//g, '-');
    if (path && path !== '' && path !== 'talk') {
      return 'pi-' + path;
    }
    return 'pi-' + Date.now();
  }

  // ── Main ────────────────────────────────────────────────────────────────────
  var turns = scrape();

  if (turns.length === 0) {
    alert(
      '[pi-memory] Could not find any messages on this page.\n\n' +
      'Make sure you are on a Pi conversation page, then try again.\n\n' +
      'For debugging: open the browser console (F12 > Console) and look\n' +
      'for the "[pi-memory] strategy N: no matches" lines to see which\n' +
      'selectors were tried. Update STRATEGIES in integrations/pi/bookmarklet.js\n' +
      'if needed and regenerate the bookmark URL.'
    );
    return;
  }

  var payload = {
    session_id: makeSessionId(),
    agent: 'pi',
    turns: turns,
    started_at: new Date().toISOString(),
    metadata: { source_url: location.href },
  };

  // POST to the local ingest server. fetch() works cross-origin because we
  // added CORSMiddleware to ingest_server.py that allows https://pi.ai.
  fetch(INGEST_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
    .then(function (r) { return r.json(); })
    .then(function (data) {
      if (data.ok) {
        alert('✓ Saved to memory: ' + turns.length + ' turns from Pi conversation.\n\nThis session will be surfaced in future Claude sessions when the topic is relevant.');
      } else {
        alert('[pi-memory] Server returned an error: ' + JSON.stringify(data));
      }
    })
    .catch(function (err) {
      alert(
        '[pi-memory] Could not reach the ingest server.\n\n' +
        'Start it with:\n' +
        '  python3 cli.py ingest-server start\n\n' +
        'Then click this bookmark again.\n\n' +
        'Error: ' + err.message
      );
    });
})();
