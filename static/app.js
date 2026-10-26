/* Topic 18 — accessible, resilient front-end (no framework).
   Preserves Flask contract: POST /api/predict {text} -> {sentiment, confidence, scratch_model, sklearn_model, logistic_regression, clean_text, tokens}. */
(function () {
  "use strict";
  var $ = function (id) { return document.getElementById(id); };
  var on = function (el, ev, fn) { if (el) el.addEventListener(ev, fn); };

  /* ---------- Light-only Warm Clinical theme (matches hospital; no dark toggle) ---------- */
  var root = document.documentElement;
  root.setAttribute("data-theme", "light");
  on($("printBtn"), "click", function () { window.print(); });
  var year = $("year"); if (year) year.textContent = String(new Date().getFullYear());

  /* ---------- Toasts (replaces alert()) ---------- */
  var toasts = $("toasts");
  function toast(msg, kind) {
    if (!toasts) { console.warn(msg); return; }
    var d = document.createElement("div");
    d.className = "toast" + (kind ? " " + kind : "");
    d.textContent = msg;
    toasts.appendChild(d);
    setTimeout(function () { d.remove(); }, 4200);
  }

  /* ---------- Mobile nav ---------- */
  var navToggle = $("navToggle"), primaryNav = $("primaryNav");
  on(navToggle, "click", function () {
    var open = primaryNav.classList.toggle("open");
    navToggle.setAttribute("aria-expanded", String(open));
    navToggle.setAttribute("aria-label", open ? "Close sections menu" : "Open sections menu");
  });

  /* ---------- Accessible tabs + deep links ---------- */
  var tabBtns = Array.prototype.slice.call(document.querySelectorAll(".tab-btn"));
  var panels = Array.prototype.slice.call(document.querySelectorAll(".tab-content"));
  function activate(id, focus) {
    tabBtns.forEach(function (b) {
      var active = b.dataset.tab === id;
      b.classList.toggle("active", active);
      b.setAttribute("aria-selected", String(active));
      b.tabIndex = active ? 0 : -1;
      if (active && focus) b.focus();
    });
    panels.forEach(function (p) {
      var active = p.id === id;
      p.classList.toggle("active", active);
      if (active) p.removeAttribute("hidden"); else p.setAttribute("hidden", "");
    });
    if (primaryNav) primaryNav.classList.remove("open");
    if (navToggle) { navToggle.setAttribute("aria-expanded", "false"); navToggle.setAttribute("aria-label", "Open sections menu"); }
    try { history.replaceState(null, "", "#" + id.replace("tab-", "")); } catch (e) {}
  }
  tabBtns.forEach(function (btn, i) {
    on(btn, "click", function () { activate(btn.dataset.tab, false); });
    on(btn, "keydown", function (e) {
      var j = null;
      if (e.key === "ArrowRight") j = (i + 1) % tabBtns.length;
      if (e.key === "ArrowLeft") j = (i - 1 + tabBtns.length) % tabBtns.length;
      if (e.key === "Home") j = 0;
      if (e.key === "End") j = tabBtns.length - 1;
      if (j !== null) { e.preventDefault(); activate(tabBtns[j].dataset.tab, true); }
    });
  });
  var hash = (location.hash || "").replace("#", "");
  if (hash) {
    var map = { classifier: "tab-classifier", benchmarks: "tab-benchmarks", scratch: "tab-scratch", manual: "tab-manual", visuals: "tab-visuals", report: "tab-report" };
    if (map[hash]) activate(map[hash], false);
  }

  /* ---------- Classifier ---------- */
  var tweetInput = $("tweetInput"), classifyBtn = $("classifyBtn"), clearBtn = $("clearBtn"),
    copyBtn = $("copyBtn"), latencyBadge = $("latencyBadge"), charCount = $("charCount"),
    wordCount = $("wordCount"), inputError = $("inputError"),
    placeholder = $("resultPlaceholder"), skeleton = $("skeleton"), result = $("resultContent"),
    banner = $("sentimentBanner"), emoji = $("sentimentEmoji"), label = $("sentimentLabel"),
    confTag = $("confidenceTag"), desc = $("sentimentDesc"), agreeLine = $("agreeLine"),
    negVal = $("negProbVal"), posVal = $("posProbVal"), fill = $("confidenceBarFill"),
    donutArc = $("donutArc"), donutLabel = $("donutLabel"),
    cleanEl = $("cleanTextDisplay"), tokenBox = $("tokensBadgeList"), tokenCount = $("tokenCount");
  var TW_LIMIT = 280, MAX_LEN = 2000, CIRC = 213.6, lastData = null, reqSeq = 0;

  function updateCounts() {
    if (!tweetInput) return;
    var n = tweetInput.value.length;
    var w = tweetInput.value.trim() ? tweetInput.value.trim().split(/\s+/).length : 0;
    if (charCount) {
      charCount.textContent = n + " / " + TW_LIMIT + " (max " + MAX_LEN + ")";
      charCount.classList.toggle("over", n > MAX_LEN);
    }
    if (wordCount) wordCount.textContent = w + (w === 1 ? " word" : " words");
    if (inputError) inputError.hidden = true;
    tweetInput.removeAttribute("aria-invalid");
  }
  on(tweetInput, "input", updateCounts); updateCounts();

  function setStatus(text, cls) {
    if (!latencyBadge) return;
    latencyBadge.textContent = text;
    latencyBadge.className = "status-badge" + (cls ? " " + cls : "");
  }
  function setLoading(loading) {
    if (!classifyBtn) return;
    classifyBtn.disabled = loading;
    if (loading) classifyBtn.setAttribute("aria-busy", "true");
    else classifyBtn.removeAttribute("aria-busy");
    classifyBtn.setAttribute("aria-busy", String(loading));
    classifyBtn.querySelector("span:last-child").textContent = loading ? "Analysing…" : "Classify sentiment";
    if (loading) {
      if (placeholder) placeholder.classList.add("hidden");
      if (result) result.classList.add("hidden");
      if (skeleton) skeleton.hidden = false;
      setStatus("Predicting…", "busy");
    } else if (skeleton) skeleton.hidden = true;
  }

  function pill(el, pred) {
    if (!el) return;
    var pos = pred === "Positive";
    el.className = "pill " + (pos ? "pos" : "neg");
    el.textContent = pred;
  }
  function miniBar(id, pct) {
    var el = $(id); if (el) el.style.width = Math.max(2, Math.min(100, pct)) + "%";
  }

  function render(data) {
    lastData = data;
    if (placeholder) placeholder.classList.add("hidden");
    if (skeleton) skeleton.hidden = true;
    if (result) result.classList.remove("hidden");
    if (copyBtn) copyBtn.hidden = false;

    var isPos = data.sentiment === "Positive";
    if (banner) banner.className = "verdict " + (isPos ? "pos" : "neg");
    if (emoji) emoji.textContent = isPos ? "☺" : "☹";
    if (label) label.textContent = data.sentiment.toUpperCase() + " SENTIMENT";
    // Single source of truth: backend `confidence` = mean winning-class prob.
    // Donut, banner tag and meter all derive from it — never mix LR-only numbers.
    var conf = Number(data.confidence);
    if (confTag) confTag.textContent = conf + "% avg. model confidence";
    if (desc) {
      var base = isPos
        ? "Positive tone — praise, gratitude, excitement or satisfaction."
        : "Negative tone — frustration, complaint, disappointment or grievance.";
      desc.textContent = data.warning ? base + " " + data.warning : base;
    }

    var pos = Number(data.avg_pos_prob !== undefined ? data.avg_pos_prob : (data.logistic_regression || {}).pos_prob);
    var neg = Number(data.avg_neg_prob !== undefined ? data.avg_neg_prob : (data.logistic_regression || {}).neg_prob);
    if (!isFinite(pos) || !isFinite(neg)) { pos = isPos ? conf : 100 - conf; neg = 100 - pos; }
    if (posVal) posVal.textContent = pos + "%";
    if (negVal) negVal.textContent = neg + "%";
    if (fill) fill.style.width = pos + "%";
    if (donutArc) donutArc.style.strokeDashoffset = String(CIRC * (1 - conf / 100));
    if (donutLabel) animateCount(donutLabel, conf);

    var s = data.scratch_model, k = data.sklearn_model, lr = data.logistic_regression || {};
    if (s) { pill($("scratchPredBadge"), s.prediction); var sd = $("scratchProbDetail"); if (sd) sd.textContent = "Pos " + s.pos_prob + "% · Neg " + s.neg_prob + "%"; miniBar("scratchBar", isPos ? s.pos_prob : s.neg_prob); }
    if (k) { pill($("sklearnPredBadge"), k.prediction); var kd = $("sklearnProbDetail"); if (kd) kd.textContent = "Pos " + k.pos_prob + "% · Neg " + k.neg_prob + "%"; miniBar("sklearnBar", isPos ? k.pos_prob : k.neg_prob); }
    pill($("lrPredBadge"), lr.prediction || data.sentiment);
    var ld = $("lrProbDetail"); if (ld) ld.textContent = "Pos " + (lr.pos_prob !== undefined ? lr.pos_prob : pos) + "% · Neg " + (lr.neg_prob !== undefined ? lr.neg_prob : neg) + "%";
    miniBar("lrBar", isPos ? (lr.pos_prob !== undefined ? lr.pos_prob : pos) : (lr.neg_prob !== undefined ? lr.neg_prob : neg));

    // Agreement is a vote count, separate from probability confidence.
    var agreeTxt = data.agreement || "";
    var votes = [s && s.prediction, k && k.prediction, lr.prediction || data.sentiment].filter(Boolean);
    var agree = data.agreement_count !== undefined ? data.agreement_count : votes.filter(function (v) { return v === data.sentiment; }).length;
    if (agreeLine) agreeLine.textContent = agree === 3
      ? "✓ Unanimous — all 3 models agree (" + agreeTxt + "). Avg. confidence " + conf + "% (uncalibrated mean)."
      : "◑ Split vote — " + agree + " of 3 models agree (" + votes.join(" · ") + "). Avg. confidence " + conf + "% (uncalibrated mean).";
    if (data.warning) toast(data.warning, "err");

    if (cleanEl) cleanEl.textContent = data.clean_text || "(empty after filtering)";
    if (tokenBox) {
      tokenBox.innerHTML = "";
      var toks = data.tokens || [];
      if (tokenCount) tokenCount.textContent = String(toks.length);
      if (!toks.length) tokenBox.innerHTML = '<span class="token">No informative tokens</span>';
      toks.slice(0, 60).forEach(function (t) {
        var sp = document.createElement("span");
        sp.className = "token" + (t.indexOf("sentiment") === 0 ? " marker" : "");
        sp.textContent = t;
        tokenBox.appendChild(sp);
      });
      if (toks.length > 60) {
        var more = document.createElement("span");
        more.className = "token"; more.textContent = "+" + (toks.length - 60) + " more";
        tokenBox.appendChild(more);
      }
    }
    pushHistory(data);
  }

  function animateCount(el, target) {
    if (!el) return;
    var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce) { el.textContent = target + "%"; return; }
    var t0 = null, dur = 600;
    function frame(ts) {
      if (t0 === null) t0 = ts;
      var p = Math.min(1, (ts - t0) / dur);
      var eased = 1 - Math.pow(1 - p, 3);
      el.textContent = (Math.round(target * eased * 10) / 10) + "%";
      if (p < 1) requestAnimationFrame(frame);
      else el.textContent = target + "%";
    }
    requestAnimationFrame(frame);
  }

  function runPrediction() {
    var text = (tweetInput.value || "").trim();
    if (!text) {
      if (inputError) inputError.hidden = false;
      tweetInput.setAttribute("aria-invalid", "true");
      tweetInput.focus();
      toast("Please type a tweet or pick an edge case first.", "err");
      return;
    }
    if (text.length > 2000) {
      if (inputError) { inputError.textContent = "Tweet exceeds 2000 characters."; inputError.hidden = false; }
      toast("Tweet exceeds 2000 characters.", "err");
      return;
    }
    setLoading(true);
    var t0 = performance.now();
    var seq = ++reqSeq;
    fetch("/api/predict", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text })
    }).then(function (r) { return r.json().then(function (d) { return { ok: r.ok, d: d }; }); })
      .then(function (out) {
        if (seq !== reqSeq) return; // ignore stale responses
        var ms = Math.round(performance.now() - t0);
        if (!out.ok || out.d.error) {
          setLoading(false);
          if (placeholder) placeholder.classList.remove("hidden");
          setStatus("Error", "err");
          toast("Error: " + (out.d.error || "request failed"), "err");
          return;
        }
        setLoading(false);
        var modelMs = out.d.inference_ms !== undefined ? out.d.inference_ms + " ms model" : "3 models";
        setStatus(modelMs + " · " + ms + " ms total · " + (out.d.agreement || ""), "ok");
        render(out.d);
      }).catch(function (err) {
        if (seq !== reqSeq) return;
        setLoading(false);
        if (placeholder) placeholder.classList.remove("hidden");
        setStatus("Offline", "err");
        toast("Network error: " + err.message, "err");
      });
  }

  on(classifyBtn, "click", runPrediction);
  on(tweetInput, "keydown", function (e) {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); runPrediction(); }
  });
  on(clearBtn, "click", function () {
    tweetInput.value = ""; updateCounts();
    if (result) result.classList.add("hidden");
    if (copyBtn) copyBtn.hidden = true;
    if (placeholder) placeholder.classList.remove("hidden");
    setStatus("Ready", "");
    tweetInput.focus();
  });
  document.querySelectorAll(".chip").forEach(function (c) {
    c.addEventListener("click", function () {
      tweetInput.value = c.getAttribute("data-sample") || "";
      updateCounts(); runPrediction();
    });
  });
  function copyText(t, msg) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(t).then(function () { toast(msg, "ok"); }, function () { toast("Copy failed.", "err"); });
    } else {
      var ta = document.createElement("textarea"); ta.value = t; document.body.appendChild(ta);
      ta.select(); try { document.execCommand("copy"); toast(msg, "ok"); } catch (e) { toast("Copy failed.", "err"); }
      ta.remove();
    }
  }
  on(copyBtn, "click", function () {
    if (!lastData) return;
    copyText(lastData.sentiment + " (" + lastData.confidence + "%) — " + (tweetInput.value || ""), "Result copied to clipboard.");
  });
  on($("copyCleanBtn"), "click", function () { copyText(cleanEl.textContent, "Cleaned text copied."); });

  /* ---------- History (local, max 6) ---------- */
  var hWrap = $("historyWrap"), hList = $("historyList"), hist = [];
  try { hist = JSON.parse(localStorage.getItem("t18-history") || "[]"); } catch (e) { hist = []; }
  function paintHistory() {
    if (!hWrap) return;
    hWrap.hidden = !hist.length;
    if (!hList) return;
    hList.innerHTML = "";
    hist.forEach(function (h, i) {
      var li = document.createElement("li");
      var p = document.createElement("span");
      p.className = "pill " + (h.s === "Positive" ? "pos" : "neg"); p.textContent = h.s;
      var t = document.createElement("span"); t.textContent = h.t.length > 64 ? h.t.slice(0, 64) + "…" : h.t;
      var b = document.createElement("button"); b.className = "link-btn"; b.type = "button"; b.textContent = "Re-run";
      b.addEventListener("click", function () { tweetInput.value = h.t; updateCounts(); runPrediction(); });
      li.appendChild(p); li.appendChild(t); li.appendChild(b);
      hList.appendChild(li);
    });
  }
  function pushHistory(d) {
    var t = (tweetInput.value || "").slice(0, 2000);
    if (hist.length && hist[0].t === t && hist[0].s === d.sentiment) { paintHistory(); return; }
    hist.unshift({ t: t, s: d.sentiment, c: d.confidence });
    hist = hist.slice(0, 6);
    try { localStorage.setItem("t18-history", JSON.stringify(hist)); } catch (e) {}
    paintHistory();
  }
  on($("historyClear"), "click", function () {
    hist = []; try { localStorage.removeItem("t18-history"); } catch (e) {}
    paintHistory();
    toast("History cleared.", "ok");
  });
  paintHistory();

  /* ---------- Manual calculator (live) ---------- */
  var inTN = $("inTN"), inFP = $("inFP"), inFN = $("inFN"), inTP = $("inTP");
  function fmt(n, dp) { return Number(n).toFixed(dp === undefined ? 4 : dp); }
  function recalc() {
    if (!inTN) return;
    var TN = Math.max(0, Number(inTN.value) || 0), FP = Math.max(0, Number(inFP.value) || 0),
      FN = Math.max(0, Number(inFN.value) || 0), TP = Math.max(0, Number(inTP.value) || 0);
    var N = TN + FP + FN + TP;
    $("cmN").textContent = N.toLocaleString("en-IN");
    var acc = N ? (TP + TN) / N : 0, prec = (TP + FP) ? TP / (TP + FP) : 0,
      rec = (TP + FN) ? TP / (TP + FN) : 0, spec = (TN + FP) ? TN / (TN + FP) : 0,
      f1 = (prec + rec) ? 2 * prec * rec / (prec + rec) : 0;
    $("vAcc").textContent = fmt(acc); $("vPrec").textContent = fmt(prec);
    $("vRec").textContent = fmt(rec); $("vSpec").textContent = fmt(spec); $("vF1").textContent = fmt(f1);
    $("sAcc").textContent = "(" + TP + " + " + TN + ") / " + N + " = " + fmt(acc);
    $("sPrec").textContent = TP + " / (" + TP + " + " + FP + ") = " + fmt(prec);
    $("sRec").textContent = TP + " / (" + TP + " + " + FN + ") = " + fmt(rec);
    $("sSpec").textContent = TN + " / (" + TN + " + " + FP + ") = " + fmt(spec);
    $("sF1").textContent = "2·(" + fmt(prec) + "·" + fmt(rec) + ")/(" + fmt(prec) + "+" + fmt(rec) + ") = " + fmt(f1);
  }
  [inTN, inFP, inFN, inTP].forEach(function (el) { on(el, "input", recalc); });
  on($("resetCmBtn"), "click", function () {
    inTN.value = 974; inFP.value = 26; inFN.value = 83; inTP.value = 916; recalc();
    toast("Confusion matrix reset to measured counts.", "ok");
  });
  recalc();

  /* ---------- Gallery lightbox ---------- */
  var dlg = $("lightbox"), dlgImg = $("lightboxImg"), dlgCap = $("lightboxCap"), dlgOpen = $("lightboxOpen");
  document.querySelectorAll(".fig-btn").forEach(function (b) {
    if (!b.hasAttribute("aria-label")) b.setAttribute("aria-label", "Enlarge figure: " + (b.getAttribute("data-cap") || "chart"));
    b.addEventListener("click", function () {
      var src = b.getAttribute("data-full"), cap = b.getAttribute("data-cap") || "Figure";
      if (dlgImg) { dlgImg.src = src; dlgImg.alt = cap; }
      if (dlgCap) dlgCap.textContent = cap;
      if (dlgOpen) dlgOpen.href = src;
      if (dlg && dlg.showModal) dlg.showModal(); else window.open(src, "_blank");
    });
  });
  if (dlg) dlg.addEventListener("click", function (e) {
    var r = dlg.getBoundingClientRect();
    if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) dlg.close();
  });

  /* ---------- Copy helpers: code + CSV ---------- */
  document.querySelectorAll("[data-copy-code]").forEach(function (b) {
    b.addEventListener("click", function () {
      var code = $("scratchCode"); if (code) copyText(code.textContent, "Scratch implementation copied.");
    });
  });
  document.querySelectorAll("[data-download-csv]").forEach(function (b) {
    b.addEventListener("click", function () {
      var rows = [["Algorithm", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Train", "Infer"],
        ["Custom NB (scratch)", "94.55%", "0.9724", "0.9169", "0.9438", "0.9905", "0.0006s", "0.0001s"],
        ["Multinomial NB (sklearn)", "94.55%", "0.9724", "0.9169", "0.9438", "0.9905", "0.0016s", "0.0002s"],
        ["Logistic Regression", "99.50%", "0.9940", "0.9960", "0.9950", "0.9997", "0.0124s", "0.0006s"],
        ["Linear SVC", "99.60%", "0.9940", "0.9980", "0.9960", "0.9996", "0.0102s", "0.0003s"],
        ["Random Forest", "99.60%", "0.9940", "0.9980", "0.9960", "0.9996", "0.3337s", "0.0183s"]];
      var csv = rows.map(function (r) { return r.join(","); }).join("\n");
      var a = document.createElement("a");
      a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
      a.download = "model_benchmarks_topic18.csv"; a.click();
      setTimeout(function () { URL.revokeObjectURL(a.href); }, 2000);
    });
  });
})();
