document.addEventListener("DOMContentLoaded", () => {
    // 1. Tab Navigation Logic
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");

    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetId = btn.dataset.tab;
            tabBtns.forEach(b => b.classList.remove("active"));
            tabContents.forEach(c => c.classList.remove("active"));

            btn.classList.add("active");
            const targetSection = document.getElementById(targetId);
            if (targetSection) {
                targetSection.classList.add("active");
            }
        });
    });

    // 2. Interactive Prediction Logic
    const tweetInput = document.getElementById("tweetInput");
    const classifyBtn = document.getElementById("classifyBtn");
    const clearBtn = document.getElementById("clearBtn");
    const latencyBadge = document.getElementById("latencyBadge");

    const resultPlaceholder = document.getElementById("resultPlaceholder");
    const resultContent = document.getElementById("resultContent");

    const sentimentBanner = document.getElementById("sentimentBanner");
    const sentimentEmoji = document.getElementById("sentimentEmoji");
    const sentimentLabel = document.getElementById("sentimentLabel");
    const confidenceTag = document.getElementById("confidenceTag");
    const sentimentDesc = document.getElementById("sentimentDesc");

    const negProbVal = document.getElementById("negProbVal");
    const posProbVal = document.getElementById("posProbVal");
    const confidenceBarFill = document.getElementById("confidenceBarFill");

    const scratchPredBadge = document.getElementById("scratchPredBadge");
    const scratchProbDetail = document.getElementById("scratchProbDetail");
    const sklearnPredBadge = document.getElementById("sklearnPredBadge");
    const sklearnProbDetail = document.getElementById("sklearnProbDetail");
    const lrPredBadge = document.getElementById("lrPredBadge");
    const lrProbDetail = document.getElementById("lrProbDetail");

    const cleanTextDisplay = document.getElementById("cleanTextDisplay");
    const tokensBadgeList = document.getElementById("tokensBadgeList");

    // Quick Test Chips
    document.querySelectorAll(".chip").forEach(chip => {
        chip.addEventListener("click", () => {
            tweetInput.value = chip.dataset.sample;
            runPrediction();
        });
    });

    clearBtn.addEventListener("click", () => {
        tweetInput.value = "";
        resultContent.classList.add("hidden");
        resultPlaceholder.classList.remove("hidden");
        latencyBadge.textContent = "Ready";
    });

    classifyBtn.addEventListener("click", () => {
        runPrediction();
    });

    tweetInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
            runPrediction();
        }
    });

    async function runPrediction() {
        const text = tweetInput.value.trim();
        if (!text) {
            alert("Please type a tweet or select a quick sample!");
            return;
        }

        classifyBtn.disabled = true;
        classifyBtn.innerHTML = "<span>Analyzing...</span>";
        latencyBadge.textContent = "Predicting...";

        const startTime = performance.now();

        try {
            const resp = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text })
            });

            const data = await resp.json();
            const latency = Math.round(performance.now() - startTime);
            latencyBadge.textContent = `${latency}ms`;

            if (data.error) {
                alert(`Error: ${data.error}`);
                return;
            }

            renderResults(data);
        } catch (err) {
            alert(`Network or Server error: ${err.message}`);
            latencyBadge.textContent = "Error";
        } finally {
            classifyBtn.disabled = false;
            classifyBtn.innerHTML = "<span class='btn-icon'>⚡</span><span>Classify Tweet Sentiment</span>";
        }
    }

    function renderResults(data) {
        resultPlaceholder.classList.add("hidden");
        resultContent.classList.remove("hidden");

        const isPositive = data.sentiment === "Positive";

        // Update Banner
        sentimentBanner.className = `sentiment-banner ${isPositive ? "pos" : "neg"}`;
        sentimentEmoji.textContent = isPositive ? "😊" : "😞";
        sentimentLabel.textContent = `${data.sentiment.toUpperCase()} SENTIMENT`;
        confidenceTag.textContent = `${data.confidence}% Confidence`;
        sentimentDesc.textContent = isPositive
            ? "Positive sentiments detected (praise, gratitude, excitement, or satisfaction)."
            : "Negative sentiments detected (frustration, complaint, disappointment, or grievance).";

        // Update Probabilities & Meter
        const posProb = data.logistic_regression.pos_prob;
        const negProb = data.logistic_regression.neg_prob;

        posProbVal.textContent = `${posProb}%`;
        negProbVal.textContent = `${negProb}%`;
        confidenceBarFill.style.width = `${posProb}%`;

        // Update Multi-Model Consensus Cards
        updateModelCard(scratchPredBadge, scratchProbDetail, data.scratch_model);
        updateModelCard(sklearnPredBadge, sklearnProbDetail, data.sklearn_model);
        updateModelCard(lrPredBadge, lrProbDetail, data.logistic_regression);

        // Preprocessing Trace
        cleanTextDisplay.textContent = data.clean_text || "(empty after filtering)";
        tokensBadgeList.innerHTML = "";

        if (data.tokens && data.tokens.length > 0) {
            data.tokens.forEach(tok => {
                const span = document.createElement("span");
                const isMarker = tok.startsWith("sentiment");
                span.className = `token-pill ${isMarker ? "marker" : ""}`;
                span.textContent = tok;
                tokensBadgeList.appendChild(span);
            });
        } else {
            tokensBadgeList.innerHTML = "<span class='token-pill'>No informative tokens</span>";
        }
    }

    function updateModelCard(badgeEl, detailEl, modelData) {
        const isPos = modelData.prediction === "Positive";
        badgeEl.className = `pred-tag ${isPos ? "pos" : "neg"}`;
        badgeEl.textContent = modelData.prediction;
        detailEl.textContent = `Pos: ${modelData.pos_prob}% | Neg: ${modelData.neg_prob}%`;
    }
});
