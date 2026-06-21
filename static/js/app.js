(function () {
  "use strict";

  // Animated counters on hero stats
  function animateCounters() {
    document.querySelectorAll(".hero-stat-value[data-count]").forEach((el) => {
      const target = parseFloat(el.dataset.count);
      const isDecimal = target % 1 !== 0;
      const duration = 1800;
      const start = performance.now();

      function update(now) {
        const progress = Math.min((now - start) / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = target * eased;
        el.textContent = isDecimal ? current.toFixed(1) : Math.round(current);
        if (progress < 1) requestAnimationFrame(update);
      }

      requestAnimationFrame(update);
    });
  }

  // Intersection observer for counter animation
  const heroStats = document.querySelector(".hero-stats");
  if (heroStats) {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            animateCounters();
            observer.disconnect();
          }
        });
      },
      { threshold: 0.5 }
    );
    observer.observe(heroStats);
  }

  // Multi-step form navigation
  let currentStep = 1;

  function goToStep(step) {
    currentStep = step;

    document.querySelectorAll(".form-step").forEach((panel) => {
      panel.classList.toggle(
        "active",
        parseInt(panel.dataset.panel, 10) === step
      );
    });

    document.querySelectorAll(".progress-step").forEach((el) => {
      const s = parseInt(el.dataset.step, 10);
      el.classList.toggle("active", s === step);
      el.classList.toggle("done", s < step);
    });
  }

  document.querySelectorAll(".btn-next").forEach((btn) => {
    btn.addEventListener("click", () => {
      const next = parseInt(btn.dataset.next, 10);
      const panel = btn.closest(".form-step");
      const inputs = panel.querySelectorAll("input, select");
      let valid = true;

      inputs.forEach((input) => {
        if (!input.checkValidity()) {
          input.reportValidity();
          valid = false;
        }
      });

      if (valid) goToStep(next);
    });
  });

  document.querySelectorAll(".btn-prev").forEach((btn) => {
    btn.addEventListener("click", () => {
      goToStep(parseInt(btn.dataset.prev, 10));
    });
  });

  // Form submission
  const form = document.getElementById("predict-form");
  const resultPanel = document.getElementById("result-panel");
  const placeholder = resultPanel.querySelector(".result-placeholder");
  const resultContent = resultPanel.querySelector(".result-content");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const submitBtn = form.querySelector(".btn-submit");
    const btnText = submitBtn.querySelector(".btn-text");
    const btnLoader = submitBtn.querySelector(".btn-loader");

    submitBtn.disabled = true;
    btnText.hidden = true;
    btnLoader.hidden = false;

    const formData = new FormData(form);
    const payload = Object.fromEntries(formData.entries());

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error || "Prediction failed");
      }

      showResult(data);
    } catch (err) {
      alert("Error: " + err.message);
    } finally {
      submitBtn.disabled = false;
      btnText.hidden = false;
      btnLoader.hidden = true;
    }
  });

  function showResult(data) {
    placeholder.hidden = true;
    resultContent.hidden = false;

    const badge = document.getElementById("result-badge");
    const title = document.getElementById("result-title");
    const subtitle = document.getElementById("result-subtitle");
    const icon = document.getElementById("result-icon");
    const confidenceValue = document.getElementById("confidence-value");
    const confidenceFill = document.getElementById("confidence-fill");
    const probApproved = document.getElementById("prob-approved");

    const approved = data.approved;
    const confidence = Math.round(data.confidence * 100);

    badge.className = "result-badge " + (approved ? "approved" : "declined");
    title.textContent = approved ? "Approved" : "Declined";
    subtitle.textContent = approved
      ? "Your application meets our criteria"
      : "Application does not meet current criteria";
    icon.textContent = approved ? "✓" : "✕";

    confidenceValue.textContent = confidence + "%";
    confidenceFill.style.width = "0";
    requestAnimationFrame(() => {
      confidenceFill.style.width = confidence + "%";
    });

    probApproved.textContent =
      (data.probability_approved * 100).toFixed(1) + "%";

    resultPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  // Smooth nav background on scroll
  const nav = document.querySelector(".nav");
  window.addEventListener(
    "scroll",
    () => {
      nav.style.background =
        window.scrollY > 50
          ? "rgba(6, 8, 15, 0.92)"
          : "rgba(6, 8, 15, 0.7)";
    },
    { passive: true }
  );
})();
