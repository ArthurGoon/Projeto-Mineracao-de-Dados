document.addEventListener("DOMContentLoaded", () => {
    // --- DOM ELEMENTS ---
    const steps = document.querySelectorAll(".step");
    const panels = document.querySelectorAll(".visual-panel");
    const progressBar = document.getElementById("progressBar");
    const vifBeforeBtn = document.getElementById("vifBeforeBtn");
    const vifAfterBtn = document.getElementById("vifAfterBtn");
    const profileBtns = document.querySelectorAll(".profile-tab-btn[data-profile]");
    const metricViewBtns = document.querySelectorAll(".profile-tab-btn[data-metric-view]");
    const coefViewBtns = document.querySelectorAll(".profile-tab-btn[data-coef-view]");
    const shapFocusSummary = document.getElementById("shapFocusSummary");
    const bgVideo = document.getElementById("bgVideo");
    const scrollRail = document.getElementById("scrollRail");
    
    const detailDrawer = document.getElementById("detailDrawer");
    const drawerTitle = document.getElementById("drawerTitle");
    const drawerContent = document.getElementById("drawerContent");
    const drawerClose = document.getElementById("drawerClose");
    const tooltip = document.getElementById("chartTooltip");

    let activeStep = 0;
    let metricView = "mape";
    let activeCoefView = "compare";
    let keyboardNavLock = false;

    // --- INITIALIZE VISUALIZATIONS ---
    setupBackgroundVideo();
    initScrollRail();
    initCoverParticles();
    setupDeckNavigation();
    buildInterpretStrip();
    setupFullscreen();
    
    // --- INTERSECTION OBSERVER FOR SCROLLYTELLING ---
    const observerOptions = {
        root: null,
        rootMargin: "-25% 0px -35% 0px",
        threshold: 0.15
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const stepIndex = parseInt(entry.target.getAttribute("data-step"));
                setActiveStep(stepIndex);
            }
        });
    }, observerOptions);

    steps.forEach(step => observer.observe(step));

    // Handle scroll progress bar
    window.addEventListener("scroll", () => {
        const docHeight = document.documentElement.scrollHeight - window.innerHeight;
        const scrolled = (window.scrollY / docHeight) * 100;
        progressBar.style.width = scrolled + "%";
    });

    function setActiveStep(index) {
        if (index === activeStep) return;
        
        // Update narrative active class
        steps.forEach((step, i) => {
            if (i === index) {
                step.classList.add("active");
            } else {
                step.classList.remove("active");
            }
        });
        updateScrollRail(index);

        // Update active visual panel
        panels.forEach((panel) => {
            const panelNum = parseInt(panel.getAttribute("data-panel"));
            if (panelNum === index) {
                panel.classList.add("active");
                triggerPanelAnimation(panelNum);
            } else {
                panel.classList.remove("active");
            }
        });

        activeStep = index;
        updateFullscreenBtnVisibility(index);
    }

    function navigateToStep(index) {
        const nextIndex = Math.max(0, Math.min(steps.length - 1, index));
        const targetStep = steps[nextIndex];
        if (!targetStep) return;

        targetStep.scrollIntoView({ behavior: "smooth", block: "start" });
        setActiveStep(nextIndex);
    }

    // --- PANEL ANIMATION TRIGGERS ---
    function triggerPanelAnimation(panelNum) {
        closeDrawer();
        
        switch (panelNum) {
            case 1:
                drawHistogram();
                break;
            case 2:
                drawPositionSalaries();
                break;
            case 3:
                drawAgeSalaryScatter();
                break;
            case 4:
                drawCorrelationHeatmap();
                break;
            case 5:
                drawVIF(false);
                vifBeforeBtn.classList.add("active");
                vifAfterBtn.classList.remove("active");
                break;
            case 6:
                drawMetrics();
                break;
            case 7:
                drawCoefPanel(activeCoefView);
                break;
            case 8:
                buildInterpretStrip();
                drawPermutationImportance();
                break;
            case 9:
                loadSHAPProfile("curry");
                break;
            case 10:
                drawExtremeErrors();
                break;
            case 11:
                drawFitMap();
                break;
            case 12:
                drawLimitations();
                break;
        }
    }

    // --- DETAIL DRAWER & TOOLTIP HELPERS ---
    function openDrawer(title, contentHTML) {
        drawerTitle.textContent = title;
        drawerContent.innerHTML = contentHTML;
        detailDrawer.classList.add("active");
    }

    function closeDrawer() {
        detailDrawer.classList.remove("active");
    }

    drawerClose.addEventListener("click", closeDrawer);

    function showTooltip(e, text) {
        tooltip.innerHTML = text;
        tooltip.style.opacity = 1;
        moveTooltip(e);
    }

    function moveTooltip(e) {
        tooltip.style.left = `${e.clientX + 15}px`;
        tooltip.style.top = `${e.clientY - 25}px`;
    }

    function hideTooltip() {
        tooltip.style.opacity = 0;
    }

    function setupBackgroundVideo() {
        if (!bgVideo) return;

        bgVideo.play().catch(() => {
            document.body.classList.add("video-paused");
        });

        bgVideo.addEventListener("error", () => {
            document.body.classList.add("video-paused");
        });
    }

    function initScrollRail() {
        if (!scrollRail) return;

        steps.forEach((step, index) => {
            const label = step.querySelector(".step-label")?.textContent || `Seção ${index + 1}`;
            const dot = document.createElement("button");
            dot.className = `scroll-dot${index === 0 ? " active" : ""}`;
            dot.type = "button";
            dot.setAttribute("aria-label", label);
            dot.title = label;
            dot.addEventListener("click", () => {
                navigateToStep(index);
            });
            scrollRail.appendChild(dot);
        });
    }

    function updateScrollRail(index) {
        if (!scrollRail) return;
        scrollRail.querySelectorAll(".scroll-dot").forEach((dot, i) => {
            dot.classList.toggle("active", i === index);
        });
    }

    function setupDeckNavigation() {
        window.addEventListener("keydown", (event) => {
            const activeElement = document.activeElement;
            const isTyping = activeElement && ["INPUT", "TEXTAREA", "SELECT"].includes(activeElement.tagName);
            if (isTyping || keyboardNavLock) return;

            const forwardKeys = ["ArrowDown", "PageDown", " "];
            const backwardKeys = ["ArrowUp", "PageUp"];

            if (forwardKeys.includes(event.key)) {
                event.preventDefault();
                keyboardNavLock = true;
                navigateToStep(activeStep + 1);
                setTimeout(() => { keyboardNavLock = false; }, 650);
            } else if (backwardKeys.includes(event.key)) {
                event.preventDefault();
                keyboardNavLock = true;
                navigateToStep(activeStep - 1);
                setTimeout(() => { keyboardNavLock = false; }, 650);
            } else if (event.key === "Home") {
                event.preventDefault();
                navigateToStep(0);
            } else if (event.key === "End") {
                event.preventDefault();
                navigateToStep(steps.length - 1);
            }
        });
    }

    function updateFullscreenBtnVisibility(step = activeStep) {
        const btn = document.getElementById("fullscreenBtn");
        if (!btn) return;
        const isFs = !!(document.fullscreenElement || document.webkitFullscreenElement);
        btn.classList.toggle("visible", step === 0 || isFs);
    }

    function setupFullscreen() {
        const btn = document.getElementById("fullscreenBtn");
        if (!btn) return;

        const syncState = () => {
            const isFs = !!(document.fullscreenElement || document.webkitFullscreenElement);
            btn.classList.toggle("is-fullscreen", isFs);
            btn.setAttribute("aria-label", isFs ? "Sair da tela cheia" : "Entrar em tela cheia");
            btn.title = isFs ? "Sair da tela cheia" : "Tela cheia";
            updateFullscreenBtnVisibility();
        };

        btn.addEventListener("click", () => {
            const el = document.documentElement;
            const isFs = !!(document.fullscreenElement || document.webkitFullscreenElement);

            if (!isFs) {
                const request = el.requestFullscreen || el.webkitRequestFullscreen;
                if (request) request.call(el);
            } else {
                const exit = document.exitFullscreen || document.webkitExitFullscreen;
                if (exit) exit.call(document);
            }
        });

        document.addEventListener("fullscreenchange", syncState);
        document.addEventListener("webkitfullscreenchange", syncState);
        syncState();
    }

    function getChartPadding(width, height, type = "default") {
        const compact = width < 620 || height < 360;
        const base = {
            top: compact ? 18 : 24,
            right: compact ? 18 : 28,
            bottom: compact ? 34 : 42,
            left: compact ? 48 : 64
        };

        if (type === "wide-label") {
            base.left = compact ? 92 : 126;
        }
        if (type === "heatmap") {
            base.left = compact ? 82 : 112;
            base.bottom = compact ? 28 : 38;
        }
        if (type === "waterfall") {
            base.left = compact ? 16 : 24;
            base.right = compact ? 18 : 30;
            base.top = compact ? 20 : 28;
            base.bottom = compact ? 42 : 54;
        }

        return base;
    }

    // --- SVG DOM GENERATION HELPERS ---
    const svgNS = "http://www.w3.org/2000/svg";
    
    function createSVGElement(tag, attrs) {
        const el = document.createElementNS(svgNS, tag);
        for (let k in attrs) {
            el.setAttribute(k, attrs[k]);
        }
        return el;
    }

    function formatMoneyM(value) {
        return `$${(value / 1000000).toFixed(value >= 10000000 ? 0 : 1)}M`;
    }

    // --- PANEL 0: COVER PARTICLES ---
    function initCoverParticles() {
        const court = document.querySelector(".neon-court");
        if (!court) return;
        court.querySelectorAll(".court-particle").forEach(p => p.remove());

        for (let i = 0; i < 10; i++) {
            const particle = document.createElement("div");
            particle.className = "court-particle";
            particle.style.left = Math.random() * 88 + 4 + "%";
            particle.style.top = Math.random() * 88 + 4 + "%";
            
            const delay = Math.random() * 4;
            const duration = 4 + Math.random() * 5;
            particle.style.animationDelay = `${delay}s`;
            particle.style.animationDuration = `${duration}s`;
            
            court.appendChild(particle);
        }
    }

    // --- PANEL 1: HISTOGRAM (SALARY DISTRIBUTION & OUTLIERS) ---
    function drawHistogram() {
        const svg = document.getElementById("histogramSvg");
        svg.innerHTML = "";

        const hist = window.SALARY_HISTOGRAM;
        if (!hist) return;

        const width = svg.clientWidth || 500;
        const height = svg.clientHeight || 300;
        const padding = getChartPadding(width, height);
        padding.bottom = Math.max(padding.bottom, 52);

        const bins = hist.bins;
        const binsCount = bins.length;
        const maxFreq = Math.max(...bins.map(b => b.count));

        const plotW = width - padding.left - padding.right;
        const barWidth = plotW / binsCount - 6;
        const yScale = (freq) => height - padding.bottom - (freq / maxFreq) * (height - padding.top - padding.bottom - 28);

        const totalText = createSVGElement("text", {
            x: width - padding.right,
            y: padding.top + 6,
            fill: "var(--text-secondary)",
            "font-size": "10px",
            "font-family": "var(--font-mono)",
            "font-weight": "800",
            "text-anchor": "end"
        });
        totalText.textContent = `Total: ${hist.total} jogadores`;
        svg.appendChild(totalText);

        for (let i = 0; i <= maxFreq; i += Math.ceil(maxFreq / 5)) {
            const y = yScale(i);
            svg.appendChild(createSVGElement("line", {
                x1: padding.left, y1: y, x2: width - padding.right, y2: y,
                class: "chart-grid-line"
            }));
            const text = createSVGElement("text", {
                x: padding.left - 12, y: y + 4, fill: "var(--text-muted)",
                "font-size": "9px", "font-family": "var(--font-mono)", "text-anchor": "end"
            });
            text.textContent = i;
            svg.appendChild(text);
        }

        svg.appendChild(createSVGElement("line", {
            x1: padding.left, y1: height - padding.bottom,
            x2: width - padding.right, y2: height - padding.bottom,
            class: "chart-axis-line"
        }));
        svg.appendChild(createSVGElement("line", {
            x1: padding.left, y1: padding.top,
            x2: padding.left, y2: height - padding.bottom,
            class: "chart-axis-line"
        }));

        const yAxisLabel = createSVGElement("text", {
            x: 14, y: (padding.top + height - padding.bottom) / 2,
            fill: "var(--text-muted)",
            "font-size": "9px",
            "font-family": "var(--font-mono)",
            "text-anchor": "middle",
            transform: `rotate(-90 14 ${(padding.top + height - padding.bottom) / 2})`
        });
        yAxisLabel.textContent = "Nº de jogadores";
        svg.appendChild(yAxisLabel);

        bins.forEach((bin, idx) => {
            const x = padding.left + idx * (plotW / binsCount) + 3;
            const y = yScale(bin.count);
            const barHeight = height - padding.bottom - y;
            const normalCount = bin.count - (bin.outliers || 0);
            const outlierCount = bin.outliers || 0;

            if (outlierCount > 0 && normalCount > 0) {
                const normalY = yScale(normalCount);
                const normalHeight = height - padding.bottom - normalY;
                const outlierY = yScale(bin.count);
                const outlierHeight = normalY - outlierY;

                const normalBar = createSVGElement("rect", {
                    x, y: normalY, width: barWidth, height: normalHeight,
                    fill: "var(--accent-purple)", opacity: 0.7, class: "svg-bar", rx: 3
                });
                const outlierBar = createSVGElement("rect", {
                    x, y: outlierY, width: barWidth, height: outlierHeight,
                    fill: "var(--accent-rose)", opacity: 0.9, class: "svg-bar", rx: 3
                });
                [normalBar, outlierBar].forEach(bar => {
                    bar.addEventListener("mousemove", (e) => {
                        showTooltip(e, `<b>Faixa ${bin.label}</b><br>${bin.count} jogadores<br>Outliers removidos: ${outlierCount}<br>Base limpa após filtro: ${hist.cleanTotal}`);
                    });
                    bar.addEventListener("mouseleave", hideTooltip);
                });
                svg.appendChild(normalBar);
                svg.appendChild(outlierBar);
            } else {
                const bar = createSVGElement("rect", {
                    x, y, width: barWidth, height: barHeight,
                    fill: outlierCount > 0 ? "var(--accent-rose)" : "var(--accent-purple)",
                    opacity: outlierCount > 0 ? 0.9 : 0.7,
                    class: "svg-bar", rx: 3
                });
                bar.addEventListener("mousemove", (e) => {
                    const extra = outlierCount ? `<br>Outliers removidos: ${outlierCount}` : "";
                    showTooltip(e, `<b>Faixa ${bin.label}</b><br>${bin.count} jogadores${extra}`);
                });
                bar.addEventListener("mouseleave", hideTooltip);
                svg.appendChild(bar);
            }

            const countText = createSVGElement("text", {
                x: x + barWidth / 2,
                y: y - 6,
                fill: "var(--text-primary)",
                "font-size": "10px",
                "font-family": "var(--font-mono)",
                "font-weight": "800",
                "text-anchor": "middle"
            });
            countText.textContent = bin.count;
            svg.appendChild(countText);

            const xLabel = createSVGElement("text", {
                x: x + barWidth / 2,
                y: height - padding.bottom + 14,
                fill: "var(--text-muted)",
                "font-size": "8px",
                "font-family": "var(--font-mono)",
                "text-anchor": "end",
                transform: `rotate(-32 ${x + barWidth / 2} ${height - padding.bottom + 14})`
            });
            xLabel.textContent = `US$ ${bin.label}`;
            svg.appendChild(xLabel);
        });

        const xAxisLabel = createSVGElement("text", {
            x: padding.left + plotW / 2,
            y: height - 8,
            fill: "var(--text-muted)",
            "font-size": "9px",
            "font-family": "var(--font-mono)",
            "text-anchor": "middle"
        });
        xAxisLabel.textContent = "Faixa salarial anual";
        svg.appendChild(xAxisLabel);
    }

    // --- PANEL 2: EDA - POSITION SALARIES BAR CHART ---
    const posSalaries = [
        { pos: "PG (Armador)", val: 13.1, color: "var(--accent-purple)", leader: "Stephen Curry ($48.0M)" },
        { pos: "PF (Ala-Pivô)", val: 9.7, color: "var(--accent-blue)", leader: "Giannis Antetokounmpo ($42.4M)" },
        { pos: "SF (Ala)", val: 8.8, color: "var(--accent-teal)", leader: "Kevin Durant ($44.1M)" },
        { pos: "SG (Ala-Armador)", val: 7.7, color: "var(--accent-orange)", leader: "Bradley Beal ($43.2M)" },
        { pos: "C (Pivô)", val: 7.5, color: "var(--accent-pink)", leader: "Rudy Gobert ($38.1M)" }
    ];

    function drawPositionSalaries() {
        const svg = document.getElementById("positionSvg");
        svg.innerHTML = "";
        
        const width = svg.clientWidth || 500;
        const height = svg.clientHeight || 300;
        const padding = getChartPadding(width, height);
        
        const maxSal = 15.0; // max scale 15M
        
        const xScale = (idx) => padding.left + (idx / posSalaries.length) * (width - padding.left - padding.right);
        const yScale = (val) => height - padding.bottom - (val / maxSal) * (height - padding.top - padding.bottom);
        
        // Grid Lines
        for (let i = 0; i <= maxSal; i += 3) {
            const y = yScale(i);
            const grid = createSVGElement("line", {
                x1: padding.left, y1: y, x2: width - padding.right, y2: y,
                class: "chart-grid-line"
            });
            svg.appendChild(grid);
            
            const text = createSVGElement("text", {
                x: padding.left - 12, y: y + 4, fill: "var(--text-muted)",
                "font-size": "9px", "font-family": "var(--font-mono)", "text-anchor": "end"
            });
            text.textContent = `$${i}M`;
            svg.appendChild(text);
        }

        // Axes
        const xAxis = createSVGElement("line", {
            x1: padding.left, y1: height - padding.bottom,
            x2: width - padding.right, y2: height - padding.bottom,
            class: "chart-axis-line"
        });
        svg.appendChild(xAxis);

        const barWidth = (width - padding.left - padding.right) / posSalaries.length - 20;

        posSalaries.forEach((d, idx) => {
            const x = xScale(idx) + 10;
            const y = yScale(d.val);
            const barHeight = height - padding.bottom - y;
            
            const bar = createSVGElement("rect", {
                x: x, y: y, width: barWidth, height: barHeight,
                fill: d.color, opacity: 0.75, class: "svg-bar", rx: 4
            });
            svg.appendChild(bar);

            const valueText = createSVGElement("text", {
                x: x + barWidth / 2,
                y: y - 10,
                fill: d.color,
                "font-size": "12px",
                "font-family": "var(--font-mono)",
                "font-weight": "900",
                "text-anchor": "middle"
            });
            valueText.textContent = `$${d.val.toFixed(1)}M`;
            svg.appendChild(valueText);
            
            // X Label
            const text = createSVGElement("text", {
                x: x + barWidth/2, y: height - padding.bottom + 18, fill: "var(--text-primary)",
                "font-size": "9.5px", "font-family": "var(--font-sans)", "font-weight": "600", "text-anchor": "middle"
            });
            text.textContent = d.pos.split(' ')[0];
            svg.appendChild(text);

            // Hover tooltip
            bar.addEventListener("mousemove", (e) => {
                showTooltip(e, `<b>Posição:</b> ${d.pos}<br><b>Média Salarial:</b> $${d.val}M<br><b>Líder:</b> ${d.leader}`);
            });
            bar.addEventListener("mouseleave", hideTooltip);
            bar.addEventListener("click", () => {
                openDrawer(d.pos, `
                    <div class="drawer-stat-grid">
                        <div class="drawer-stat-card"><div class="val">$${d.val}M</div><div class="lbl">Média Salarial</div></div>
                        <div class="drawer-stat-card"><div class="val">${d.pos.split(" ")[0]}</div><div class="lbl">Posição</div></div>
                    </div>
                    <div class="narrative-alert purple" style="margin-top: 1rem;">
                        <div class="alert-title">Líder da Faixa</div>
                        ${d.leader}
                    </div>
                `);
            });
            
        });
    }

    // --- PANEL 3: EDA - AGE VS SALARY SCATTER PLOT ---
    function drawAgeSalaryScatter() {
        const svg = document.getElementById("ageSalarySvg");
        svg.innerHTML = "";
        
        const width = svg.clientWidth || 500;
        const height = svg.clientHeight || 300;
        const padding = getChartPadding(width, height);
        
        const minAge = 19, maxAge = 40;
        const maxSal = 50000000;
        
        const xScale = (age) => padding.left + ((age - minAge) / (maxAge - minAge)) * (width - padding.left - padding.right);
        const yScale = (sal) => height - padding.bottom - (sal / maxSal) * (height - padding.top - padding.bottom);
        
        // Grid lines Y
        for (let i = 0; i <= maxSal; i += 10000000) {
            const y = yScale(i);
            const grid = createSVGElement("line", {
                x1: padding.left, y1: y, x2: width - padding.right, y2: y,
                class: "chart-grid-line"
            });
            svg.appendChild(grid);
            
            const text = createSVGElement("text", {
                x: padding.left - 12, y: y + 4, fill: "var(--text-muted)",
                "font-size": "9px", "font-family": "var(--font-mono)", "text-anchor": "end"
            });
            text.textContent = `$${i/1000000}M`;
            svg.appendChild(text);
        }

        // Grid lines X (Age)
        for (let age = 20; age <= 40; age += 4) {
            const x = xScale(age);
            const grid = createSVGElement("line", {
                x1: x, y1: padding.top, x2: x, y2: height - padding.bottom,
                class: "chart-grid-line"
            });
            svg.appendChild(grid);

            const text = createSVGElement("text", {
                x: x, y: height - padding.bottom + 18, fill: "var(--text-muted)",
                "font-size": "9px", "font-family": "var(--font-mono)", "text-anchor": "middle"
            });
            text.textContent = `${age} anos`;
            svg.appendChild(text);
        }

        // Axes
        const xAxis = createSVGElement("line", {
            x1: padding.left, y1: height - padding.bottom,
            x2: width - padding.right, y2: height - padding.bottom,
            class: "chart-axis-line"
        });
        svg.appendChild(xAxis);

        // Draw Parabolic regression fit line (quadratic representation)
        let pathD = "";
        for (let age = minAge; age <= maxAge; age += 0.5) {
            // y = -0.16 * (age - 29.5)^2 + 19.5 (in millions)
            let sal = -0.155 * Math.pow(age - 29.8, 2) + 21.0;
            if (sal < 1.0) sal = 1.0;
            const x = xScale(age);
            const y = yScale(sal * 1000000);
            
            if (age === minAge) pathD += `M ${x} ${y}`;
            else pathD += ` L ${x} ${y}`;
        }

        const regressionLine = createSVGElement("path", {
            d: pathD, fill: "none", stroke: "var(--accent-orange)",
            "stroke-width": 2.5, "stroke-dasharray": "3 3", opacity: 0.8
        });
        svg.appendChild(regressionLine);

        // Plot dots
        window.NBA_DATA.forEach(p => {
            if (p.type === "Outlier") return; // Skip G-League outliers
            
            const cx = xScale(p.age);
            const cy = yScale(p.salary);
            
            let color = "var(--accent-purple)";
            if (p.name === "Stephen Curry") color = "var(--accent-teal)";
            else if (p.name === "Frank Kaminsky") color = "var(--accent-orange)";
            else if (p.name === "Jaden Hardy") color = "var(--accent-pink)";
            
            const isFeatured = ["Stephen Curry", "Frank Kaminsky", "Jaden Hardy"].includes(p.name);
            const radius = isFeatured ? 8 : Math.max(3.4, Math.min(6.4, 3.4 + (p.salary / maxSal) * 3.2));
            const dot = createSVGElement("circle", {
                cx: cx, cy: cy, r: radius, fill: color, opacity: isFeatured ? 0.96 : 0.48,
                class: "scatter-dot",
                stroke: isFeatured ? "rgba(255,255,255,0.86)" : "transparent",
                "stroke-width": isFeatured ? 1.4 : 0
            });
            
            // Hover
            dot.addEventListener("mousemove", (e) => {
                showTooltip(e, `<b>${p.name}</b><br>Idade: ${p.age} anos<br>Salário: $${(p.salary/1000000).toFixed(2)}M<br>PPG: ${p.pts}`);
            });
            dot.addEventListener("mouseleave", hideTooltip);
            
            // Click drawer
            dot.addEventListener("click", () => {
                const detailHTML = `
                    <div class="player-mini-card">
                        <div class="mini-avatar" style="background:${color};">${p.name[0]}</div>
                        <div class="mini-info">
                            <div class="mini-name">${p.name}</div>
                            <div class="mini-pos">${p.pos} | ${p.age} anos</div>
                        </div>
                    </div>
                    <div class="drawer-stat-grid" style="margin-top: 1rem;">
                        <div class="drawer-stat-card"><div class="val">$${(p.salary/1000000).toFixed(2)}M</div><div class="lbl">Salário</div></div>
                        <div class="drawer-stat-card"><div class="val">${p.mp}</div><div class="lbl">Minutos</div></div>
                        <div class="drawer-stat-card"><div class="val">${p.pts}</div><div class="lbl">Pontos/GP</div></div>
                        <div class="drawer-stat-card"><div class="val">${p.usg}%</div><div class="lbl">USG%</div></div>
                    </div>
                `;
                openDrawer(p.name, detailHTML);
            });
            
            svg.appendChild(dot);

            if (isFeatured) {
                const label = createSVGElement("text", {
                    x: Math.min(width - padding.right - 20, cx + 11),
                    y: Math.max(padding.top + 12, cy - 10),
                    fill: color,
                    "font-size": "11px",
                    "font-family": "var(--font-sans)",
                    "font-weight": "900"
                });
                label.textContent = p.name.split(" ").slice(-1)[0];
                svg.appendChild(label);
            }
        });

        const curveLabel = createSVGElement("text", {
            x: xScale(30.5),
            y: yScale(22500000),
            fill: "var(--accent-orange)",
            "font-size": "11px",
            "font-family": "var(--font-mono)",
            "font-weight": "800",
            "text-anchor": "middle"
        });
        curveLabel.textContent = "pico salarial estimado";
        svg.appendChild(curveLabel);
    }

    // --- PANEL 4: EDA - CORRELATION HEATMAP ---
    const heatmapLabels = ['Salário', 'Minutos (MP)', 'Idade', 'Pontos (PTS)', 'Eficiência', 'Uso (USG%)'];
    const corrMatrix = [
        [1.00,  0.74,  0.22,  0.37,  0.42,  0.32],
        [0.74,  1.00,  0.18,  0.82,  0.64,  0.55],
        [0.22,  0.18,  1.00,  0.12,  0.15,  0.08],
        [0.37,  0.82,  0.12,  1.00,  0.58,  0.78],
        [0.42,  0.64,  0.15,  0.58,  1.00,  0.48],
        [0.32,  0.55,  0.08,  0.78,  0.48,  1.00]
    ];

    function drawCorrelationHeatmap() {
        const svg = document.getElementById("heatmapSvg");
        svg.innerHTML = "";
        
        const width = svg.clientWidth || 500;
        const height = svg.clientHeight || 300;
        const padding = getChartPadding(width, height, "heatmap");
        
        const legendWidth = width > 720 ? 72 : 0;
        const size = Math.min((width - padding.left - padding.right - legendWidth), (height - padding.top - padding.bottom - 8));
        const cellSize = size / heatmapLabels.length;
        
        // Center the heatmap
        const startX = padding.left + ((width - padding.left - padding.right) - size)/2;
        const startY = padding.top;

        // Draw heat cells
        for (let r = 0; r < heatmapLabels.length; r++) {
            for (let c = 0; c < heatmapLabels.length; c++) {
                const val = corrMatrix[r][c];
                const x = startX + c * cellSize;
                const y = startY + r * cellSize;
                
                const hue = 250 - (val * 210);
                const sat = 58 + (val * 28);
                const light = 18 + (val * 50);
                const fillStyle = `hsl(${hue}, ${sat}%, ${light}%)`;
                
                const cell = createSVGElement("rect", {
                    x: x, y: y, width: cellSize - 2, height: cellSize - 2,
                    fill: fillStyle, class: "heatmap-cell", rx: 2
                });
                
                // cell hover tooltip
                cell.addEventListener("mousemove", (e) => {
                    showTooltip(e, `<b>Correlação:</b><br>${heatmapLabels[r]} vs ${heatmapLabels[c]}<br>Coeficiente: <b>${val.toFixed(2)}</b>`);
                });
                cell.addEventListener("mouseleave", hideTooltip);
                cell.addEventListener("click", () => {
                    openDrawer("Correlação", `
                        <div class="drawer-stat-grid">
                            <div class="drawer-stat-card"><div class="val">${val.toFixed(2)}</div><div class="lbl">Coeficiente</div></div>
                            <div class="drawer-stat-card"><div class="val">${val >= 0.75 ? "Alta" : val >= 0.45 ? "Média" : "Baixa"}</div><div class="lbl">Intensidade</div></div>
                        </div>
                        <div class="narrative-alert ${val >= 0.75 ? "warning" : "purple"}" style="margin-top: 1rem;">
                            <div class="alert-title">${heatmapLabels[r]} × ${heatmapLabels[c]}</div>
                            ${val >= 0.75 ? "Relação forte: exige atenção porque pode inflar variância e confundir leitura causal." : "Relação monitorada: ajuda a entender dependências sem dominar o pipeline."}
                        </div>
                    `);
                });
                
                svg.appendChild(cell);

                const valueText = createSVGElement("text", {
                    x: x + cellSize / 2,
                    y: y + cellSize / 2 + 4,
                    fill: val >= 0.58 ? "#050505" : "rgba(255,255,255,0.92)",
                    "font-size": Math.max(9, Math.min(13, cellSize * 0.22)) + "px",
                    "font-family": "var(--font-mono)",
                    "font-weight": "800",
                    "text-anchor": "middle",
                    "pointer-events": "none"
                });
                valueText.textContent = val.toFixed(2);
                svg.appendChild(valueText);
            }

            // Row labels (Left)
            const textL = createSVGElement("text", {
                x: startX - 10, y: startY + r * cellSize + cellSize/2 + 3, fill: "var(--text-primary)",
                "font-size": "9.5px", "font-family": "var(--font-sans)", "font-weight": "600", "text-anchor": "end"
            });
            textL.textContent = heatmapLabels[r];
            svg.appendChild(textL);

            // Column labels (Bottom)
            const textB = createSVGElement("text", {
                x: startX + r * cellSize + cellSize/2, y: startY + size + 16, fill: "var(--text-secondary)",
                "font-size": "9.5px", "font-family": "var(--font-sans)", "text-anchor": "middle"
            });
            textB.textContent = heatmapLabels[r].split(' ')[0];
            svg.appendChild(textB);
        }

        if (legendWidth > 0) {
            const legendX = startX + size + 26;
            const legendY = startY;
            const legendH = size;
            const defs = createSVGElement("defs", {});
            const grad = createSVGElement("linearGradient", {
                id: "corrLegendGrad", x1: "0%", y1: "100%", x2: "0%", y2: "0%"
            });
            [
                { offset: "0%", color: "hsl(250, 58%, 18%)" },
                { offset: "55%", color: "hsl(135, 72%, 46%)" },
                { offset: "100%", color: "hsl(40, 86%, 68%)" }
            ].forEach(stop => {
                grad.appendChild(createSVGElement("stop", { offset: stop.offset, "stop-color": stop.color }));
            });
            defs.appendChild(grad);
            svg.appendChild(defs);

            const legend = createSVGElement("rect", {
                x: legendX, y: legendY, width: 10, height: legendH,
                fill: "url(#corrLegendGrad)", rx: 999
            });
            svg.appendChild(legend);

            [
                { y: legendY + 4, label: "1.00" },
                { y: legendY + legendH / 2 + 4, label: "0.50" },
                { y: legendY + legendH, label: "0.00" }
            ].forEach(item => {
                const text = createSVGElement("text", {
                    x: legendX + 18, y: item.y, fill: "var(--text-secondary)",
                    "font-size": "9px", "font-family": "var(--font-mono)"
                });
                text.textContent = item.label;
                svg.appendChild(text);
            });
        }
    }

    // --- PANEL 5: VIF BEFORE / AFTER ---
    function drawVIF(isAfter) {
        const svg = document.getElementById("vifSvg");
        svg.innerHTML = "";
        
        const width = svg.clientWidth || 500;
        const height = svg.clientHeight || 300;
        const padding = getChartPadding(width, height, "wide-label");
        
        const data = isAfter ? vifAfterData : vifBeforeData;
        const maxVif = isAfter ? 15 : 450;
        
        const yScale = (idx) => padding.top + (idx / data.length) * (height - padding.top - padding.bottom);
        const xScale = (val) => padding.left + (val / maxVif) * (width - padding.left - padding.right);
        
        const barHeight = (height - padding.top - padding.bottom) / data.length - 8;
        
        // Grid lines
        for (let i = 0; i <= maxVif; i += Math.ceil(maxVif/5)) {
            const x = xScale(i);
            const grid = createSVGElement("line", {
                x1: x, y1: padding.top, x2: x, y2: height - padding.bottom,
                class: "chart-grid-line"
            });
            svg.appendChild(grid);
            
            const text = createSVGElement("text", {
                x: x, y: height - padding.bottom + 14, fill: "var(--text-muted)",
                "font-size": "8px", "font-family": "var(--font-mono)", "text-anchor": "middle"
            });
            text.textContent = i;
            svg.appendChild(text);
        }

        data.forEach((d, idx) => {
            const y = yScale(idx);
            const x = xScale(d.val);
            const w = x - padding.left;
            
            const color = isAfter ? "var(--accent-emerald)" : "var(--accent-rose)";
            
            const bar = createSVGElement("rect", {
                x: padding.left, y: y, width: w, height: barHeight,
                fill: color, opacity: 0.7, class: "svg-horizontal-bar", rx: 3
            });
            
            const text = createSVGElement("text", {
                x: padding.left - 10, y: y + barHeight/2 + 4, fill: "var(--text-primary)",
                "font-size": "10px", "font-family": "var(--font-mono)", "text-anchor": "end"
            });
            text.textContent = d.name;
            
            const valText = createSVGElement("text", {
                x: x + 8, y: y + barHeight/2 + 4, fill: "var(--text-secondary)",
                "font-size": "10px", "font-family": "var(--font-mono)"
            });
            valText.textContent = d.val.toFixed(1);
            
            bar.addEventListener("mousemove", (e) => {
                showTooltip(e, `<b>Feature:</b> ${d.name}<br><b>VIF:</b> ${d.val}<br><b>Status:</b> ${d.status}`);
            });
            bar.addEventListener("mouseleave", hideTooltip);
            
            svg.appendChild(bar);
            svg.appendChild(text);
            svg.appendChild(valText);
        });

        const xAxis = createSVGElement("line", {
            x1: padding.left, y1: height - padding.bottom,
            x2: width - padding.right, y2: height - padding.bottom,
            class: "chart-axis-line"
        });
        svg.appendChild(xAxis);
    }

    vifBeforeBtn.addEventListener("click", () => {
        vifBeforeBtn.classList.add("active");
        vifAfterBtn.classList.remove("active");
        drawVIF(false);
    });

    vifAfterBtn.addEventListener("click", () => {
        vifBeforeBtn.classList.remove("active");
        vifAfterBtn.classList.add("active");
        drawVIF(true);
    });

    metricViewBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            metricViewBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            metricView = btn.getAttribute("data-metric-view");
            drawMetrics();
        });
    });

    coefViewBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            coefViewBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            activeCoefView = btn.getAttribute("data-coef-view");
            drawCoefPanel(activeCoefView);
        });
    });

    function buildInterpretStrip() {
        const el = document.getElementById("interpretStrip");
        if (!el || !window.modelMetrics) return;
        const shortName = (name) => {
            if (name.includes("HGB")) return "HGB";
            if (name.includes("Random")) return "RF";
            if (name.includes("Ridge")) return "Ridge";
            if (name.includes("Lasso")) return "Lasso";
            return "OLS";
        };
        el.innerHTML = modelMetrics.map((m) => {
            const isHgb = m.name.includes("HGB");
            return `<div class="interpret-chip${isHgb ? " active" : ""}" style="--chip-color:${m.color}">
                <strong>${shortName(m.name)}</strong>
                <span>${m.interpret || "—"}</span>
            </div>`;
        }).join("");
    }

    // --- PANEL 6: MODEL METRICS ---
    function drawMetrics() {
        const svg = document.getElementById("metricsSvg");
        svg.innerHTML = "";
        
        const width = svg.clientWidth || 500;
        const height = svg.clientHeight || 300;
        const padding = getChartPadding(width, height, "wide-label");
        
        const isR2 = metricView === "r2";
        const sortedMetrics = [...modelMetrics].sort((a, b) => isR2 ? b.r2 - a.r2 : a.mape - b.mape);
        const yScale = (idx) => padding.top + (idx / sortedMetrics.length) * (height - padding.top - padding.bottom);
        const xScale = (value) => {
            if (isR2) {
                return padding.left + ((value - 0.45) / (0.60 - 0.45)) * (width - padding.left - padding.right);
            }
            return padding.left + ((value - 45) / (65 - 45)) * (width - padding.left - padding.right);
        };
        
        const barHeight = (height - padding.top - padding.bottom) / sortedMetrics.length - 10;
        
        // Grid lines
        const ticks = isR2 ? [0.45, 0.50, 0.55, 0.60] : [45, 50, 55, 60, 65];
        ticks.forEach((m) => {
            const x = xScale(m);
            const grid = createSVGElement("line", {
                x1: x, y1: padding.top, x2: x, y2: height - padding.bottom,
                class: "chart-grid-line"
            });
            svg.appendChild(grid);
            
            const text = createSVGElement("text", {
                x: x, y: height - padding.bottom + 14, fill: "var(--text-muted)",
                "font-size": "8px", "font-family": "var(--font-mono)", "text-anchor": "middle"
            });
            text.textContent = isR2 ? m.toFixed(2) : `${m}%`;
            svg.appendChild(text);
        });

        sortedMetrics.forEach((d, idx) => {
            const y = yScale(idx);
            const metricValue = isR2 ? d.r2 : d.mape;
            const xVal = xScale(metricValue);
            const w = xVal - padding.left;
            const isWinner = idx === 0;
            
            const bar = createSVGElement("rect", {
                x: padding.left, y: y, width: w, height: barHeight,
                fill: isWinner ? "var(--accent-teal)" : d.color,
                opacity: isWinner ? 0.96 : 0.64,
                class: "svg-horizontal-bar",
                rx: 6
            });
            if (isWinner) {
                bar.setAttribute("filter", "drop-shadow(0px 0px 16px rgba(45,212,191,0.34))");
            }
            
            const text = createSVGElement("text", {
                x: padding.left - 10, y: y + barHeight/2 + 4, fill: "var(--text-primary)",
                "font-size": "10px", "font-family": "var(--font-sans)", "font-weight": "600", "text-anchor": "end"
            });
            text.textContent = d.name;
            
            const valText = createSVGElement("text", {
                x: xVal + 8, y: y + barHeight/2 + 4, fill: "var(--text-secondary)",
                "font-size": "10px", "font-family": "var(--font-mono)"
            });
            valText.textContent = isR2 ? `R²: ${d.r2.toFixed(3)}` : `MAPE: ${d.mape.toFixed(1)}%`;
            
            bar.addEventListener("mousemove", (e) => {
                const maeLabel = d.maeUsd ? `<br><b>MAE:</b> US$ ${(d.maeUsd / 1e6).toFixed(2)}M` : "";
                showTooltip(e, `<b>Modelo:</b> ${d.name}<br><b>MAPE:</b> ${d.mape}%<br><b>R² Teste:</b> ${d.r2}${maeLabel}<br><b>Interpretação:</b> ${d.interpret || "—"}`);
            });
            bar.addEventListener("mouseleave", hideTooltip);
            bar.addEventListener("click", () => {
                openDrawer(d.name, `
                    <div class="drawer-stat-grid">
                        <div class="drawer-stat-card"><div class="val">${d.mape.toFixed(2)}%</div><div class="lbl">MAPE</div></div>
                        <div class="drawer-stat-card"><div class="val">${d.r2.toFixed(3)}</div><div class="lbl">R² Teste</div></div>
                    </div>
                    <div class="narrative-alert ${isWinner ? "purple" : "warning"}" style="margin-top: 1rem;">
                        <div class="alert-title">${isWinner ? "Modelo de referência" : "Trade-off de performance"}</div>
                        ${isWinner ? "Melhor leitura no modo selecionado. É o candidato mais convincente para explicar o teto preditivo do dataset." : "Modelo útil para comparação, mas fica atrás do HGB no equilíbrio entre erro percentual e variância explicada."}
                    </div>
                `);
            });
            
            svg.appendChild(bar);
            svg.appendChild(text);
            svg.appendChild(valText);
        });

        // Reference threshold line
        const referenceValue = isR2 ? 0.55 : 50;
        const threshX = xScale(referenceValue);
        const refLine = createSVGElement("line", {
            x1: threshX, y1: padding.top, x2: threshX, y2: height - padding.bottom,
            stroke: "rgba(255,255,255,0.46)", "stroke-width": 1, "stroke-dasharray": "3 3"
        });
        svg.appendChild(refLine);

        const xAxis = createSVGElement("line", {
            x1: padding.left, y1: height - padding.bottom,
            x2: width - padding.right, y2: height - padding.bottom,
            class: "chart-axis-line"
        });
        svg.appendChild(xAxis);
    }

    // --- PANEL 7: COEFFICIENTS (OLS / RIDGE / LASSO / COMPARE) ---
    function drawCoefPanel(view = "compare") {
        if (view === "compare") {
            drawCoefComparison();
            return;
        }
        const dataMap = { ols: olsCoefs, ridge: ridgeCoefs, lasso: lassoCoefs };
        const labelMap = { ols: "OLS", ridge: "Ridge (L2)", lasso: "Lasso (L1)" };
        const colorMap = { ols: "var(--accent-pink)", ridge: "var(--accent-blue)", lasso: "var(--accent-orange)" };
        drawCoefBars(dataMap[view] || olsCoefs, labelMap[view] || "OLS", colorMap[view] || "var(--accent-blue)");
    }

    function drawCoefComparison() {
        const svg = document.getElementById("olsSvg");
        svg.innerHTML = "";
        const data = window.coefCompareData || [];
        const width = svg.clientWidth || 500;
        const height = svg.clientHeight || 300;
        const padding = getChartPadding(width, height, "wide-label");
        padding.left = Math.max(padding.left, width < 760 ? 130 : 160);
        padding.top = Math.max(padding.top, 36);

        const maxVal = 0.85;
        const rowH = (height - padding.top - padding.bottom) / data.length;
        const groupH = Math.min(22, rowH * 0.55);
        const barH = Math.max(4, (groupH - 6) / 3);
        const center = padding.left + (width - padding.left - padding.right) / 2;
        const xScale = (val) => center + (val / maxVal) * ((width - padding.left - padding.right) / 2);

        const legendY = padding.top - 18;
        [
            { label: "OLS", color: "var(--accent-pink)", x: padding.left },
            { label: "Ridge", color: "var(--accent-blue)", x: padding.left + 70 },
            { label: "Lasso", color: "var(--accent-orange)", x: padding.left + 150 }
        ].forEach(item => {
            svg.appendChild(createSVGElement("rect", { x: item.x, y: legendY - 6, width: 10, height: 10, fill: item.color, rx: 2 }));
            const text = createSVGElement("text", {
                x: item.x + 14, y: legendY + 2, fill: "var(--text-secondary)",
                "font-size": "9px", "font-family": "var(--font-mono)", "font-weight": "700"
            });
            text.textContent = item.label;
            svg.appendChild(text);
        });

        for (let i = -0.8; i <= 0.8; i += 0.4) {
            if (Math.abs(i) < 0.01) continue;
            const x = xScale(i);
            svg.appendChild(createSVGElement("line", {
                x1: x, y1: padding.top, x2: x, y2: height - padding.bottom, class: "chart-grid-line"
            }));
        }
        svg.appendChild(createSVGElement("line", {
            x1: center, y1: padding.top, x2: center, y2: height - padding.bottom,
            stroke: "rgba(255,255,255,0.3)", "stroke-width": 2
        }));

        const models = [
            { key: "ols", color: "var(--accent-pink)" },
            { key: "ridge", color: "var(--accent-blue)" },
            { key: "lasso", color: "var(--accent-orange)" }
        ];

        data.forEach((d, idx) => {
            const rowTop = padding.top + idx * rowH + (rowH - groupH) / 2;
            const label = createSVGElement("text", {
                x: padding.left - 12, y: rowTop + groupH / 2 + 4,
                fill: d.warning ? "var(--accent-rose)" : "var(--text-primary)",
                "font-size": "10px", "font-family": "var(--font-sans)", "font-weight": "600", "text-anchor": "end"
            });
            label.textContent = d.name;
            svg.appendChild(label);

            models.forEach((m, mi) => {
                const val = d[m.key];
                const y = rowTop + mi * (barH + 2);
                const xEnd = xScale(val);
                let xPos, w;
                if (val >= 0) { xPos = center; w = xEnd - center; }
                else { xPos = xEnd; w = center - xEnd; }
                const bar = createSVGElement("rect", {
                    x: xPos, y, width: Math.max(1, w), height: barH,
                    fill: m.color, opacity: 0.82, rx: 2, class: "svg-horizontal-bar"
                });
                bar.addEventListener("mousemove", (e) => {
                    showTooltip(e, `<b>${d.name}</b> · ${m.key.toUpperCase()}<br>Coeficiente: ${val >= 0 ? "+" : ""}${val.toFixed(2)}`);
                });
                bar.addEventListener("mouseleave", hideTooltip);
                svg.appendChild(bar);
            });
        });

        svg.appendChild(createSVGElement("line", {
            x1: padding.left, y1: height - padding.bottom,
            x2: width - padding.right, y2: height - padding.bottom, class: "chart-axis-line"
        }));
    }

    function drawCoefBars(coefData, modelLabel, accentColor) {
        const svg = document.getElementById("olsSvg");
        svg.innerHTML = "";
        
        const width = svg.clientWidth || 500;
        const height = svg.clientHeight || 300;
        const padding = getChartPadding(width, height, "wide-label");
        
        const center = padding.left + (width - padding.left - padding.right) / 2;
        const maxVal = 0.9;
        
        const yScale = (idx) => padding.top + (idx / coefData.length) * (height - padding.top - padding.bottom);
        const xScale = (val) => center + (val / maxVal) * ((width - padding.left - padding.right) / 2);
        
        const barHeight = (height - padding.top - padding.bottom) / coefData.length - 8;

        const title = createSVGElement("text", {
            x: padding.left, y: padding.top - 8,
            fill: accentColor, "font-size": "9px", "font-family": "var(--font-mono)", "font-weight": "900"
        });
        title.textContent = `Coeficientes padronizados — ${modelLabel}`;
        svg.appendChild(title);
        
        for (let i = -0.8; i <= 0.8; i += 0.4) {
            if (Math.abs(i) < 0.01) continue;
            const x = xScale(i);
            svg.appendChild(createSVGElement("line", {
                x1: x, y1: padding.top, x2: x, y2: height - padding.bottom, class: "chart-grid-line"
            }));
            const text = createSVGElement("text", {
                x: x, y: height - padding.bottom + 14, fill: "var(--text-muted)",
                "font-size": "8px", "font-family": "var(--font-mono)", "text-anchor": "middle"
            });
            text.textContent = (i > 0 ? "+" : "") + i.toFixed(1);
            svg.appendChild(text);
        }

        svg.appendChild(createSVGElement("line", {
            x1: center, y1: padding.top, x2: center, y2: height - padding.bottom,
            stroke: "rgba(255,255,255,0.3)", "stroke-width": 2
        }));

        coefData.forEach((d, idx) => {
            const y = yScale(idx);
            const x = xScale(d.val);
            let xPos, w;
            if (d.val >= 0) { xPos = center; w = x - center; }
            else { xPos = x; w = center - x; }
            
            let color = d.val >= 0 ? accentColor : "var(--accent-orange)";
            if (d.warning) color = "var(--accent-rose)";
            
            const bar = createSVGElement("rect", {
                x: xPos, y, width: w, height: barHeight,
                fill: color, opacity: 0.75, class: "svg-horizontal-bar", rx: 3
            });
            
            const text = createSVGElement("text", {
                x: padding.left - 10, y: y + barHeight/2 + 4, fill: "var(--text-primary)",
                "font-size": "10px", "font-family": "var(--font-sans)", "font-weight": "600", "text-anchor": "end"
            });
            text.textContent = d.name;
            svg.appendChild(text);
            
            const valText = createSVGElement("text", {
                x: d.val >= 0 ? x + 6 : x - 32, y: y + barHeight/2 + 4, fill: "var(--text-secondary)",
                "font-size": "9px", "font-family": "var(--font-mono)"
            });
            valText.textContent = (d.val >= 0 ? "+" : "") + d.val.toFixed(2);
            svg.appendChild(valText);
            
            if (d.warning) {
                svg.appendChild(createSVGElement("circle", {
                    cx: Math.max(14, padding.left - 24), cy: y + barHeight/2, r: 5,
                    fill: "var(--accent-rose)", class: "glow-pulse"
                }));
            }
            
            bar.addEventListener("mousemove", (e) => {
                let warningText = d.warning ? `<br><span style="color:var(--accent-rose)"><b>ALERTA:</b> Inversão de sinal por colinearidade.</span>` : "";
                showTooltip(e, `<b>${modelLabel}</b> · ${d.name}<br>Coeficiente: ${d.val >= 0 ? "+" : ""}${d.val.toFixed(2)}${warningText}`);
            });
            bar.addEventListener("mouseleave", hideTooltip);
            bar.addEventListener("click", () => {
                const detailHTML = d.warning ? `
                    <div class="narrative-alert warning" style="margin: 0;">
                        <div class="alert-title">Inversão de Sinal (${modelLabel})</div>
                        PTS/GP negativo mesmo com MP e USG% controlados — artefato de colinearidade residual.
                    </div>` : `
                    <div class="narrative-alert purple" style="margin: 0;">
                        <div class="alert-title">Efeito Marginal (${modelLabel})</div>
                        <b>${d.name}</b> impacta o log-salário em <b>${d.val >= 0 ? "+" : ""}${d.val.toFixed(2)}</b> (variáveis padronizadas).
                    </div>`;
                openDrawer(`${modelLabel}: ${d.name}`, detailHTML);
            });
            
            svg.appendChild(bar);
        });

        svg.appendChild(createSVGElement("line", {
            x1: padding.left, y1: height - padding.bottom,
            x2: width - padding.right, y2: height - padding.bottom, class: "chart-axis-line"
        }));
    }

    // --- PANEL 8: PERMUTATION IMPORTANCE ---
    function drawPermutationImportance() {
        const svg = document.getElementById("permutationSvg");
        svg.innerHTML = "";
        
        const width = svg.clientWidth || 500;
        const height = svg.clientHeight || 300;
        const padding = getChartPadding(width, height, "wide-label");
        
        const yScale = (idx) => padding.top + (idx / permutationData.length) * (height - padding.top - padding.bottom);
        const xScale = (val) => padding.left + (val / 0.5) * (width - padding.left - padding.right);
        
        const barHeight = (height - padding.top - padding.bottom) / permutationData.length - 8;
        
        // Grid lines
        for (let i = 0; i <= 0.5; i += 0.1) {
            const x = xScale(i);
            const grid = createSVGElement("line", {
                x1: x, y1: padding.top, x2: x, y2: height - padding.bottom,
                class: "chart-grid-line"
            });
            svg.appendChild(grid);
            
            const text = createSVGElement("text", {
                x: x, y: height - padding.bottom + 14, fill: "var(--text-muted)",
                "font-size": "8px", "font-family": "var(--font-mono)", "text-anchor": "middle"
            });
            text.textContent = (i * 100).toFixed(0) + "%";
            svg.appendChild(text);
        }

        permutationData.forEach((d, idx) => {
            const y = yScale(idx);
            const x = xScale(d.val);
            const w = x - padding.left;
            
            const isTop = idx < 2;
            const fillStyle = isTop ? "url(#importanceGrad)" : "rgba(255,255,255,0.13)";
            
            if (isTop && !document.getElementById("importanceGrad")) {
                const defs = createSVGElement("defs", {});
                const grad = createSVGElement("linearGradient", {
                    id: "importanceGrad", x1: "0%", y1: "0%", x2: "100%", y2: "0%"
                });
                grad.appendChild(createSVGElement("stop", { offset: "0%", "stop-color": "#a78bfa" }));
                grad.appendChild(createSVGElement("stop", { offset: "100%", "stop-color": "#2dd4bf" }));
                defs.appendChild(grad);
                svg.appendChild(defs);
            }
            
            const bar = createSVGElement("rect", {
                x: padding.left, y: y, width: w, height: barHeight,
                fill: fillStyle, rx: 4, class: "svg-horizontal-bar"
            });
            
            if (isTop) {
                bar.setAttribute("filter", "drop-shadow(0px 0px 12px rgba(167, 139, 250, 0.34))");
            }
            
            const text = createSVGElement("text", {
                x: padding.left - 10, y: y + barHeight/2 + 4, fill: "var(--text-primary)",
                "font-size": "10px", "font-family": "var(--font-sans)", "font-weight": "600", "text-anchor": "end"
            });
            text.textContent = d.name;
            svg.appendChild(text);
            
            const valText = createSVGElement("text", {
                x: x + 8, y: y + barHeight/2 + 4, fill: isTop ? "var(--text-primary)" : "var(--text-secondary)",
                "font-size": "10px", "font-family": "var(--font-mono)", "font-weight": isTop ? "700" : "400"
            });
            valText.textContent = (d.val * 100).toFixed(1) + "%";
            svg.appendChild(valText);
            
            bar.addEventListener("mousemove", (e) => {
                showTooltip(e, `<b>Feature:</b> ${d.name}<br><b>Queda no R²:</b> ${(d.val * 100).toFixed(2)}%`);
            });
            bar.addEventListener("mouseleave", hideTooltip);
            bar.addEventListener("click", () => {
                openDrawer(d.name, `
                    <div class="drawer-stat-grid">
                        <div class="drawer-stat-card"><div class="val">${(d.val * 100).toFixed(1)}%</div><div class="lbl">Queda no R²</div></div>
                        <div class="drawer-stat-card"><div class="val">${isTop ? "Crítica" : "Secundária"}</div><div class="lbl">Prioridade</div></div>
                    </div>
                    <div class="narrative-alert ${isTop ? "purple" : "warning"}" style="margin-top: 1rem;">
                        <div class="alert-title">Leitura de importância</div>
                        ${isTop ? "Quando essa variável é embaralhada, a explicação do modelo colapsa. Ela domina a leitura econômica do salário." : "A variável ajuda a refinar a predição, mas não sustenta o modelo sozinha."}
                    </div>
                `);
            });
            
            svg.appendChild(bar);
        });

        const xAxis = createSVGElement("line", {
            x1: padding.left, y1: height - padding.bottom,
            x2: width - padding.right, y2: height - padding.bottom,
            class: "chart-axis-line"
        });
        svg.appendChild(xAxis);
    }

    // --- PANEL 9: SHAP SHOWCASE & WATERFALLS ---
    const playerStatsMap = {
        curry: [
            { lbl: "Perfil", val: "Superstar" },
            { lbl: "Salário real", val: "$48.0M" },
            { lbl: "Previsto HGB", val: "$42.0M" }
        ],
        kaminsky: [
            { lbl: "Perfil", val: "Role player" },
            { lbl: "Salário real", val: "$2.5M" },
            { lbl: "Previsto HGB", val: "$2.0M" }
        ],
        hardy: [
            { lbl: "Perfil", val: "Rookie scale" },
            { lbl: "Salário real", val: "$1.0M" },
            { lbl: "Previsto HGB", val: "$1.5M" }
        ]
    };

    profileBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            profileBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            
            const profileKey = btn.getAttribute("data-profile");
            loadSHAPProfile(profileKey);
        });
    });

    function loadSHAPProfile(key) {
        const profile = shapProfiles[key];
        if (!profile) return;
        
        // Update focused summary without portrait distractions
        const stats = playerStatsMap[key];
        if (shapFocusSummary) {
            shapFocusSummary.innerHTML = stats.map(s => `
            <div class="shap-summary-card">
                <div class="val">${s.val}</div>
                <div class="lbl">${s.lbl}</div>
            </div>
            `).join("");
        }

        // Draw SHAP Waterfall
        drawSHAPWaterfall(profile);
    }

    function drawSHAPWaterfall(profile) {
        const svg = document.getElementById("shapWaterfallSvg");
        svg.innerHTML = "";
        
        const width = svg.clientWidth || 500;
        const height = svg.clientHeight || 160;
        const padding = getChartPadding(width, height, "waterfall");
        padding.left = Math.max(padding.left, width < 760 ? 118 : 168);
        padding.right = Math.max(padding.right, 72);
        
        const cumulative = [];
        let running = profile.baseVal;
        profile.forces.forEach(force => {
            const start = running;
            running += force.val;
            cumulative.push({ ...force, start, end: running });
        });

        const values = [profile.baseVal, profile.predVal, ...cumulative.flatMap(item => [item.start, item.end])];
        const span = Math.max(...values) - Math.min(...values);
        const minVal = Math.max(0, Math.min(...values) - Math.max(1, span * 0.12));
        const maxVal = Math.max(...values) + Math.max(1.5, span * 0.18);
        const xScale = (val) => padding.left + ((val - minVal) / (maxVal - minVal)) * (width - padding.left - padding.right);

        const usableH = height - padding.top - padding.bottom;
        const rowSpacing = usableH / (profile.forces.length + 1);
        const blockHeight = Math.max(18, Math.min(30, rowSpacing * 0.46));
        const baseY = padding.top + 6;
        const finalY = padding.top + rowSpacing * (profile.forces.length + 0.72);

        const axis = createSVGElement("line", {
            x1: padding.left, y1: finalY,
            x2: width - padding.right, y2: finalY,
            class: "chart-axis-line"
        });
        svg.appendChild(axis);

        const ticks = 5;
        for (let i = 0; i <= ticks; i++) {
            const val = minVal + ((maxVal - minVal) * i / ticks);
            const x = xScale(val);
            svg.appendChild(createSVGElement("line", {
                x1: x, y1: padding.top, x2: x, y2: finalY,
                class: "chart-grid-line"
            }));
            const tickText = createSVGElement("text", {
                x, y: finalY + 18, fill: "var(--text-muted)",
                "font-size": "9px", "font-family": "var(--font-mono)", "text-anchor": "middle"
            });
            tickText.textContent = `$${val.toFixed(0)}M`;
            svg.appendChild(tickText);
        }

        const baseX = xScale(profile.baseVal);
        const predX = xScale(profile.predVal);
        svg.appendChild(createSVGElement("line", {
            x1: baseX, y1: padding.top,
            x2: baseX, y2: finalY,
            stroke: "rgba(255,255,255,0.34)", "stroke-width": 1.2, "stroke-dasharray": "4 4"
        }));
        svg.appendChild(createSVGElement("line", {
            x1: predX, y1: padding.top,
            x2: predX, y2: finalY,
            stroke: "var(--accent-purple)", "stroke-width": 2
        }));

        const basePill = createSVGElement("rect", {
            x: baseX - 46, y: baseY - 16, width: 92, height: 28,
            fill: "rgba(255,255,255,0.1)", stroke: "rgba(255,255,255,0.24)", rx: 999
        });
        svg.appendChild(basePill);
        const baseText = createSVGElement("text", {
            x: baseX, y: baseY + 3, fill: "var(--text-secondary)",
            "font-size": "10px", "font-family": "var(--font-mono)", "font-weight": "700", "text-anchor": "middle"
        });
        baseText.textContent = `Base: $${profile.baseVal}M`;
        svg.appendChild(baseText);
        
        cumulative.forEach((f, idx) => {
            const y = padding.top + rowSpacing * (idx + 1);
            const xStart = xScale(f.start);
            const xEnd = xScale(f.end);
            
            const x1 = Math.min(xStart, xEnd);
            const x2 = Math.max(xStart, xEnd);
            const w = Math.max(x2 - x1, 2); // Ensure visible width
            
            const color = f.positive ? "var(--accent-teal)" : "var(--accent-rose)";
            const rowLabel = createSVGElement("text", {
                x: padding.left - 16, y: y + 4,
                fill: "var(--text-primary)",
                "font-size": width < 760 ? "10px" : "12px",
                "font-family": "var(--font-sans)",
                "font-weight": "800",
                "text-anchor": "end"
            });
            rowLabel.textContent = f.name;
            svg.appendChild(rowLabel);

            svg.appendChild(createSVGElement("line", {
                x1: xStart, y1: y - rowSpacing * 0.36,
                x2: xStart, y2: y + rowSpacing * 0.36,
                stroke: "rgba(255,255,255,0.16)", "stroke-width": 1, "stroke-dasharray": "3 3"
            }));

            const block = createSVGElement("rect", {
                x: x1, y: y - blockHeight / 2, width: w, height: blockHeight,
                fill: color, opacity: 0.86, rx: 6,
                class: "svg-horizontal-bar"
            });
            
            const label = createSVGElement("text", {
                x: f.positive ? x2 + 8 : x1 - 8,
                y: y + 4,
                fill: color,
                "font-size": width < 760 ? "10px" : "12px",
                "font-family": "var(--font-mono)",
                "font-weight": "800",
                "text-anchor": f.positive ? "start" : "end"
            });
            label.textContent = `${f.val > 0 ? "+" : ""}${f.val.toFixed(1)}M`;
            
            block.addEventListener("mousemove", (e) => {
                showTooltip(e, `<b>Feature:</b> ${f.name}<br><b>Contribuição SHAP:</b> ${f.val > 0 ? "+" : ""}${f.val.toFixed(1)}M`);
            });
            block.addEventListener("mouseleave", hideTooltip);
            block.addEventListener("click", () => {
                openDrawer(`SHAP: ${f.name}`, `
                    <div class="drawer-stat-grid">
                        <div class="drawer-stat-card"><div class="val">${f.val > 0 ? "+" : ""}${f.val.toFixed(1)}M</div><div class="lbl">Contribuição</div></div>
                        <div class="drawer-stat-card"><div class="val">${f.positive ? "Aumenta" : "Reduz"}</div><div class="lbl">Direção</div></div>
                    </div>
                    <div class="narrative-alert ${f.positive ? "purple" : "warning"}" style="margin-top: 1rem;">
                        <div class="alert-title">Força local</div>
                        Esta variável ${f.positive ? "empurra a previsão para cima" : "puxa a previsão para baixo"} em relação ao valor base do modelo.
                    </div>
                `);
            });
            
            svg.appendChild(block);
            svg.appendChild(label);
        });

        const predPillWidth = 116;
        svg.appendChild(createSVGElement("rect", {
            x: Math.min(width - padding.right - predPillWidth, Math.max(padding.left, predX - predPillWidth / 2)),
            y: finalY - 44,
            width: predPillWidth,
            height: 32,
            fill: "var(--accent-purple)",
            rx: 999,
            filter: "drop-shadow(0px 0px 16px rgba(167,139,250,0.32))"
        }));
        const predLabelX = Math.min(width - padding.right - predPillWidth, Math.max(padding.left, predX - predPillWidth / 2)) + predPillWidth / 2;
        const predLabel = createSVGElement("text", {
            x: predLabelX, y: finalY - 23,
            fill: "#050505",
            "font-size": "11px",
            "font-family": "var(--font-mono)",
            "font-weight": "900",
            "text-anchor": "middle"
        });
        predLabel.textContent = `Previsto: $${profile.predVal.toFixed(1)}M`;
        svg.appendChild(predLabel);
    }

    // --- PANEL 10: EXTREME ERRORS ---
    function drawExtremeErrors() {
        const svg = document.getElementById("errorsSvg");
        svg.innerHTML = "";
        
        const width = svg.clientWidth || 500;
        const height = svg.clientHeight || 300;
        const padding = getChartPadding(width, height, "wide-label");
        
        const yScale = (idx) => padding.top + (idx / extremeErrors.length) * (height - padding.top - padding.bottom);
        const xScale = (val) => padding.left + (val / 50.0) * (width - padding.left - padding.right);
        
        const barHeight = (height - padding.top - padding.bottom) / extremeErrors.length - 12;
        
        // Grid lines X
        for (let i = 0; i <= 50; i += 10) {
            const x = xScale(i);
            const grid = createSVGElement("line", {
                x1: x, y1: padding.top, x2: x, y2: height - padding.bottom,
                class: "chart-grid-line"
            });
            svg.appendChild(grid);
            
            const text = createSVGElement("text", {
                x: x, y: height - padding.bottom + 14, fill: "var(--text-muted)",
                "font-size": "8px", "font-family": "var(--font-mono)", "text-anchor": "middle"
            });
            text.textContent = `$${i}M`;
            svg.appendChild(text);
        }

        extremeErrors.forEach((d, idx) => {
            const y = yScale(idx);
            
            // Real Salary (Faded background)
            const realW = xScale(d.real) - padding.left;
            const realBar = createSVGElement("rect", {
                x: padding.left, y: y, width: realW, height: barHeight,
                fill: "rgba(255,255,255,0.05)", rx: 2
            });
            svg.appendChild(realBar);
            
            // Predicted Salary (HGB Blue)
            const predW = xScale(d.pred) - padding.left;
            const predBar = createSVGElement("rect", {
                x: padding.left, y: y, width: predW, height: barHeight,
                fill: "var(--accent-blue)", opacity: 0.7, rx: 2
            });
            svg.appendChild(predBar);
            
            // Gap Error (Red overlay)
            const errorX = padding.left + predW;
            const errorW = realW - predW;
            if (errorW > 0) {
                const errorBar = createSVGElement("rect", {
                    x: errorX, y: y, width: errorW, height: barHeight,
                    fill: "var(--accent-rose)", opacity: 0.45, rx: 2
                });
                svg.appendChild(errorBar);
                
                const errTick = createSVGElement("line", {
                    x1: errorX + errorW, y1: y, x2: errorX + errorW, y2: y + barHeight,
                    stroke: "var(--accent-rose)", "stroke-width": 1.5
                });
                svg.appendChild(errTick);
            }
            
            const text = createSVGElement("text", {
                x: padding.left - 10, y: y + barHeight/2 + 4, fill: "var(--text-primary)",
                "font-size": "10px", "font-family": "var(--font-sans)", "font-weight": "600", "text-anchor": "end"
            });
            text.textContent = d.name;
            svg.appendChild(text);
            
            const valText = createSVGElement("text", {
                x: xScale(d.real) + 8, y: y + barHeight/2 + 4, fill: "var(--accent-rose)",
                "font-size": "9.5px", "font-family": "var(--font-mono)", "font-weight": "700"
            });
            valText.textContent = `+$${d.error.toFixed(1)}M`;
            svg.appendChild(valText);
            
            // Hover Target
            const hoverTarget = createSVGElement("rect", {
                x: padding.left, y: y, width: realW, height: barHeight,
                fill: "transparent", cursor: "pointer"
            });
            
            hoverTarget.addEventListener("mousemove", (e) => {
                showTooltip(e, `<b>${d.name}</b><br>Salário Real: $${d.real}M<br>Previsto: $${d.pred}M<br>Subestimação: $${d.error}M`);
            });
            hoverTarget.addEventListener("mouseleave", hideTooltip);
            
            hoverTarget.addEventListener("click", () => {
                const detailHTML = `
                    <div class="player-mini-card" style="border-color: rgba(244, 63, 94, 0.35);">
                        <div class="mini-avatar" style="background: var(--accent-rose);">${d.name[0]}</div>
                        <div class="mini-info">
                            <div class="mini-name">${d.name}</div>
                            <div class="mini-pos">${d.pos} | ${d.age} anos | Jogos: ${d.gp}</div>
                        </div>
                    </div>
                    <div class="drawer-stat-grid" style="margin-top: 1rem;">
                        <div class="drawer-stat-card"><div class="val">$${d.real}M</div><div class="lbl">Salário Real</div></div>
                        <div class="drawer-stat-card"><div class="val">$${d.pred}M</div><div class="lbl">Previsto</div></div>
                    </div>
                    <div class="narrative-alert warning" style="margin-top: 1rem;">
                        <div class="alert-title">Causa Raiz do Erro</div>
                        ${d.cause}
                        <br><br>
                        Como as estatísticas em quadra de jogadores lesionados caem vertiginosamente, o modelo prediz salários de atletas de final de banco. 
                        A discrepância se deve à rigidez contratual (contratos longos garantidos no passado).
                    </div>
                `;
                openDrawer(`Diagnóstico de Erro: ${d.name}`, detailHTML);
            });
            
            svg.appendChild(hoverTarget);
        });

        const xAxis = createSVGElement("line", {
            x1: padding.left, y1: height - padding.bottom,
            x2: width - padding.right, y2: height - padding.bottom,
            class: "chart-axis-line"
        });
        svg.appendChild(xAxis);
    }

    // --- PANEL 11: MAE BY PLAYER PROFILE (USD) ---
    function drawFitMap() {
        const svg = document.getElementById("fitMapSvg");
        svg.innerHTML = "";

        const data = window.fitMapData || [];
        const overallMae = window.HGB_OVERALL_MAE_M || 3.68;
        const maxMae = Math.max(16, ...data.map(d => d.maeM), overallMae + 1);

        const width = svg.clientWidth || 500;
        const height = svg.clientHeight || 300;
        const padding = getChartPadding(width, height, "wide-label");
        padding.top = Math.max(padding.top, 40);
        padding.left = Math.max(padding.left, width < 760 ? 150 : 190);
        padding.right = Math.max(padding.right, 110);
        padding.bottom = Math.max(padding.bottom, 38);

        const plotW = width - padding.left - padding.right;
        const xScale = (maeM) => padding.left + (maeM / maxMae) * plotW;
        const rowH = (height - padding.top - padding.bottom) / data.length;
        const barH = Math.max(14, Math.min(22, rowH * 0.38));

        const refX = xScale(overallMae);
        svg.appendChild(createSVGElement("line", {
            x1: refX, y1: padding.top - 6, x2: refX, y2: height - padding.bottom,
            stroke: "rgba(255,255,255,0.5)", "stroke-width": 1.5, "stroke-dasharray": "5 4"
        }));
        const refLabel = createSVGElement("text", {
            x: refX, y: padding.top - 12,
            fill: "var(--text-muted)", "font-size": "8px", "font-family": "var(--font-mono)",
            "text-anchor": "middle", "font-weight": "700"
        });
        refLabel.textContent = `MAE global US$ ${overallMae.toFixed(2)}M`;
        svg.appendChild(refLabel);

        [0, 4, 8, 12, 16].filter(t => t <= maxMae).forEach(tick => {
            const x = xScale(tick);
            svg.appendChild(createSVGElement("line", {
                x1: x, y1: padding.top, x2: x, y2: height - padding.bottom, class: "chart-grid-line"
            }));
            const label = createSVGElement("text", {
                x, y: height - padding.bottom + 16,
                fill: "var(--text-muted)", "font-size": "9px", "font-family": "var(--font-mono)", "text-anchor": "middle"
            });
            label.textContent = tick === 0 ? "US$ 0" : `US$ ${tick}M`;
            svg.appendChild(label);
        });

        const axisTitle = createSVGElement("text", {
            x: padding.left + plotW / 2, y: height - 4,
            fill: "var(--text-muted)", "font-size": "9px", "font-family": "var(--font-mono)", "text-anchor": "middle"
        });
        axisTitle.textContent = "Desvio absoluto médio (MAE) em milhões de dólares";
        svg.appendChild(axisTitle);

        data.forEach((d, idx) => {
            const centerY = padding.top + idx * rowH + rowH / 2;
            const barW = xScale(d.maeM) - padding.left;

            const groupLabel = createSVGElement("text", {
                x: padding.left - 14, y: centerY - 6,
                fill: "var(--text-primary)", "font-size": width < 760 ? "11px" : "12px",
                "font-family": "var(--font-sans)", "font-weight": "900", "text-anchor": "end"
            });
            groupLabel.textContent = d.group;
            svg.appendChild(groupLabel);

            const descLabel = createSVGElement("text", {
                x: padding.left - 14, y: centerY + 9,
                fill: "var(--text-muted)", "font-size": "8px", "font-family": "var(--font-sans)", "text-anchor": "end"
            });
            descLabel.textContent = `${d.desc} · n=${d.n}`;
            svg.appendChild(descLabel);

            const bar = createSVGElement("rect", {
                x: padding.left, y: centerY - barH / 2, width: Math.max(2, barW), height: barH,
                fill: d.color, opacity: 0.88, rx: 4, class: "svg-horizontal-bar"
            });

            const maeText = createSVGElement("text", {
                x: padding.left + barW + 8, y: centerY + 4,
                fill: d.color, "font-size": "11px", "font-family": "var(--font-mono)", "font-weight": "900"
            });
            maeText.textContent = `US$ ${d.maeM.toFixed(2)}M`;
            svg.appendChild(maeText);

            const ratio = d.maeM / overallMae;
            const ratioColor = ratio <= 1 ? "var(--accent-teal)" : ratio <= 2 ? "var(--accent-orange)" : "var(--accent-rose)";
            const ratioText = createSVGElement("text", {
                x: padding.left + barW + 8, y: centerY + 17,
                fill: ratioColor, "font-size": "8px", "font-family": "var(--font-mono)", "font-weight": "700"
            });
            ratioText.textContent = `${ratio.toFixed(1)}× o MAE global`;
            svg.appendChild(ratioText);

            bar.addEventListener("mousemove", (e) => {
                showTooltip(e, `<b>${d.group}</b> (n=${d.n})<br>${d.desc}<br><br><b>MAE médio:</b> US$ ${d.maeM.toFixed(2)}M<br><b>vs global:</b> ${ratio.toFixed(1)}×<br><br>${d.example}`);
            });
            bar.addEventListener("mouseleave", hideTooltip);
            bar.addEventListener("click", () => {
                openDrawer(d.group, `
                    <div class="drawer-stat-grid">
                        <div class="drawer-stat-card"><div class="val">US$ ${d.maeM.toFixed(2)}M</div><div class="lbl">MAE médio</div></div>
                        <div class="drawer-stat-card"><div class="val">${d.n}</div><div class="lbl">Jogadores (teste)</div></div>
                        <div class="drawer-stat-card"><div class="val">${ratio.toFixed(1)}×</div><div class="lbl">vs MAE global</div></div>
                    </div>
                    <div class="narrative-alert ${ratio > 2 ? "warning" : "purple"}" style="margin-top: 1rem;">
                        <div class="alert-title">${d.desc}</div>
                        ${d.example}
                    </div>
                `);
            });
            svg.appendChild(bar);
        });
    }

    // --- PANEL 12: DATASET LIMITATIONS ---
    const limitationsData = [
        { name: "Mercado", impact: 0.92, detail: "All-Star, draft, tamanho de mercado e popularidade explicam parte relevante do salário." },
        { name: "Lesões", impact: 0.82, detail: "Histórico médico explica contratos tóxicos que o box-score da temporada não capta." },
        { name: "Contrato", impact: 0.78, detail: "Tipo de contrato, anos restantes e data de assinatura mudam completamente o valor pago." },
        { name: "Playoffs", impact: 0.54, detail: "Performance em jogos decisivos influencia reputação e negociação salarial." },
        { name: "Tempo", impact: 0.48, detail: "Uma temporada isolada não captura evolução de carreira ou declínio de longo prazo." }
    ];

    function drawLimitations() {
        const svg = document.getElementById("limitationsSvg");
        svg.innerHTML = "";

        const width = svg.clientWidth || 500;
        const height = svg.clientHeight || 300;
        const padding = getChartPadding(width, height, "wide-label");
        padding.left = Math.max(padding.left, width < 760 ? 116 : 150);
        padding.right = Math.max(padding.right, 90);

        const yScale = (idx) => padding.top + (idx / limitationsData.length) * (height - padding.top - padding.bottom);
        const xScale = (value) => padding.left + value * (width - padding.left - padding.right);
        const barHeight = (height - padding.top - padding.bottom) / limitationsData.length - 12;

        [0, 0.25, 0.5, 0.75, 1].forEach(tick => {
            const x = xScale(tick);
            svg.appendChild(createSVGElement("line", {
                x1: x, y1: padding.top, x2: x, y2: height - padding.bottom,
                class: "chart-grid-line"
            }));

            const label = createSVGElement("text", {
                x, y: height - padding.bottom + 16,
                fill: "var(--text-muted)",
                "font-size": "9px",
                "font-family": "var(--font-mono)",
                "text-anchor": "middle"
            });
            label.textContent = `${Math.round(tick * 100)}%`;
            svg.appendChild(label);
        });

        limitationsData.forEach((d, idx) => {
            const y = yScale(idx);
            const w = xScale(d.impact) - padding.left;
            const fill = idx < 3 ? "url(#limitationsGrad)" : "rgba(255,255,255,0.18)";

            if (idx === 0) {
                const defs = createSVGElement("defs", {});
                const grad = createSVGElement("linearGradient", {
                    id: "limitationsGrad", x1: "0%", y1: "0%", x2: "100%", y2: "0%"
                });
                grad.appendChild(createSVGElement("stop", { offset: "0%", "stop-color": "#fb7185" }));
                grad.appendChild(createSVGElement("stop", { offset: "100%", "stop-color": "#fb923c" }));
                defs.appendChild(grad);
                svg.appendChild(defs);
            }

            const name = createSVGElement("text", {
                x: padding.left - 14,
                y: y + barHeight / 2 + 4,
                fill: "var(--text-primary)",
                "font-size": "12px",
                "font-family": "var(--font-sans)",
                "font-weight": "900",
                "text-anchor": "end"
            });
            name.textContent = d.name;
            svg.appendChild(name);

            const bar = createSVGElement("rect", {
                x: padding.left,
                y,
                width: w,
                height: barHeight,
                fill,
                rx: 8,
                class: "svg-horizontal-bar"
            });

            bar.addEventListener("mousemove", (e) => {
                showTooltip(e, `<b>${d.name}</b><br>Impacto no teto preditivo: ${(d.impact * 100).toFixed(0)}%<br>${d.detail}`);
            });
            bar.addEventListener("mouseleave", hideTooltip);
            bar.addEventListener("click", () => {
                openDrawer(`Variável ausente: ${d.name}`, `
                    <div class="drawer-stat-grid">
                        <div class="drawer-stat-card"><div class="val">${(d.impact * 100).toFixed(0)}%</div><div class="lbl">Impacto Qualitativo</div></div>
                        <div class="drawer-stat-card"><div class="val">Ausente</div><div class="lbl">No Dataset</div></div>
                    </div>
                    <div class="narrative-alert warning" style="margin-top: 1rem;">
                        <div class="alert-title">Por que limita o modelo</div>
                        ${d.detail}
                    </div>
                `);
            });
            svg.appendChild(bar);

            const value = createSVGElement("text", {
                x: Math.min(width - padding.right + 10, padding.left + w + 10),
                y: y + barHeight / 2 + 4,
                fill: idx < 3 ? "var(--accent-orange)" : "var(--text-secondary)",
                "font-size": "11px",
                "font-family": "var(--font-mono)",
                "font-weight": "900"
            });
            value.textContent = `${Math.round(d.impact * 100)}%`;
            svg.appendChild(value);
        });

        svg.appendChild(createSVGElement("line", {
            x1: padding.left, y1: height - padding.bottom,
            x2: width - padding.right, y2: height - padding.bottom,
            class: "chart-axis-line"
        }));
    }
    
    // Trigger initial load animations
    triggerPanelAnimation(0);
});
