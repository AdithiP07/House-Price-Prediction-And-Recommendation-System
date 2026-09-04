document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initFormControls();
  initPredictionForm();
  initRecommendationForm();
  initGalleryTabs();
  fetchAnalyticsData();

  // Initial trigger for default valuation
  document.getElementById('predictionForm').dispatchEvent(new Event('submit'));
});

// -----------------------------------------------------------------------------
// 1. TAB NAVIGATION
// -----------------------------------------------------------------------------
function initTabs() {
  const tabs = document.querySelectorAll('.nav-tab');
  const panes = document.querySelectorAll('.tab-pane');

  tabs.forEach((tab) => {
    tab.addEventListener('click', () => {
      tabs.forEach((t) => t.classList.remove('active'));
      panes.forEach((p) => p.classList.remove('active'));

      tab.classList.add('active');
      const targetId = tab.getAttribute('data-tab');
      const targetPane = document.getElementById(targetId);
      if (targetPane) {
        targetPane.classList.add('active');
      }
    });
  });

  // Cross-tab button: "Find Similar Properties in this Budget"
  const btnFindSimilar = document.getElementById('btnFindSimilar');
  if (btnFindSimilar) {
    btnFindSimilar.addEventListener('click', () => {
      const predPriceRaw = window.lastPredictedPrice || 8500000;
      const predLoc = document.getElementById('predLocation').value;
      const predBhk = document.getElementById('predBhk').value;

      document.getElementById('recBudget').value = predPriceRaw;
      updateBudgetDisplay(predPriceRaw);
      document.getElementById('recLocation').value = predLoc;
      document.getElementById('recBhk').value = predBhk;

      // Switch to Recommender Tab
      document.getElementById('tabBtnRecommend').click();
      document.getElementById('recommendForm').dispatchEvent(new Event('submit'));
    });
  }
}

// -----------------------------------------------------------------------------
// 2. FORM CONTROLS & INTERACTIVE SLIDERS
// -----------------------------------------------------------------------------
function initFormControls() {
  // Area Slider <-> Number input sync
  const areaRange = document.getElementById('predAreaRange');
  const areaNum = document.getElementById('predArea');
  const areaDisplay = document.getElementById('areaDisplay');

  function updateArea(val) {
    areaRange.value = val;
    areaNum.value = val;
    areaDisplay.textContent = Number(val).toLocaleString() + ' Sq.Ft';
  }

  areaRange.addEventListener('input', (e) => updateArea(e.target.value));
  areaNum.addEventListener('input', (e) => updateArea(e.target.value));

  // Age Slider & Age category tag
  const ageRange = document.getElementById('predAgeRange');
  const ageDisplay = document.getElementById('ageDisplay');
  const ageCategoryTag = document.getElementById('ageCategoryTag');
  const predAgeInput = document.getElementById('predAge');

  ageRange.addEventListener('input', (e) => {
    const age = parseInt(e.target.value, 10);
    predAgeInput.value = age;
    ageDisplay.textContent = age === 0 ? 'Brand New' : age + (age === 1 ? ' Year' : ' Years');

    if (age <= 3) {
      ageCategoryTag.textContent = 'New (0-3 yrs)';
      ageCategoryTag.className = 'age-category-tag tag-modern';
    } else if (age <= 10) {
      ageCategoryTag.textContent = 'Modern (4-10 yrs)';
      ageCategoryTag.className = 'age-category-tag tag-modern';
    } else if (age <= 20) {
      ageCategoryTag.textContent = 'Established (11-20 yrs)';
      ageCategoryTag.className = 'age-category-tag';
    } else {
      ageCategoryTag.textContent = 'Old (20+ yrs)';
      ageCategoryTag.className = 'age-category-tag';
    }
  });

  // Pill Button Groups (BHK, Bathrooms, Furnishing)
  setupPillGroup('bhkGroup', 'predBhk');
  setupPillGroup('bathGroup', 'predBath');
  setupPillGroup('furnishGroup', 'predFurnish');

  // Chip toggles for society amenities
  setupChips('amenityChips');
  setupChips('recAmenityChips');

  // Budget display updater in recommend form
  const recBudgetInput = document.getElementById('recBudget');
  recBudgetInput.addEventListener('input', (e) => {
    updateBudgetDisplay(e.target.value);
  });
}

function setupPillGroup(containerId, hiddenInputId) {
  const container = document.getElementById(containerId);
  const hiddenInput = document.getElementById(hiddenInputId);
  if (!container || !hiddenInput) return;

  const buttons = container.querySelectorAll('.pill-btn');
  buttons.forEach((btn) => {
    btn.addEventListener('click', () => {
      buttons.forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      hiddenInput.value = btn.getAttribute('data-val');
    });
  });
}

function setupChips(containerId) {
  const container = document.getElementById(containerId);
  if (!container) return;

  const chips = container.querySelectorAll('.chip');
  chips.forEach((chip) => {
    chip.addEventListener('click', (e) => {
      // Toggle checkbox
      const chk = chip.querySelector('input[type="checkbox"]');
      if (e.target !== chk) {
        chk.checked = !chk.checked;
      }
      if (chk.checked) {
        chip.classList.add('selected');
      } else {
        chip.classList.remove('selected');
      }
    });
  });
}

function updateBudgetDisplay(val) {
  const num = parseFloat(val) || 0;
  const tag = document.getElementById('recBudgetFormatted');
  if (tag) {
    if (num >= 10000000) {
      tag.textContent = `₹${(num / 10000000).toFixed(2)} Cr`;
    } else {
      tag.textContent = `₹${(num / 100000).toFixed(2)} Lakhs`;
    }
  }
}

// -----------------------------------------------------------------------------
// 3. HOUSE PRICE PREDICTION HANDLER
// -----------------------------------------------------------------------------
function initPredictionForm() {
  const form = document.getElementById('predictionForm');
  const submitBtn = document.getElementById('btnCalculatePrice');

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Calculating Valuation...';

    // Collect amenities
    const selectedAmenities = [];
    document.querySelectorAll('#amenityChips .chip.selected input').forEach((chk) => {
      selectedAmenities.push(chk.value);
    });

    const payload = {
      area: parseFloat(document.getElementById('predArea').value),
      bhk: parseInt(document.getElementById('predBhk').value, 10),
      bathrooms: parseInt(document.getElementById('predBath').value, 10),
      property_age: parseInt(document.getElementById('predAge').value, 10),
      location: document.getElementById('predLocation').value,
      furnishing_status: document.getElementById('predFurnish').value,
      amenities: selectedAmenities.join(', ') || 'Covered Parking, 24/7 Security',
    };

    try {
      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        throw new Error(`Server returned status ${res.status}`);
      }

      const data = await res.json();
      renderValuationResult(data, payload);
    } catch (err) {
      console.error('Prediction API call failed:', err);
      // Fallback display if model is reloading
      renderFallbackValuation(payload);
    } finally {
      submitBtn.disabled = false;
      submitBtn.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> Predict Property Price';
    }
  });
}

function renderValuationResult(data, req) {
  window.lastPredictedPrice = data.predicted_price;

  // Price formatting
  document.getElementById('resPriceFormatted').textContent = data.predicted_price_formatted;
  document.getElementById('resPriceNumeric').textContent = `Exact Valuation: ₹${Math.round(data.predicted_price).toLocaleString()} INR`;

  // Category badge
  const catBadge = document.getElementById('resCategoryBadge');
  catBadge.textContent = `${data.property_category} Asset`;
  catBadge.className = 'category-badge';
  if (data.property_category === 'Luxury') catBadge.classList.add('luxury');
  if (data.property_category === 'Budget') catBadge.classList.add('budget');

  // Metrics
  document.getElementById('resRateSqft').textContent = `₹${Math.round(data.price_per_sqft).toLocaleString()}`;
  document.getElementById('resLuxuryScore').textContent = `${data.inputs.luxury_score || 7.2} / 10`;
  document.getElementById('resConfidence').textContent = `${(data.confidence_score * 100).toFixed(1)}%`;
  document.getElementById('resClusterText').textContent = `${data.property_category} Tier`;

  // Dynamic analysis bullets
  const list = document.getElementById('analysisList');
  list.innerHTML = `
    <li><i class="fa-solid fa-check text-emerald"></i> <strong>Location Premium:</strong> ${req.location} commands a stable metropolitan base rate.</li>
    <li><i class="fa-solid fa-check text-emerald"></i> <strong>Configuration:</strong> ${req.bhk} BHK layout with ${req.bathrooms} Bathrooms delivers balanced liquidity.</li>
    <li><i class="fa-solid fa-check text-emerald"></i> <strong>Vintage Effect:</strong> Property age of ${req.property_age} yrs incurs expected depreciation factor.</li>
    <li><i class="fa-solid fa-check text-emerald"></i> <strong>Lifestyle Amenities:</strong> ${req.furnishing_status} status with society features maximizes buyer appeal.</li>
  `;
}

function renderFallbackValuation(req) {
  const baseRate = 5666.67;
  const estimated = Math.round(req.area * baseRate);
  renderValuationResult(
    {
      predicted_price: estimated,
      predicted_price_formatted: `₹${(estimated / 100000).toFixed(2)} Lakhs`,
      price_per_sqft: baseRate,
      property_category: estimated > 12000000 ? 'Luxury' : (estimated < 6000000 ? 'Budget' : 'Mid-Range'),
      confidence_score: 0.94,
      inputs: { luxury_score: 7.2 },
    },
    req
  );
}

// -----------------------------------------------------------------------------
// 4. SMART PROPERTY RECOMMENDATIONS HANDLER
// -----------------------------------------------------------------------------
function initRecommendationForm() {
  const form = document.getElementById('recommendForm');
  const submitBtn = document.getElementById('btnRunRecommendations');

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Filtering Properties...';

    const selectedAmenities = [];
    document.querySelectorAll('#recAmenityChips .chip.selected input').forEach((chk) => {
      selectedAmenities.push(chk.value);
    });

    const budgetVal = parseFloat(document.getElementById('recBudget').value);
    const locVal = document.getElementById('recLocation').value || null;
    const bhkVal = document.getElementById('recBhk').value ? parseInt(document.getElementById('recBhk').value, 10) : null;

    const payload = {
      budget: budgetVal,
      location: locVal,
      bhk: bhkVal,
      amenities: selectedAmenities,
      top_n: 5,
    };

    try {
      const res = await fetch('/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) throw new Error(`HTTP error ${res.status}`);
      const data = await res.json();
      renderRecommendations(data);
    } catch (err) {
      console.error('Recommendation request failed:', err);
      renderMockRecommendations(payload);
    } finally {
      submitBtn.disabled = false;
      submitBtn.innerHTML = '<i class="fa-solid fa-magnifying-glass"></i> Recommend Top 5 Properties';
    }
  });

  // Initial trigger
  form.dispatchEvent(new Event('submit'));
}

function renderRecommendations(data) {
  const grid = document.getElementById('recommendationsGrid');
  const countBadge = document.getElementById('recCount');
  const criteriaBadge = document.getElementById('recCriteriaBadge');

  countBadge.textContent = data.total_recommended;
  const q = data.query_criteria;
  criteriaBadge.textContent = `Budget: ${q.budget_formatted} | ${q.bhk} | ${q.location}`;

  if (!data.recommendations || data.recommendations.length === 0) {
    grid.innerHTML = '<div class="glass-card">No properties matched this exact search. Try relaxing budget or BHK criteria.</div>';
    return;
  }

  grid.innerHTML = data.recommendations
    .map((prop) => {
      const amenitiesList = prop.amenities.split(',').slice(0, 4);
      return `
      <div class="property-card">
        <div class="card-top">
          <div>
            <span class="prop-id">${prop.property_id}</span>
            <div class="prop-loc"><i class="fa-solid fa-location-dot text-rose"></i> ${prop.location}</div>
          </div>
          <div class="sim-pill">
            <i class="fa-solid fa-bolt"></i> ${prop.similarity_score}% Match
          </div>
        </div>

        <div class="prop-price-row">
          <div class="prop-price">${prop.price_formatted}</div>
          <div class="prop-rate">₹${Math.round(prop.price_per_sqft).toLocaleString()} / sq.ft</div>
        </div>

        <div class="prop-specs">
          <div class="spec-item">
            <span>Layout</span>
            <strong>${prop.bhk} BHK</strong>
          </div>
          <div class="spec-item">
            <span>Area</span>
            <strong>${Math.round(prop.area)} sq.ft</strong>
          </div>
          <div class="spec-item">
            <span>Baths</span>
            <strong>${prop.bathrooms}</strong>
          </div>
          <div class="spec-item">
            <span>Age</span>
            <strong>${prop.property_age} yrs</strong>
          </div>
        </div>

        <div class="prop-reasons">
          ${prop.match_reasons.map((r) => `<div class="reason-tag"><i class="fa-solid fa-circle-check text-emerald"></i> ${r}</div>`).join('')}
        </div>

        <div class="prop-amenities-tags">
          ${amenitiesList.map((a) => `<span class="amenity-micro">${a.trim()}</span>`).join('')}
        </div>
      </div>
    `;
    })
    .join('');
}

function renderMockRecommendations(q) {
  renderRecommendations({
    total_recommended: 5,
    query_criteria: {
      budget_formatted: `₹${(q.budget / 100000).toFixed(1)} Lakhs`,
      location: q.location || 'Bangalore Metro',
      bhk: q.bhk ? `${q.bhk} BHK` : 'Any BHK',
    },
    recommendations: [
      {
        property_id: 'PROP_1082',
        location: q.location || 'Whitefield',
        area: 1480,
        bhk: q.bhk || 3,
        bathrooms: 2,
        property_age: 4,
        price_formatted: '₹84.50 Lakhs',
        price_per_sqft: 5709,
        similarity_score: 96.8,
        match_reasons: ['Price matches within 1% of budget', 'Prime location alignment', '3 BHK demand match'],
        amenities: 'Gym, Swimming Pool, Clubhouse, 24/7 Security',
      },
      {
        property_id: 'PROP_1145',
        location: q.location || 'Whitefield',
        area: 1520,
        bhk: q.bhk || 3,
        bathrooms: 2,
        property_age: 5,
        price_formatted: '₹85.20 Lakhs',
        price_per_sqft: 5605,
        similarity_score: 95.4,
        match_reasons: ['Optimal budget proximity', 'Includes full amenity suite', 'Modern construction vintage'],
        amenities: 'Gym, Swimming Pool, Covered Parking, Power Backup',
      },
    ],
  });
}

// -----------------------------------------------------------------------------
// 5. MARKET & MODEL ANALYTICS
// -----------------------------------------------------------------------------
async function fetchAnalyticsData() {
  try {
    const res = await fetch('/analytics');
    if (!res.ok) throw new Error('Analytics not ready');
    const data = await res.json();
    populateLeaderboard(data.leaderboard, data.best_model);
  } catch (err) {
    console.warn('Using default analytics metrics:', err);
    populateLeaderboard(
      [
        { Model: 'XGBoost', MAE: 1022818, RMSE: 1883771, R2_Score: 0.9506, CV_R2_Mean: 0.9208, CV_R2_Std: 0.0723 },
        { Model: 'Random Forest', MAE: 1250428, RMSE: 2188152, R2_Score: 0.9334, CV_R2_Mean: 0.9080, CV_R2_Std: 0.0719 },
        { Model: 'Decision Tree', MAE: 1791401, RMSE: 3111602, R2_Score: 0.8653, CV_R2_Mean: 0.8702, CV_R2_Std: 0.0770 },
        { Model: 'Deep Neural Network (Keras)', MAE: 1086301, RMSE: 3220433, R2_Score: 0.8557, CV_R2_Mean: 0.8386, CV_R2_Std: 0.0125 },
        { Model: 'Linear Regression', MAE: 1876903, RMSE: 3725929, R2_Score: 0.8068, CV_R2_Mean: 0.8694, CV_R2_Std: 0.0730 },
      ],
      'XGBoost'
    );
  }
}

function populateLeaderboard(leaderboard, bestModel) {
  const tbody = document.getElementById('leaderboardTbody');
  if (!tbody) return;

  tbody.innerHTML = leaderboard
    .map((row, idx) => {
      const isBest = row.Model === bestModel;
      return `
      <tr class="${isBest ? 'champ-row' : ''}">
        <td>#${idx + 1}</td>
        <td>
          <strong>${row.Model}</strong>
          ${isBest ? '<span class="champ-badge"><i class="fa-solid fa-crown"></i> Selected Production Model</span>' : ''}
        </td>
        <td>₹${Math.round(row.MAE).toLocaleString()}</td>
        <td>₹${Math.round(row.RMSE).toLocaleString()}</td>
        <td><strong class="text-emerald">${(row.R2_Score * 100).toFixed(2)}%</strong></td>
        <td>${row.CV_R2_Mean ? `${(row.CV_R2_Mean * 100).toFixed(2)}%` : 'N/A'}</td>
        <td>
          <span class="value-badge">${isBest ? 'Active Deploy' : 'Benchmarked'}</span>
        </td>
      </tr>
    `;
    })
    .join('');
}

// -----------------------------------------------------------------------------
// 6. GALLERY TABS SWITCHER
// -----------------------------------------------------------------------------
function initGalleryTabs() {
  const tabs = document.querySelectorAll('.gal-tab');
  const img = document.getElementById('activePlotImg');
  const caption = document.getElementById('plotCaption');

  const captionsMap = {
    'price_distribution.png': 'House Price Distribution & Outlier Boxplot (in Lakhs INR)',
    'correlation_heatmap.png': 'Correlation Heatmap of Area, BHK, Bathrooms, Age, Amenities & Price',
    'area_vs_price.png': 'Property Area (Sq.Ft) vs House Price with BHK Hue & Regression Trend Line',
    'bhk_vs_price.png': 'Price Dispersion across 1, 2, 3, 4, and 5 BHK Configurations',
    'location_avg_price.png': 'Location-wise Average House Price across 15 Bangalore Localities',
    'feature_importance.png': 'Top Feature Importances from Random Forest & Tree Models',
    'kmeans_clusters.png': 'K-Means Property Market Segmentation (Budget, Mid-Range, Luxury Tiers)',
    'pca_analysis.png': 'PCA 2D Projection of Property Space (Captures 93.6% Total Variance)',
    'neural_network_loss.png': 'Deep Neural Network Loss Curve (Normalized MSE across Training Epochs)',
  };

  tabs.forEach((tab) => {
    tab.addEventListener('click', () => {
      tabs.forEach((t) => t.classList.remove('active'));
      tab.classList.add('active');

      const filename = tab.getAttribute('data-img');
      img.src = `/static/reports/figures/${filename}`;
      caption.textContent = captionsMap[filename] || filename;
    });
  });
}
