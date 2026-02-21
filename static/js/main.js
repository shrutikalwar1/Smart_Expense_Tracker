// ==========================================
// THEME TOGGLE
// ==========================================
const toggleBtn = document.querySelector('.theme-toggle');
const toggleIcon = document.querySelector('.theme-toggle__icon');
const toggleText = document.querySelector('.theme-toggle__text');

function getSavedTheme() {
    return localStorage.getItem('theme') || 'light';
}

function emitThemeChanged(theme) {
    window.dispatchEvent(new CustomEvent('theme-changed', { detail: { theme } }));
}

function updateThemeToggle(theme) {
    if (!toggleBtn) return;
    if (toggleIcon) toggleIcon.textContent = theme === 'dark' ? '☀️' : '🌙';
    if (toggleText) toggleText.textContent = theme === 'dark' ? 'Light mode' : 'Dark mode';
    toggleBtn.setAttribute('aria-label', theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode');
}

function applyTheme(theme, announce = true) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
    updateThemeToggle(theme);
    if (announce) emitThemeChanged(theme);
}

const currentTheme = getSavedTheme();
applyTheme(currentTheme);

if (toggleBtn) {
    toggleBtn.addEventListener('click', () => {
        const theme = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
        applyTheme(theme);
    });
}

// Re-emit after DOM is ready so late listeners also sync with persisted theme.
document.addEventListener('DOMContentLoaded', () => emitThemeChanged(getSavedTheme()));

// ==========================================
// NAVIGATION ACTIVE STATE
// ==========================================
document.addEventListener('DOMContentLoaded', () => {
    const currentPath = window.location.pathname;
    
    document.querySelectorAll('.nav-btn').forEach(btn => {
        if ((currentPath === '/' && btn.href.includes('index')) ||
            (currentPath.includes('expenses') && btn.href.includes('expenses')) ||
            (currentPath.includes('analytics') && btn.href.includes('analytics'))) {
            btn.classList.add('active');
        } else if (currentPath === '/' && !btn.href.includes('expenses') && !btn.href.includes('analytics')) {
            btn.classList.add('active');
        }
    });
});

// ==========================================
// AI CATEGORY SUGGESTION
// ==========================================
const descInput = document.getElementById('description');
const categorySelect = document.getElementById('category');
const suggestText = document.getElementById('suggestText');

if (descInput && categorySelect && suggestText) {
    let typingTimer;
    
    descInput.addEventListener('input', () => {
        clearTimeout(typingTimer);
        const text = descInput.value.trim();
        
        if (!text) {
            suggestText.textContent = '';
            return;
        }
        
        typingTimer = setTimeout(async () => {
            try {
                const res = await fetch('/api/suggest-category', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ description: text })
                });
                
                if (res.ok) {
                    const data = await res.json();
                    categorySelect.value = data.category;
                    suggestText.textContent = `🤖 AI suggests: ${data.category.toUpperCase()}`;
                    suggestText.style.color = '#10b981';
                }
            } catch (e) {
                console.error('AI suggestion error:', e);
            }
        }, 400);
    });
}

// ==========================================
// DASHBOARD STATS LOADING
// ==========================================
async function loadDashboardStats() {
    try {
        const res = await fetch('/api/categories');
        const categories = await res.json();
        const total = Object.values(categories).reduce((a, b) => a + b, 0);
        
        const totalEl = document.getElementById('totalExpenses');
        if (totalEl) {
            totalEl.textContent = '₹' + total.toLocaleString('en-IN');
        }
    } catch(e) {
        console.error('Stats loading error:', e);
    }
}

// Load stats on page load
if (window.location.pathname === '/') {
    loadDashboardStats();
}

// ==========================================
// FORM VALIDATION & FEEDBACK
// ==========================================
const expenseForm = document.querySelector('.expense-form');
if (expenseForm) {
    expenseForm.addEventListener('submit', async (e) => {
        const amount = expenseForm.querySelector('input[name="amount"]').value;
        const desc = expenseForm.querySelector('input[name="description"]').value;
        
        if (!amount || parseFloat(amount) <= 0) {
            e.preventDefault();
            alert('Please enter a valid amount');
            return;
        }
        
        if (!desc.trim()) {
            e.preventDefault();
            alert('Please enter a description');
            return;
        }
    });
}

