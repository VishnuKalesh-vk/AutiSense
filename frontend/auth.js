'use strict';

// ── Utilities ─────────────────────────────────────────────────

function getUser() {
  try {
    return JSON.parse(sessionStorage.getItem('gsi_user') || 'null');
  } catch (_) {
    return null;
  }
}

function logout() {
  sessionStorage.removeItem('gsi_user');
  window.location.replace('login.html');
}

// ── Page detection ─────────────────────────────────────────────
const isLoginPage = !!document.getElementById('loginForm');

if (isLoginPage) {

  // ── Login page: accept any credentials ────────────────────
  document.getElementById('loginForm').addEventListener('submit', function (e) {
    e.preventDefault();
    const username = (document.getElementById('loginUser').value || 'User').trim();
    sessionStorage.setItem('gsi_user', JSON.stringify({
      name:    username || 'User',
      email:   username,
      picture: '',
    }));
    window.location.replace('index.html');
  });

} else {

  // ── Protected page: auth guard ─────────────────────────────
  const user = getUser();

  if (!user) {
    window.location.replace('login.html');
  } else {
    const avatarEl  = document.getElementById('userAvatar');
    const nameEl    = document.getElementById('userName');
    const logoutBtn = document.getElementById('logoutBtn');

    if (avatarEl) { avatarEl.src = user.picture; avatarEl.alt = user.name; }
    if (nameEl)     nameEl.textContent = user.name || user.email;
    if (logoutBtn)  logoutBtn.addEventListener('click', logout);
  }
}
