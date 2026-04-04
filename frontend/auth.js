'use strict';

// ─────────────────────────────────────────────────────────────
// auth.js — Google Identity Services auth for AutiSENSE
//
// HOW TO GET YOUR CLIENT ID:
//   1. Go to https://console.cloud.google.com/
//   2. Create a project (or select an existing one)
//   3. Navigate to APIs & Services → Credentials
//   4. Click "Create Credentials" → "OAuth 2.0 Client ID"
//   5. Application type: Web application
//   6. Add Authorised JavaScript origin: http://localhost:5500
//   7. Copy the Client ID and paste it below
// ─────────────────────────────────────────────────────────────
const GOOGLE_CLIENT_ID = 'YOUR_CLIENT_ID_HERE.apps.googleusercontent.com';

// ── Utilities ─────────────────────────────────────────────────

/** Return the signed-in user object from session storage, or null. */
function getUser() {
  try {
    return JSON.parse(sessionStorage.getItem('gsi_user') || 'null');
  } catch (_) {
    return null;
  }
}

/** Sign out: clear session and redirect to login page. */
function logout() {
  sessionStorage.removeItem('gsi_user');
  try { google.accounts.id.disableAutoSelect(); } catch (_) {}
  window.location.replace('login.html');
}

// ── Decode a Google JWT credential (client-side only) ─────────
function decodeJwt(token) {
  const base64 = token.split('.')[1].replace(/-/g, '+').replace(/_/g, '/');
  const padded  = base64 + '='.repeat((4 - base64.length % 4) % 4);
  return JSON.parse(atob(padded));
}

// ── Page detection ─────────────────────────────────────────────
// login.html has #googleSignInDiv; index.html does not.
const isLoginPage = !!document.getElementById('googleSignInDiv');

if (isLoginPage) {

  // ── Login page: initialise Google Sign-In ──────────────────
  google.accounts.id.initialize({
    client_id: GOOGLE_CLIENT_ID,
    callback: function (response) {
      try {
        const payload = decodeJwt(response.credential);
        sessionStorage.setItem('gsi_user', JSON.stringify({
          name:    payload.name    || '',
          email:   payload.email   || '',
          picture: payload.picture || '',
        }));
        window.location.replace('index.html');
      } catch (_) {
        const el = document.getElementById('loginError');
        if (el) {
          el.textContent = '⚠ Sign-in failed. Please try again.';
          el.classList.remove('hidden');
        }
      }
    },
  });

  google.accounts.id.renderButton(
    document.getElementById('googleSignInDiv'),
    { theme: 'filled_blue', size: 'large', shape: 'pill', width: 260 }
  );

} else {

  // ── Protected page: auth guard ─────────────────────────────
  const user = getUser();

  if (!user) {
    // Not signed in — send to login
    window.location.replace('login.html');
  } else {
    // Populate header user info
    const avatarEl  = document.getElementById('userAvatar');
    const nameEl    = document.getElementById('userName');
    const logoutBtn = document.getElementById('logoutBtn');

    if (avatarEl) { avatarEl.src = user.picture; avatarEl.alt = user.name; }
    if (nameEl)     nameEl.textContent = user.name || user.email;
    if (logoutBtn)  logoutBtn.addEventListener('click', logout);
  }
}
