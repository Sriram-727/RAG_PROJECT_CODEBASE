/**
 * Unified Asymmetric RAG Verifier - Frontend Dashboard Controller
 */

document.addEventListener('DOMContentLoaded', () => {
  // Utility: HTML Escaper (handles string, numbers, objects safely)
  function escapeHtml(val) {
    if (val === null || val === undefined) return '';
    const str = typeof val === 'string' ? val : String(val);
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function formatTimestamp(ts) {
    if (!ts) return 'Just now';
    if (typeof ts === 'number') {
      const date = new Date(ts > 1e11 ? ts : ts * 1000);
      return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }
    return String(ts);
  }

  // Elements
  const healthDot = document.getElementById('health-dot');
  const healthLabel = document.getElementById('health-label');
  const verifyForm = document.getElementById('verify-form');
  const queryInput = document.getElementById('query-input');
  const userIdInput = document.getElementById('user-id-input');
  const orgIdInput = document.getElementById('org-id-input');
  const btnSubmit = document.getElementById('btn-submit');
  const btnSpinner = document.getElementById('btn-spinner');
  const btnText = document.getElementById('btn-text');

  // Hero Banner Elements
  const verdictBanner = document.getElementById('verdict-banner');
  const verdictBadge = document.getElementById('verdict-badge');
  const verdictTiming = document.getElementById('verdict-timing');
  const verdictHeadline = document.getElementById('verdict-headline');
  const verdictSubtext = document.getElementById('verdict-subtext');

  // Metrics
  const metricStatus = document.getElementById('metric-status');
  const metricGrounding = document.getElementById('metric-grounding');
  const groundingProgress = document.getElementById('grounding-progress');
  const metricCoverage = document.getElementById('metric-coverage');
  const coverageProgress = document.getElementById('coverage-progress');
  const metricChunks = document.getElementById('metric-chunks');

  // Gateways
  const cardGateAcl = document.getElementById('card-gate-acl');
  const pillGateAcl = document.getElementById('pill-gate-acl');
  const detailGateAcl = document.getElementById('detail-gate-acl');
  const metaAclUser = document.getElementById('meta-acl-user');
  const metaAclOrg = document.getElementById('meta-acl-org');

  const cardGateIntegrity = document.getElementById('card-gate-integrity');
  const pillGateIntegrity = document.getElementById('pill-gate-integrity');
  const detailGateIntegrity = document.getElementById('detail-gate-integrity');
  const metaIntegrityRoot = document.getElementById('meta-integrity-root');

  const cardGateDecryption = document.getElementById('card-gate-decryption');
  const pillGateDecryption = document.getElementById('pill-gate-decryption');
  const detailGateDecryption = document.getElementById('detail-gate-decryption');
  const metaDecryptedCount = document.getElementById('meta-decrypted-count');

  const cardGateBoundary = document.getElementById('card-gate-boundary');
  const pillGateBoundary = document.getElementById('pill-gate-boundary');
  const detailGateBoundary = document.getElementById('detail-gate-boundary');
  const tagsRequired = document.getElementById('tags-required-clusters');
  const tagsRetrieved = document.getElementById('tags-retrieved-clusters');
  const rowMissing = document.getElementById('row-missing-clusters');
  const tagsMissing = document.getElementById('tags-missing-clusters');

  const cardGateGrounding = document.getElementById('card-gate-grounding');
  const pillGateGrounding = document.getElementById('pill-gate-grounding');
  const detailGateGrounding = document.getElementById('detail-gate-grounding');
  const meterScoreText = document.getElementById('meter-score-text');
  const meterBarFill = document.getElementById('meter-bar-fill');

  // Answer & Chunks
  const maskedQueryVal = document.getElementById('masked-query-val');
  const answerContent = document.getElementById('answer-content');
  const btnCopyAnswer = document.getElementById('btn-copy-answer');
  const chunksCountBadge = document.getElementById('chunks-count-badge');
  const chunksList = document.getElementById('chunks-list');

  // Telemetry
  const telemetryJson = document.getElementById('telemetry-json');
  const btnCopyJson = document.getElementById('btn-copy-json');

  let lastResponseData = null;

  // 1. Health Check
  async function checkHealth() {
    try {
      const res = await fetch('/health');
      if (res.ok) {
        healthDot.className = 'status-dot online';
        healthLabel.textContent = 'API Connected (127.0.0.1:8000)';
      } else {
        throw new Error('API degraded');
      }
    } catch (e) {
      healthDot.className = 'status-dot offline';
      healthLabel.textContent = 'API Disconnected';
    }
  }

  checkHealth();
  setInterval(checkHealth, 10000);

  // 2. Organization Selector Handler
  const btnNewOrg = document.getElementById('btn-new-org');
  const btnViewOrgDocs = document.getElementById('btn-view-org-docs');
  const workbenchDocCount = document.getElementById('workbench-doc-count');
  const orgOptions = document.getElementById('org-options');

  let docCountDebounceTimer = null;
  async function updateWorkbenchDocCount(orgId) {
    if (!workbenchDocCount) return;
    if (!orgId || !orgId.trim()) {
      workbenchDocCount.textContent = '0';
      return;
    }
    try {
      const res = await fetch(`/api/v1/documents/${encodeURIComponent(orgId.trim())}`);
      if (res.ok) {
        const data = await res.json();
        workbenchDocCount.textContent = (data.documents ? data.documents.length : 0);
      } else {
        workbenchDocCount.textContent = '0';
      }
    } catch (e) {
      workbenchDocCount.textContent = '0';
    }
  }

  async function loadOrganizations() {
    try {
      const res = await fetch('/api/v1/organizations');
      if (!res.ok) return;
      const data = await res.json();
      const orgs = data.organizations || [];
      if (orgOptions) {
        orgOptions.innerHTML = '';
        orgs.forEach(org => {
          const opt = document.createElement('option');
          opt.value = org;
          opt.textContent = `${getOrgDisplayName(org)} (${org})`;
          orgOptions.appendChild(opt);
        });
      }
      // If orgIdInput is empty and we have active orgs, select first
      if (orgIdInput && !orgIdInput.value.trim() && orgs.length > 0) {
        orgIdInput.value = orgs[0];
      }
      if (orgIdInput && orgIdInput.value.trim()) {
        updateWorkbenchDocCount(orgIdInput.value.trim());
      }
    } catch (err) {
      console.warn('Failed to load organizations:', err);
    }
  }

  loadOrganizations();

  function setTenant(orgId, userId, label) {
    if (orgIdInput && orgId) orgIdInput.value = orgId;
    if (userIdInput && userId) userIdInput.value = userId;
    const ingestOrg = document.getElementById('ingest-org-id');
    const ingestUser = document.getElementById('ingest-user-id');
    const googleChooserTenant = document.getElementById('google-chooser-tenant');
    const signupOrg = document.getElementById('signup-org');

    if (ingestOrg && orgId) ingestOrg.value = orgId;
    if (ingestUser && userId) ingestUser.value = userId;
    if (googleChooserTenant && orgId) googleChooserTenant.value = orgId;
    if (signupOrg && orgId) signupOrg.value = orgId;

    updateWorkbenchDocCount(orgId);
  }

  orgIdInput?.addEventListener('input', () => {
    const val = orgIdInput.value.trim();
    clearTimeout(docCountDebounceTimer);
    docCountDebounceTimer = setTimeout(() => {
      setTenant(val, userIdInput ? userIdInput.value : '');
    }, 350);
  });

  btnNewOrg?.addEventListener('click', () => {
    if (orgIdInput) {
      orgIdInput.value = '';
      orgIdInput.placeholder = 'Type new Organization ID (e.g. my_project)...';
      orgIdInput.focus();
    }
    if (workbenchDocCount) workbenchDocCount.textContent = '0';
  });

  btnViewOrgDocs?.addEventListener('click', () => {
    openOrgDocumentsViewer();
  });

  // =========================================================================
  // AUTHENTICATION & MULTI-TENANT APPROVAL STATE MANAGEMENT
  // =========================================================================
  const userBadgeWrap = document.getElementById('user-badge-wrap');
  const sessionUsername = document.getElementById('session-username');
  const sessionRole = document.getElementById('session-role');
  const sessionOrg = document.getElementById('session-org');
  const sessionStatus = document.getElementById('session-status');
  const btnOpenAdminModal = document.getElementById('btn-open-admin-modal');
  const adminPendingBadge = document.getElementById('admin-pending-badge');
  const btnAuthAction = document.getElementById('btn-auth-action');
  const authActionText = document.getElementById('auth-action-text');

  const userStatusNotice = document.getElementById('user-status-notice');
  const userStatusNoticeText = document.getElementById('user-status-notice-text');

  // Modals
  const authModal = document.getElementById('auth-modal');
  const btnCloseAuthModal = document.getElementById('btn-close-auth-modal');
  const tabLoginBtn = document.getElementById('tab-login-btn');
  const tabSignupBtn = document.getElementById('tab-signup-btn');
  const tabLoginPanel = document.getElementById('tab-login-panel');
  const tabSignupPanel = document.getElementById('tab-signup-panel');

  const formLogin = document.getElementById('form-login');
  const loginUsername = document.getElementById('login-username');
  const loginPassword = document.getElementById('login-password');
  const loginFeedback = document.getElementById('login-feedback');
  const btnSubmitLogin = document.getElementById('btn-submit-login');

  const formSignup = document.getElementById('form-signup');
  const signupFullname = document.getElementById('signup-fullname');
  const signupEmail = document.getElementById('signup-email');
  const signupUsername = document.getElementById('signup-username');
  const signupPassword = document.getElementById('signup-password');
  const signupOrg = document.getElementById('signup-org');
  const signupFeedback = document.getElementById('signup-feedback');
  const btnSubmitSignup = document.getElementById('btn-submit-signup');
  const linkSwitchToSignup = document.getElementById('link-switch-to-signup');
  const linkSwitchToLogin = document.getElementById('link-switch-to-login');

  // LeetCode Google SSO & Account Chooser Elements
  const btnGoogleSso = document.getElementById('btn-google-sso');
  const btnGoogleSsoLabel = document.getElementById('btn-google-sso-label');
  const googleAccountModal = document.getElementById('google-account-modal');
  const btnCloseGoogleChooser = document.getElementById('btn-close-google-chooser');
  const googleChooserTenant = document.getElementById('google-chooser-tenant');
  const btnUseAnotherGoogleAccount = document.getElementById('btn-use-another-google-account');
  const googleCustomAccountBox = document.getElementById('google-custom-account-box');
  const googleCustomEmail = document.getElementById('google-custom-email');
  const googleCustomName = document.getElementById('google-custom-name');
  const btnSubmitCustomGoogle = document.getElementById('btn-submit-custom-google');
  const googleChooserFeedback = document.getElementById('google-chooser-feedback');

  const adminModal = document.getElementById('admin-modal');
  const adminModalSubtitle = document.getElementById('admin-modal-subtitle');
  const btnCloseAdminModal = document.getElementById('btn-close-admin-modal');
  const adminPendingTotal = document.getElementById('admin-pending-total');
  const btnApproveAllUsers = document.getElementById('btn-approve-all-users');
  const btnRefreshApprovals = document.getElementById('btn-refresh-approvals');
  const adminActionFeedback = document.getElementById('admin-action-feedback');
  const adminPendingList = document.getElementById('admin-pending-list');

  // Initial user session (persisted in localStorage)
  let currentSession = null;
  try {
    const saved = localStorage.getItem('rag_verifier_session');
    if (saved) currentSession = JSON.parse(saved);
  } catch (e) {}

  if (!currentSession) {
    currentSession = {
      username: 'admin',
      full_name: 'Administrator',
      email: 'admin@local.rag',
      org_id: 'default_org',
      role: 'admin',
      status: 'approved'
    };
  }

  function getOrgDisplayName(orgId) {
    if (!orgId) return 'Default Org';
    return orgId
      .replace(/^org_/, '')
      .replace(/[_-]+/g, ' ')
      .trim()
      .replace(/\b\w/g, c => c.toUpperCase()) || orgId;
  }

  async function updateAdminBadgeCount() {
    if (!currentSession || currentSession.role !== 'admin') {
      if (btnOpenAdminModal) btnOpenAdminModal.style.display = 'none';
      return;
    }
    try {
      const res = await fetch(`/api/v1/admin/pending-users?admin_id=${encodeURIComponent(currentSession.username)}`);
      if (res.ok) {
        const data = await res.json();
        const pendingCount = data.total_pending || 0;
        if (adminPendingBadge) adminPendingBadge.textContent = pendingCount;
        if (btnOpenAdminModal) btnOpenAdminModal.style.display = 'inline-flex';
      } else {
        if (btnOpenAdminModal) btnOpenAdminModal.style.display = 'none';
      }
    } catch (err) {
      console.warn('Failed to fetch pending admin approvals:', err);
    }
  }

  function renderSessionUI(syncInputs = true) {
    if (!currentSession) {
      if (userBadgeWrap) userBadgeWrap.style.display = 'none';
      if (btnOpenAdminModal) btnOpenAdminModal.style.display = 'none';
      if (authActionText) authActionText.textContent = 'Sign In / Register';
      return;
    }

    if (userBadgeWrap) userBadgeWrap.style.display = 'inline-flex';
    if (sessionUsername) {
      if (currentSession.auth_provider === 'google') {
        sessionUsername.innerHTML = `<span class="sso-badge-google">G</span> ${escapeHtml(currentSession.username)}`;
      } else {
        sessionUsername.textContent = currentSession.username;
      }
    }
    
    if (sessionRole) {
      sessionRole.textContent = currentSession.role.toUpperCase();
      sessionRole.className = `user-badge-role ${currentSession.role === 'admin' ? 'role-admin' : 'role-user'}`;
    }

    if (sessionOrg) {
      sessionOrg.textContent = getOrgDisplayName(currentSession.org_id);
    }

    if (sessionStatus) {
      sessionStatus.textContent = currentSession.status.toUpperCase();
      sessionStatus.className = `status-tag ${currentSession.status}`;
    }

    if (authActionText) {
      authActionText.textContent = 'Switch User / Logout';
    }

    // Synchronize Form inputs
    if (syncInputs) {
      userIdInput.value = currentSession.username;
      orgIdInput.value = currentSession.org_id;
      setTenant(currentSession.org_id, currentSession.username);
    }

    // Update workbench notice banner
    if (userStatusNotice && userStatusNoticeText) {
      userStatusNotice.style.display = 'flex';
      const ssoTag = currentSession.auth_provider === 'google' ? ' <span class="sso-badge-google">Google SSO</span>' : '';
      if (currentSession.status === 'pending') {
        userStatusNotice.className = 'user-notice-banner notice-pending';
        userStatusNoticeText.innerHTML = `<strong>Pending Approval:</strong> User <code>${escapeHtml(currentSession.username)}</code> requested access to <strong>${getOrgDisplayName(currentSession.org_id)}</strong>.`;
      } else if (currentSession.status === 'rejected') {
        userStatusNotice.className = 'user-notice-banner notice-rejected';
        userStatusNoticeText.innerHTML = `<strong>Access Denied:</strong> User <code>${escapeHtml(currentSession.username)}</code> was not approved for <strong>${getOrgDisplayName(currentSession.org_id)}</strong>.`;
      } else {
        userStatusNotice.className = 'user-notice-banner notice-approved';
        userStatusNoticeText.innerHTML = `<strong>Session:</strong> <code>${escapeHtml(currentSession.username)}</code> (${escapeHtml(currentSession.full_name || 'User')}) &bull; Role: <strong>${escapeHtml(currentSession.role.toUpperCase())}</strong> &bull; Org: <strong>${getOrgDisplayName(currentSession.org_id)}</strong> &bull; Status: <strong>Approved</strong>`;
      }
    }

    updateAdminBadgeCount();
  }

  // --- LEETCODE-STYLE AUTHENTICATION & GOOGLE SSO CONTROLLER ---
  function activateAuthTab(tab) {
    if (tab === 'signup') {
      tabSignupBtn?.classList.add('active');
      tabLoginBtn?.classList.remove('active');
      if (tabSignupPanel) tabSignupPanel.style.display = 'block';
      if (tabLoginPanel) tabLoginPanel.style.display = 'none';
      if (btnGoogleSsoLabel) btnGoogleSsoLabel.textContent = 'Sign up with Google';
    } else {
      tabLoginBtn?.classList.add('active');
      tabSignupBtn?.classList.remove('active');
      if (tabLoginPanel) tabLoginPanel.style.display = 'block';
      if (tabSignupPanel) tabSignupPanel.style.display = 'none';
      if (btnGoogleSsoLabel) btnGoogleSsoLabel.textContent = 'Continue with Google';
    }
  }

  tabLoginBtn?.addEventListener('click', () => activateAuthTab('login'));
  tabSignupBtn?.addEventListener('click', () => activateAuthTab('signup'));
  linkSwitchToSignup?.addEventListener('click', (e) => { e.preventDefault(); activateAuthTab('signup'); });
  linkSwitchToLogin?.addEventListener('click', (e) => { e.preventDefault(); activateAuthTab('login'); });

  // Open & Close Auth Modal
  btnAuthAction?.addEventListener('click', () => {
    if (loginFeedback) loginFeedback.style.display = 'none';
    if (signupFeedback) signupFeedback.style.display = 'none';
    activateAuthTab('login');
    authModal?.showModal();
  });

  btnCloseAuthModal?.addEventListener('click', () => {
    authModal?.close();
  });

  authModal?.addEventListener('click', (e) => {
    if (e.target === authModal) authModal.close();
  });

  // Open Google Account Chooser (LeetCode-Style)
  btnGoogleSso?.addEventListener('click', () => {
    if (googleChooserFeedback) {
      googleChooserFeedback.style.display = 'none';
      googleChooserFeedback.textContent = '';
    }
    if (googleCustomAccountBox) {
      googleCustomAccountBox.style.display = 'none';
    }
    // Sync target organization to currently selected tenant in workbench
    if (googleChooserTenant && orgIdInput) {
      const activeOrg = orgIdInput.value.trim();
      if (activeOrg) {
        googleChooserTenant.value = activeOrg;
      }
    }
    googleAccountModal?.showModal();
  });

  btnCloseGoogleChooser?.addEventListener('click', () => {
    googleAccountModal?.close();
  });

  googleAccountModal?.addEventListener('click', (e) => {
    if (e.target === googleAccountModal) googleAccountModal.close();
  });

  // Simple string hash helper for deterministic sub ID
  function getGoogleSubId(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      hash = ((hash << 5) - hash) + str.charCodeAt(i);
      hash |= 0;
    }
    return 'google_sub_' + Math.abs(hash);
  }

  // Google SSO Execution (JIT Provisioning or returning login)
  async function performGoogleSSO(email, fullName) {
    const targetOrg = (googleChooserTenant && googleChooserTenant.value.trim()) || (orgIdInput && orgIdInput.value.trim()) || 'default_org';
    
    if (googleChooserFeedback) {
      googleChooserFeedback.style.display = 'block';
      googleChooserFeedback.style.background = 'rgba(59, 130, 246, 0.15)';
      googleChooserFeedback.style.border = '1px solid rgba(59, 130, 246, 0.4)';
      googleChooserFeedback.style.color = '#bfdbfe';
      googleChooserFeedback.textContent = `Signing in with Google as ${fullName} (${email})...`;
    }

    try {
      const res = await fetch('/api/v1/auth/google', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email: email,
          full_name: fullName,
          target_org_id: targetOrg,
          google_id: getGoogleSubId(email)
        })
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Google sign in failed');
      }

      currentSession = data.user;
      localStorage.setItem('rag_verifier_session', JSON.stringify(currentSession));
      renderSessionUI(true);

      if (googleChooserFeedback) {
        if (currentSession.status === 'approved') {
          googleChooserFeedback.style.background = 'rgba(16, 185, 129, 0.15)';
          googleChooserFeedback.style.border = '1px solid rgba(16, 185, 129, 0.4)';
          googleChooserFeedback.style.color = '#6ee7b7';
          googleChooserFeedback.innerHTML = `✅ <strong>Signed in!</strong> Welcome back, <strong>${escapeHtml(currentSession.full_name)}</strong>.`;
        } else {
          googleChooserFeedback.style.background = 'rgba(245, 158, 11, 0.15)';
          googleChooserFeedback.style.border = '1px solid rgba(245, 158, 11, 0.4)';
          googleChooserFeedback.style.color = '#fde68a';
          googleChooserFeedback.innerHTML = `⏳ <strong>Account Registered!</strong> User <code>${escapeHtml(currentSession.username)}</code> pending Admin approval for <strong>${getOrgDisplayName(currentSession.org_id)}</strong>.`;
        }
      }

      setTimeout(() => {
        googleAccountModal?.close();
        authModal?.close();
      }, 1000);

    } catch (err) {
      if (googleChooserFeedback) {
        googleChooserFeedback.style.display = 'block';
        googleChooserFeedback.style.background = 'rgba(239, 68, 68, 0.15)';
        googleChooserFeedback.style.border = '1px solid rgba(239, 68, 68, 0.4)';
        googleChooserFeedback.style.color = '#fca5a5';
        googleChooserFeedback.textContent = `❌ ${err.message}`;
      }
    }
  }

  // Bind predefined Google Account cards in the chooser
  document.querySelectorAll('.google-account-item[data-email]').forEach(item => {
    item.addEventListener('click', () => {
      const email = item.dataset.email;
      const name = item.dataset.name || email.split('@')[0];
      performGoogleSSO(email, name);
    });
  });

  // Toggle "Use another account" input form
  btnUseAnotherGoogleAccount?.addEventListener('click', () => {
    if (googleCustomAccountBox) {
      const isHidden = googleCustomAccountBox.style.display === 'none' || !googleCustomAccountBox.style.display;
      googleCustomAccountBox.style.display = isHidden ? 'block' : 'none';
      if (isHidden && googleCustomEmail) {
        googleCustomEmail.focus();
      }
    }
  });

  // Submit custom Google account
  btnSubmitCustomGoogle?.addEventListener('click', () => {
    const email = googleCustomEmail ? googleCustomEmail.value.trim() : '';
    const name = googleCustomName ? googleCustomName.value.trim() : '';
    
    if (!email || !email.includes('@')) {
      if (googleChooserFeedback) {
        googleChooserFeedback.style.display = 'block';
        googleChooserFeedback.style.background = 'rgba(239, 68, 68, 0.15)';
        googleChooserFeedback.style.border = '1px solid rgba(239, 68, 68, 0.4)';
        googleChooserFeedback.style.color = '#fca5a5';
        googleChooserFeedback.textContent = '❌ Please enter a valid Google email address.';
      }
      return;
    }

    const displayName = name || email.split('@')[0];
    performGoogleSSO(email, displayName);
  });

  // Quick Admin shortcuts
  document.querySelectorAll('.admin-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      loginUsername.value = chip.dataset.user;
      loginPassword.value = chip.dataset.pass;
      loginFeedback.style.display = 'none';
      loginUsername.focus();
    });
  });

  // Login form submission
  formLogin?.addEventListener('submit', async (e) => {
    e.preventDefault();
    btnSubmitLogin.disabled = true;
    loginFeedback.style.display = 'none';

    try {
      const res = await fetch('/api/v1/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          username: loginUsername.value.trim(),
          password: loginPassword.value
        })
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Login failed');
      }

      currentSession = data.user;
      localStorage.setItem('rag_verifier_session', JSON.stringify(currentSession));
      renderSessionUI(true);
      authModal.close();
    } catch (err) {
      loginFeedback.style.display = 'block';
      loginFeedback.style.background = 'rgba(239, 68, 68, 0.15)';
      loginFeedback.style.border = '1px solid rgba(239, 68, 68, 0.4)';
      loginFeedback.style.color = '#fca5a5';
      loginFeedback.textContent = `❌ ${err.message}`;
    } finally {
      btnSubmitLogin.disabled = false;
    }
  });

  // Signup form submission
  formSignup?.addEventListener('submit', async (e) => {
    e.preventDefault();
    btnSubmitSignup.disabled = true;
    signupFeedback.style.display = 'none';

    try {
      const res = await fetch('/api/v1/auth/signup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          full_name: signupFullname.value.trim(),
          email: signupEmail.value.trim(),
          username: signupUsername.value.trim(),
          password: signupPassword.value,
          org_id: signupOrg.value
        })
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Registration failed');
      }

      currentSession = data.user;
      localStorage.setItem('rag_verifier_session', JSON.stringify(currentSession));
      renderSessionUI(true);

      signupFeedback.style.display = 'block';
      signupFeedback.style.background = 'rgba(16, 185, 129, 0.15)';
      signupFeedback.style.border = '1px solid rgba(16, 185, 129, 0.4)';
      signupFeedback.style.color = '#6ee7b7';
      signupFeedback.textContent = `✅ Account created in 'pending' status! You are now logged in as '${currentSession.username}'. Queries will be blocked at Gate 1 until your Org Admin approves.`;

      setTimeout(() => {
        authModal.close();
      }, 1500);
    } catch (err) {
      signupFeedback.style.display = 'block';
      signupFeedback.style.background = 'rgba(239, 68, 68, 0.15)';
      signupFeedback.style.border = '1px solid rgba(239, 68, 68, 0.4)';
      signupFeedback.style.color = '#fca5a5';
      signupFeedback.textContent = `❌ ${err.message}`;
    } finally {
      btnSubmitSignup.disabled = false;
    }
  });

  // Admin Approvals Modal
  btnOpenAdminModal?.addEventListener('click', () => {
    adminActionFeedback.style.display = 'none';
    adminModal.showModal();
    loadAdminPendingUsers();
  });

  btnCloseAdminModal?.addEventListener('click', () => {
    adminModal.close();
  });

  adminModal?.addEventListener('click', (e) => {
    if (e.target === adminModal) adminModal.close();
  });

  btnRefreshApprovals?.addEventListener('click', () => {
    loadAdminPendingUsers();
  });

  btnApproveAllUsers?.addEventListener('click', async () => {
    if (!currentSession || currentSession.role !== 'admin') return;
    const confirmApprove = confirm('Approve all pending user requests across all organizations?');
    if (!confirmApprove) return;

    try {
      btnApproveAllUsers.disabled = true;
      btnApproveAllUsers.textContent = 'Approving...';
      const res = await fetch(`/api/v1/admin/approve-all?admin_id=${encodeURIComponent(currentSession.username)}`, {
        method: 'POST'
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Bulk approval failed');
      }

      adminActionFeedback.style.display = 'block';
      adminActionFeedback.style.background = 'rgba(16, 185, 129, 0.15)';
      adminActionFeedback.style.border = '1px solid rgba(16, 185, 129, 0.4)';
      adminActionFeedback.style.color = '#6ee7b7';
      adminActionFeedback.textContent = data.message || `Successfully approved ${data.approved_count} user(s).`;
      loadAdminPendingUsers();
    } catch (err) {
      adminActionFeedback.style.display = 'block';
      adminActionFeedback.style.background = 'rgba(239, 68, 68, 0.15)';
      adminActionFeedback.style.border = '1px solid rgba(239, 68, 68, 0.4)';
      adminActionFeedback.style.color = '#fca5a5';
      adminActionFeedback.textContent = `❌ ${err.message}`;
    } finally {
      btnApproveAllUsers.disabled = false;
      btnApproveAllUsers.textContent = '✓ Approve All';
    }
  });

  async function loadAdminPendingUsers() {
    if (!currentSession || currentSession.role !== 'admin') return;

    if (adminModalSubtitle) {
      adminModalSubtitle.innerHTML = `Admin: <strong>${escapeHtml(currentSession.username)}</strong> &bull; Multi-Org User Approval Authority`;
    }
    adminPendingList.innerHTML = `<div style="text-align: center; padding: 1.5rem; color: #94a3b8;">Loading pending requests...</div>`;

    try {
      const res = await fetch(`/api/v1/admin/pending-users?admin_id=${encodeURIComponent(currentSession.username)}`);
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Failed to fetch pending users');
      }

      const pendingUsers = data.pending_users || [];
      if (adminPendingTotal) adminPendingTotal.textContent = pendingUsers.length;
      if (adminPendingBadge) adminPendingBadge.textContent = pendingUsers.length;

      if (pendingUsers.length === 0) {
        adminPendingList.innerHTML = `
          <div style="text-align: center; padding: 2rem; color: #94a3b8; font-size: 0.85rem;">
            ✅ All clear! There are no pending sign-up requests awaiting approval across any organization.
          </div>
        `;
        return;
      }

      let html = '';
      pendingUsers.forEach(u => {
        html += `
          <div class="approval-user-card" id="user-card-${escapeHtml(u.username)}">
            <div class="approval-user-info">
              <div class="approval-user-name">
                ${escapeHtml(u.full_name || u.username)} 
                <span style="font-family: monospace; font-size: 0.76rem; color: #a5b4fc;">(@${escapeHtml(u.username)})</span>
                ${u.auth_provider === 'google' ? '<span class="sso-badge-google">Google SSO</span>' : ''}
              </div>
              <div class="approval-user-meta">
                <span>📧 ${escapeHtml(u.email || 'N/A')}</span> &bull; 
                <span>🏢 ${escapeHtml(u.org_id)}</span> &bull; 
                <span>⏱️ ${escapeHtml(formatTimestamp(u.created_at))}</span>
              </div>
            </div>
            <div class="approval-actions">
              <button type="button" class="btn-approve" data-user="${escapeHtml(u.username)}">
                ✓ Approve
              </button>
              <button type="button" class="btn-reject" data-user="${escapeHtml(u.username)}">
                ✕ Reject
              </button>
            </div>
          </div>
        `;
      });

      adminPendingList.innerHTML = html;

      // Wire action buttons
      adminPendingList.querySelectorAll('.btn-approve').forEach(btn => {
        btn.addEventListener('click', async () => {
          const target = btn.dataset.user;
          await handleApprovalDecision(target, 'approved');
        });
      });

      adminPendingList.querySelectorAll('.btn-reject').forEach(btn => {
        btn.addEventListener('click', async () => {
          const target = btn.dataset.user;
          await handleApprovalDecision(target, 'rejected');
        });
      });

    } catch (err) {
      adminPendingList.innerHTML = `
        <div style="padding: 1rem; color: #f87171; background: rgba(239, 68, 68, 0.1); border-radius: 8px;">
          Failed to load approvals: ${escapeHtml(err.message)}
        </div>
      `;
    }
  }

  async function handleApprovalDecision(targetUsername, decision) {
    try {
      const res = await fetch('/api/v1/admin/approve-user', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          admin_id: currentSession.username,
          target_username: targetUsername,
          status: decision
        })
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Approval request failed');
      }

      adminActionFeedback.style.display = 'block';
      adminActionFeedback.style.background = decision === 'approved' ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)';
      adminActionFeedback.style.border = decision === 'approved' ? '1px solid rgba(16, 185, 129, 0.4)' : '1px solid rgba(239, 68, 68, 0.4)';
      adminActionFeedback.style.color = decision === 'approved' ? '#6ee7b7' : '#fca5a5';
      adminActionFeedback.textContent = `User '${targetUsername}' has been ${decision.toUpperCase()}.`;

      // Refresh list
      loadAdminPendingUsers();
    } catch (err) {
      adminActionFeedback.style.display = 'block';
      adminActionFeedback.style.background = 'rgba(239, 68, 68, 0.15)';
      adminActionFeedback.style.border = '1px solid rgba(239, 68, 68, 0.4)';
      adminActionFeedback.style.color = '#fca5a5';
      adminActionFeedback.textContent = `❌ ${err.message}`;
    }
  }

  // Initialize Session UI on load
  renderSessionUI(true);

  // 4. Keyboard shortcut: Ctrl + Enter to submit
  document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      verifyForm.requestSubmit();
    }
  });

  // 5. Verification Form Submission
  verifyForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const queryText = queryInput.value.trim();
    if (!queryText) return;

    // Loading State
    btnSubmit.classList.add('loading');
    btnSubmit.disabled = true;
    if (verdictBadge) verdictBadge.textContent = 'SEARCHING...';
    if (verdictHeadline) verdictHeadline.textContent = 'Searching Documents...';
    if (verdictSubtext) verdictSubtext.textContent = 'Retrieving encrypted chunks and verifying security...';
    if (verdictBanner) verdictBanner.className = 'verdict-banner';
    if (answerContent) {
      answerContent.innerHTML = '<p class="placeholder-text">Searching documents...</p>';
    }

    const payload = {
      query_text: queryText,
      user_id: userIdInput.value.trim(),
      org_id: orgIdInput.value.trim(),
      simulate_missing_cluster: false,
      simulate_integrity_failure: false
    };

    const startTime = performance.now();

    try {
      const response = await fetch('/api/v1/query-verify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      const clientLatency = Math.round(performance.now() - startTime);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'HTTP ' + response.status }));
        throw new Error(errorData.detail || 'Pipeline execution returned error');
      }

      const data = await response.json();
      lastResponseData = data;
      renderVerificationResults(data, clientLatency);

    } catch (err) {
      console.error('Verification error:', err);
      renderPipelineError(err.message);
    } finally {
      btnSubmit.classList.remove('loading');
      btnSubmit.disabled = false;
    }
  });

  // 6. Render Verification Results
  function renderVerificationResults(data, clientLatency) {
    const latency = data.execution_time_ms || clientLatency;
    if (verdictTiming) verdictTiming.textContent = `Latency: ${latency} ms`;

    // Telemetry JSON (safe check)
    if (telemetryJson) telemetryJson.textContent = JSON.stringify(data, null, 2);

    const breakdown = data.verification_breakdown || {};
    const aclInfo = breakdown.acl || { passed: data.acl_verified ?? true };
    const integrityInfo = breakdown.integrity || { passed: data.integrity_verified };
    const decryptionInfo = breakdown.decryption || { passed: data.decryption_verified ?? true };
    const boundaryInfo = breakdown.boundary || { 
      passed: data.ivf_boundary_verification?.is_complete ?? true,
      coverage_ratio: data.ivf_boundary_verification?.coverage_ratio ?? 1.0,
      required_clusters: data.ivf_boundary_verification?.required_clusters || [],
      retrieved_clusters: data.ivf_boundary_verification?.retrieved_clusters || [],
      missing_clusters: data.ivf_boundary_verification?.missing_clusters || []
    };
    const groundingInfo = breakdown.grounding || {
      passed: !data.hallucination_report?.flagged,
      grounding_score: data.hallucination_report?.grounding_score || 0.0
    };

    // Overall Status Banner
    const isSuccess = data.status === 'SUCCESS' && !data.hallucination_report?.flagged;
    if (verdictBanner) {
      if (isSuccess) {
        verdictBanner.className = 'verdict-banner verdict-success';
        if (verdictBadge) verdictBadge.textContent = 'VERIFIED & SECURE';
        if (verdictHeadline) verdictHeadline.textContent = 'Pipeline Verification Passed (100% Validated)';
        if (verdictSubtext) verdictSubtext.textContent = 'Multi-tenant ACL permissions, cryptographic MMR integrity proof, and authenticated AES-256 decryption verified.';
      } else {
        verdictBanner.className = 'verdict-banner verdict-fail';
        if (verdictBadge) verdictBadge.textContent = data.status;
        if (verdictHeadline) verdictHeadline.textContent = `Verification Halted: ${data.status}`;
        if (verdictSubtext) verdictSubtext.textContent = data.hallucination_report?.reason || 'Pipeline verification detected anomaly or security failure.';
      }
    }

    // Top Metrics
    if (metricStatus) {
      metricStatus.textContent = data.status;
      metricStatus.style.color = isSuccess ? 'var(--status-pass)' : 'var(--status-fail)';
    }

    const score = groundingInfo.grounding_score || 0.0;
    const scorePct = (score * 100).toFixed(1);
    if (metricGrounding) metricGrounding.textContent = `${scorePct}%`;
    if (groundingProgress) {
      groundingProgress.style.width = `${Math.min(100, Math.max(0, score * 100))}%`;
      groundingProgress.style.background = score >= 0.35 ? '#10b981' : '#f59e0b';
    }

    const coverage = (boundaryInfo.coverage_ratio !== undefined ? boundaryInfo.coverage_ratio : 1.0) * 100;
    if (metricCoverage) metricCoverage.textContent = `${coverage.toFixed(1)}%`;
    if (coverageProgress) {
      coverageProgress.style.width = `${coverage}%`;
      coverageProgress.style.background = coverage >= 100 ? 'var(--status-pass)' : 'var(--status-warn)';
    }

    const count = data.retrieved_chunks_count ?? 0;
    if (metricChunks) metricChunks.textContent = count;
    const metricChunksBadge = document.getElementById('metric-chunks-badge');
    if (metricChunksBadge) {
      if (data.status === 'ACL_VIOLATION' || !data.acl_verified) {
        metricChunksBadge.textContent = 'Access Blocked (0 Chunks)';
        metricChunksBadge.style.color = 'var(--color-danger)';
      } else {
        metricChunksBadge.textContent = `${count} Chunks Verified`;
        metricChunksBadge.style.color = '';
      }
    }

    // --- GATEWAY 1: ACL ---
    updateGateCard(
      cardGateAcl, pillGateAcl, detailGateAcl,
      aclInfo.passed,
      aclInfo.passed ? 'PASS' : 'BLOCKED'
    );
    if (metaAclUser) metaAclUser.textContent = aclInfo.user_id || userIdInput.value;
    if (metaAclOrg) metaAclOrg.textContent = aclInfo.org_id || orgIdInput.value;

    // --- GATEWAY 2: INTEGRITY ---
    updateGateCard(
      cardGateIntegrity, pillGateIntegrity, detailGateIntegrity,
      integrityInfo.passed,
      integrityInfo.passed ? 'VALID' : 'TAMPERED'
    );
    if (metaIntegrityRoot && integrityInfo.enterprise_root) {
      metaIntegrityRoot.textContent = integrityInfo.enterprise_root;
    }

    // --- GATEWAY 3: DECRYPTION ---
    updateGateCard(
      cardGateDecryption, pillGateDecryption, detailGateDecryption,
      decryptionInfo.passed,
      decryptionInfo.passed ? 'DECRYPTED' : 'FAIL'
    );
    if (metaDecryptedCount) metaDecryptedCount.textContent = `${data.retrieved_chunks_count || 0} chunks`;

    // --- GATEWAY 4: BOUNDARY ---
    updateGateCard(
      cardGateBoundary, pillGateBoundary, detailGateBoundary,
      boundaryInfo.passed,
      boundaryInfo.passed ? `${coverage.toFixed(0)}%` : 'DROPPED'
    );
    if (tagsRequired) renderClusterTags(tagsRequired, boundaryInfo.required_clusters || [], 'required');
    if (tagsRetrieved) renderClusterTags(tagsRetrieved, boundaryInfo.retrieved_clusters || [], 'retrieved');
    
    const missing = boundaryInfo.missing_clusters || [];
    if (rowMissing) {
      if (missing.length > 0) {
        rowMissing.style.display = 'flex';
        if (tagsMissing) renderClusterTags(tagsMissing, missing, 'missing');
      } else {
        rowMissing.style.display = 'none';
      }
    }

    // --- GATEWAY 5: GROUNDING ---
    updateGateCard(
      cardGateGrounding, pillGateGrounding, detailGateGrounding,
      groundingInfo.passed,
      groundingInfo.passed ? `${scorePct}%` : 'LOW'
    );
    if (meterScoreText) meterScoreText.textContent = score.toFixed(4);
    if (meterBarFill) {
      meterBarFill.style.width = `${Math.min(100, Math.max(0, score * 100))}%`;
      meterBarFill.style.background = score >= 0.35 ? 'var(--status-pass)' : 'var(--status-fail)';
    }

    // Masked Query Preview
    if (maskedQueryVal) maskedQueryVal.textContent = data.masked_query || queryInput.value;

    // Verified LLM Answer
    if (answerContent) {
      if (data.status === 'ACL_VIOLATION' || !data.acl_verified) {
        answerContent.innerHTML = `
          <div style="color: #f87171; background: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.25); padding: 14px 16px; border-radius: 6px; line-height: 1.5;">
            <strong>Cross-Tenant Access Blocked:</strong> ${escapeHtml(data.answer || 'Access denied by multi-tenant ACL gate.')}
          </div>
        `;
      } else if (data.answer) {
        answerContent.innerHTML = `<p>${escapeHtml(data.answer)}</p>`;
      } else {
        answerContent.innerHTML = `
          <div style="color: #f87171; background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); padding: 12px 14px; border-radius: 6px;">
            <strong>Generation Blocked:</strong> ${escapeHtml(data.hallucination_report?.reason || 'Verified context not available.')}
          </div>
        `;
      }
    }

    // Decrypted Context Chunks (safe check)
    const chunks = data.decrypted_chunks || [];
    if (chunksCountBadge) chunksCountBadge.textContent = chunks.length;
    if (chunksList) {
      if (chunks.length > 0) {
        chunksList.innerHTML = chunks.map(chunk => `
          <div class="chunk-card">
            <div class="chunk-meta-row">
              <span class="chunk-badge">Chunk #${chunk.chunk_index} &bull; Cluster ${chunk.cluster_id}</span>
              <span>Similarity: ${(chunk.score * 100).toFixed(1)}%</span>
            </div>
            <div class="chunk-text">${escapeHtml(chunk.text)}</div>
          </div>
        `).join('');
      } else {
        chunksList.innerHTML = `<p class="placeholder-text" style="color: #fca5a5;">No context chunks available (verification rejected or empty retrieval).</p>`;
      }
    }
  }

  function updateGateCard(cardEl, pillEl, detailEl, passed, detailText) {
    if (cardEl) {
      cardEl.classList.remove('gate-pass', 'gate-fail', 'gate-warn');
      cardEl.classList.add(passed ? 'gate-pass' : 'gate-fail');
    }
    if (pillEl) {
      pillEl.textContent = detailText || (passed ? 'PASS' : 'FAIL');
      pillEl.className = `gate-ribbon-pill ${passed ? 'pass' : 'fail'}`;
    }
    if (detailEl) detailEl.textContent = detailText;
  }

  function renderClusterTags(container, list, type) {
    if (!container) return;
    if (!list || list.length === 0) {
      container.innerHTML = `<span class="empty-tag">None</span>`;
      return;
    }
    container.innerHTML = list.map(c => `
      <span class="c-tag ${type}">C-${c}</span>
    `).join('');
  }

  function renderPipelineError(errorMessage) {
    if (verdictBanner) verdictBanner.className = 'verdict-banner verdict-fail';
    if (verdictBadge) verdictBadge.textContent = 'API ERROR';
    if (verdictHeadline) verdictHeadline.textContent = 'Pipeline Execution Request Failed';
    if (verdictSubtext) verdictSubtext.textContent = errorMessage;
    if (metricStatus) {
      metricStatus.textContent = 'ERROR';
      metricStatus.style.color = 'var(--status-fail)';
    }
    if (answerContent) {
      answerContent.innerHTML = `
        <div style="color: #f87171; background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); padding: 12px 14px; border-radius: 6px;">
          <strong>Request Failed:</strong> ${escapeHtml(errorMessage)}
        </div>
      `;
    }
  }

  // Copy Answer
  btnCopyAnswer?.addEventListener('click', () => {
    const text = answerContent?.innerText;
    if (text) {
      navigator.clipboard.writeText(text).then(() => {
        const span = btnCopyAnswer.querySelector('span');
        if (span) {
          const orig = span.textContent;
          span.textContent = 'Copied!';
          setTimeout(() => { span.textContent = orig; }, 1500);
        }
      });
    }
  });

  // Copy JSON (safe check)
  btnCopyJson?.addEventListener('click', () => {
    const jsonStr = telemetryJson?.innerText;
    if (jsonStr) {
      navigator.clipboard.writeText(jsonStr).then(() => {
        btnCopyJson.textContent = 'Copied!';
        setTimeout(() => { btnCopyJson.textContent = 'Copy JSON'; }, 1500);
      });
    }
  });

  // (Secondary reference pointing to safe top-level escapeHtml)
  // --- Document Ingestion Modal & Handlers ---
  const btnOpenIngestModal = document.getElementById('btn-open-ingest-modal');
  const btnCloseIngestModal = document.getElementById('btn-close-ingest-modal');
  const ingestModal = document.getElementById('ingest-modal');
  const tabIngestFile = document.getElementById('tab-ingest-file');
  const tabIngestText = document.getElementById('tab-ingest-text');
  const fileUploadWrap = document.getElementById('file-upload-wrap');
  const textUploadWrap = document.getElementById('text-upload-wrap');
  const btnRunIngest = document.getElementById('btn-run-ingest');
  const ingestBtnText = document.getElementById('ingest-btn-text');
  const ingestStatusBox = document.getElementById('ingest-status-box');

  if (btnOpenIngestModal && ingestModal) {
    const ingestOrg = document.getElementById('ingest-org-id');
    const ingestUser = document.getElementById('ingest-user-id');

    if (ingestOrg && ingestUser) {
      ingestOrg.addEventListener('input', () => {
        const val = ingestOrg.value.trim();
        if (val) {
          const clean = val.replace(/^org_/, '').replace(/[^a-zA-Z0-9_]/g, '_').toLowerCase();
          ingestUser.value = `usr_${clean || 'admin'}`;
        }
      });
    }

    btnOpenIngestModal.addEventListener('click', () => {
      if (ingestOrg && orgIdInput) ingestOrg.value = orgIdInput.value.trim() || '';
      if (ingestUser && userIdInput) ingestUser.value = userIdInput.value.trim() || 'usr_admin';
      if (ingestStatusBox) ingestStatusBox.style.display = 'none';
      ingestModal.showModal();
    });
  }

  if (btnCloseIngestModal && ingestModal) {
    btnCloseIngestModal.addEventListener('click', () => {
      ingestModal.close();
    });
    ingestModal.addEventListener('click', (e) => {
      if (e.target === ingestModal) {
        ingestModal.close();
      }
    });
  }

  if (tabIngestFile && tabIngestText) {
    tabIngestFile.addEventListener('click', () => {
      tabIngestFile.classList.add('active');
      tabIngestText.classList.remove('active');
      fileUploadWrap.style.display = 'block';
      textUploadWrap.style.display = 'none';
    });

    tabIngestText.addEventListener('click', () => {
      tabIngestText.classList.add('active');
      tabIngestFile.classList.remove('active');
      fileUploadWrap.style.display = 'none';
      textUploadWrap.style.display = 'block';
    });
  }

  // Auto-populate Doc ID from uploaded file name
  const ingestFileInput = document.getElementById('ingest-file-input');
  if (ingestFileInput) {
    ingestFileInput.addEventListener('change', () => {
      if (ingestFileInput.files && ingestFileInput.files.length > 0) {
        const fname = ingestFileInput.files[0].name;
        const cleanName = 'doc_' + fname.replace(/\.[^/.]+$/, '').replace(/[^a-zA-Z0-9_-]/g, '_').toLowerCase();
        const docIdInput = document.getElementById('ingest-doc-id');
        if (docIdInput) {
          docIdInput.value = cleanName;
        }
      }
    });
  }

  if (btnRunIngest) {
    btnRunIngest.addEventListener('click', async () => {
      let docId = document.getElementById('ingest-doc-id').value.trim();
      const userId = document.getElementById('ingest-user-id').value.trim() || 'usr_admin';
      const orgId = document.getElementById('ingest-org-id').value.trim() || (orgIdInput ? orgIdInput.value.trim() : '') || 'default_org';

      ingestBtnText.textContent = '⏳ Preprocessing, Encrypting & Indexing...';
      btnRunIngest.disabled = true;
      ingestStatusBox.style.display = 'none';

      try {
        let result;
        const isFileTab = tabIngestFile.classList.contains('active');

        if (isFileTab) {
          const fileInput = document.getElementById('ingest-file-input');
          if (!fileInput.files || fileInput.files.length === 0) {
            throw new Error('Please select a PDF or TXT file to upload.');
          }
          const file = fileInput.files[0];
          if (!docId) {
            docId = 'doc_' + file.name.replace(/\.[^/.]+$/, '').replace(/[^a-zA-Z0-9_-]/g, '_').toLowerCase();
          }
          const formData = new FormData();
          formData.append('file', file);
          formData.append('doc_id', docId);
          formData.append('user_id', userId);
          formData.append('org_id', orgId);
          formData.append('chunk_size', '500');
          formData.append('overlap', '50');

          const res = await fetch('/api/v1/ingest-document', {
            method: 'POST',
            body: formData,
          });
          if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'File ingestion failed');
          }
          result = await res.json();
        } else {
          const textInput = document.getElementById('ingest-text-input');
          const text = textInput.value.trim();
          if (!text) {
            throw new Error('Please enter plaintext content to ingest.');
          }

          const res = await fetch('/api/v1/ingest-text', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              text: text,
              doc_id: docId,
              user_id: userId,
              org_id: orgId,
              chunk_size: 500,
              overlap: 50,
              filename: `${docId}.txt`
            }),
          });
          if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Text ingestion failed');
          }
          result = await res.json();
        }

        // Show Success Feedback
        ingestStatusBox.style.display = 'block';
        ingestStatusBox.style.background = 'rgba(16, 185, 129, 0.12)';
        ingestStatusBox.style.borderColor = 'rgba(16, 185, 129, 0.35)';
        ingestStatusBox.style.color = '#34d399';
        ingestStatusBox.innerHTML = `
          <strong>Document Ingested Successfully</strong><br/>
          • <strong>${result.total_chunks} Chunks</strong> created<br/>
          • <strong>${result.qdrant_points_upserted} Points</strong> indexed in Qdrant.<br/>
          <div style="margin-top: 8px;">
            <button type="button" id="btn-quick-one-line" class="preset-btn">Summarize in one line</button>
          </div>
        `;

        const quickBtn = document.getElementById('btn-quick-one-line');
        if (quickBtn) {
          quickBtn.addEventListener('click', () => {
            queryInput.value = 'Summarize this document in one line.';
            queryInput.focus();
            queryInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
            if (ingestModal) ingestModal.close();
          });
        }

        // Update Workbench inputs to match ingested document
        userIdInput.value = userId;
        orgIdInput.value = orgId;
        setTenant(orgId, userId, orgId);
        loadOrganizations();
        updateWorkbenchDocCount(orgId);
      } catch (err) {
        ingestStatusBox.style.display = 'block';
        ingestStatusBox.style.background = 'rgba(239, 68, 68, 0.12)';
        ingestStatusBox.style.borderColor = 'rgba(239, 68, 68, 0.35)';
        ingestStatusBox.style.color = '#f87171';
        ingestStatusBox.innerHTML = `<strong>Ingestion Error:</strong> ${escapeHtml(err.message)}`;
      } finally {
        ingestBtnText.textContent = 'Upload & Ingest';
        btnRunIngest.disabled = false;
      }
    });
  }

  // --- Qdrant Stored Documents Inspector Modal ---
  const btnOpenOrgDocs = document.getElementById('btn-open-org-docs');
  const btnViewQdrant = document.getElementById('btn-view-qdrant');
  const qdrantModal = document.getElementById('qdrant-modal');
  const btnCloseQdrantModal = document.getElementById('btn-close-qdrant-modal');
  const qdrantModalContent = document.getElementById('qdrant-modal-content');

  async function openOrgDocumentsViewer() {
    if (!qdrantModal || !qdrantModalContent) return;
    const orgId = (orgIdInput && orgIdInput.value.trim()) || 'default_org';
    const orgDisplayName = getOrgDisplayName(orgId);
    const titleEl = document.getElementById('qdrant-modal-title');
    if (titleEl) titleEl.textContent = `Documents in ${orgDisplayName} (${orgId})`;

    qdrantModalContent.innerHTML = `<p style="color: #a5b4fc; padding: 1rem;">⏳ Loading documents for <strong>${escapeHtml(orgDisplayName)}</strong>...</p>`;
    qdrantModal.showModal();

    try {
      const res = await fetch(`/api/v1/documents/${encodeURIComponent(orgId)}`);
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'HTTP ' + res.status }));
        throw new Error(err.detail || 'Failed to fetch documents from Qdrant');
      }
      const data = await res.json();
      const documents = data.documents || [];
      const totalChunks = data.total_chunks || 0;

      if (documents.length === 0) {
        qdrantModalContent.innerHTML = `
          <div style="padding: 2rem; text-align: center; color: var(--text-secondary);">
            <h4 style="margin: 0 0 0.4rem 0; font-size: 1rem; color: var(--text-primary);">No documents stored for ${escapeHtml(orgDisplayName)}</h4>
            <p style="font-size: 0.82rem; margin-bottom: 1rem;">Upload a PDF or TXT document to this organization to start searching.</p>
            <button type="button" id="btn-modal-go-ingest" class="submit-btn" style="max-width: 200px; margin: 0 auto;">
              Upload Document
            </button>
          </div>
        `;
        const goIngestBtn = document.getElementById('btn-modal-go-ingest');
        if (goIngestBtn) {
          goIngestBtn.addEventListener('click', () => {
            qdrantModal.close();
            if (btnOpenIngestModal) btnOpenIngestModal.click();
          });
        }
        return;
      }

      let html = `
        <!-- TOP TOOLBAR: COUNTER & SEARCH FILTER -->
        <div style="margin-bottom: 1rem; padding: 0.85rem 1rem; background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.25); border-radius: 8px; display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 10px;">
          <div>
            <div style="font-weight: 600; color: #ffffff; font-size: 0.9rem;">
              📚 ${documents.length} Stored Document${documents.length === 1 ? '' : 's'} (${totalChunks} Chunks)
            </div>
            <div style="font-size: 0.72rem; color: #94a3b8; margin-top: 2px;">
              Organization: <strong>${escapeHtml(orgDisplayName)}</strong> &bull; Secured with AES-256-GCM
            </div>
          </div>
          <div style="display: flex; gap: 8px; align-items: center;">
            <input type="text" id="doc-search-filter" placeholder="Filter documents..." style="padding: 5px 10px; font-size: 0.78rem; width: 170px;" />
            <button type="button" id="btn-purge-collection" class="preset-btn" style="color: var(--color-danger);">Delete All</button>
          </div>
        </div>

        <!-- DOCUMENT CARDS LIST -->
        <div id="docs-card-list" style="display: flex; flex-direction: column; gap: 10px;">
      `;

      documents.forEach((doc, i) => {
        html += `
          <div class="org-doc-card" data-doc-title="${escapeHtml(doc.filename.toLowerCase())}" data-doc-text="${escapeHtml(doc.full_text.slice(0, 300).toLowerCase())}" style="background: var(--bg-inset); border: 1px solid var(--border-color); border-radius: 6px; padding: 12px;">
            <!-- DOCUMENT CARD HEADER -->
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px; border-bottom: 1px solid var(--border-color); padding-bottom: 8px;">
              <div>
                <div style="display: flex; align-items: center; gap: 8px;">
                  <h4 style="margin: 0; font-size: 0.92rem; color: var(--text-primary); font-weight: 600;">${escapeHtml(doc.filename)}</h4>
                  <span class="chunk-badge">Encrypted</span>
                </div>
                <div style="font-size: 0.74rem; color: var(--text-secondary); margin-top: 4px; display: flex; flex-wrap: wrap; gap: 10px;">
                  <span>Doc ID: <code>${escapeHtml(doc.doc_id)}</code></span>
                  <span>Chunks: ${doc.total_chunks}</span>
                  <span>User: <code>${escapeHtml(doc.user_id)}</code></span>
                </div>
              </div>
              <div style="display: flex; gap: 6px; align-items: center;">
                <button type="button" class="btn-query-doc preset-btn" data-query="Summarize ${escapeHtml(doc.filename)}">
                  Search Doc
                </button>
                <button type="button" class="btn-delete-doc preset-btn" data-docid="${escapeHtml(doc.doc_id)}" title="Delete document">
                  Delete
                </button>
              </div>
            </div>

            <!-- COMPLETE DOCUMENT TEXT -->
            <div style="background: var(--bg-surface); border: 1px solid var(--border-color); border-radius: 4px; padding: 10px; max-height: 200px; overflow-y: auto;">
              <div style="font-size: 0.82rem; line-height: 1.6; color: var(--text-primary); white-space: pre-wrap;">${escapeHtml(doc.full_text || '[No content text]')}</div>
            </div>

            <details style="margin-top: 8px; font-size: 0.74rem; color: var(--text-secondary);">
              <summary style="cursor: pointer; user-select: none;">View ${doc.total_chunks} chunks</summary>
              <div style="margin-top: 6px; display: flex; flex-direction: column; gap: 6px;">
                ${doc.chunks.map((chk, cIdx) => `
                  <div style="padding: 6px 10px; background: rgba(0, 0, 0, 0.35); border-radius: 4px; border: 1px solid rgba(255, 255, 255, 0.04); font-size: 0.71rem;">
                    <div style="display: flex; justify-content: space-between; color: #93c5fd; margin-bottom: 2px;">
                      <span>Chunk #${cIdx + 1} (${chk.chunk_id})</span>
                      <span>Cluster C-${chk.cluster_id}</span>
                    </div>
                    <div style="color: #cbd5e1; font-family: monospace; font-size: 0.68rem; word-break: break-all;">
                      Leaf Hash: ${escapeHtml(chk.leaf_hash?.slice(0, 32))}...
                    </div>
                  </div>
                `).join('')}
              </div>
            </details>
          </div>
        `;
      });

      html += `</div>`;
      qdrantModalContent.innerHTML = html;

      // 1. Live document search filter
      const filterInput = document.getElementById('doc-search-filter');
      if (filterInput) {
        filterInput.addEventListener('input', () => {
          const q = filterInput.value.toLowerCase().trim();
          const docCards = qdrantModalContent.querySelectorAll('.org-doc-card');
          docCards.forEach(card => {
            const title = card.dataset.docTitle || '';
            const text = card.dataset.docText || '';
            const match = !q || title.includes(q) || text.includes(q);
            card.style.display = match ? 'block' : 'none';
          });
        });
      }

      // 2. Quick search buttons on documents
      const queryBtns = qdrantModalContent.querySelectorAll('.btn-query-doc');
      queryBtns.forEach(btn => {
        btn.addEventListener('click', () => {
          const q = btn.dataset.query;
          if (q) {
            queryInput.value = q;
            queryInput.focus();
          }
          qdrantModal.close();
        });
      });

      // 3. Individual document delete buttons
      const deleteBtns = qdrantModalContent.querySelectorAll('.btn-delete-doc');
      deleteBtns.forEach(btn => {
        btn.addEventListener('click', async () => {
          const did = btn.dataset.docid;
          if (confirm(`Are you sure you want to delete document '${did}'?`)) {
            btn.textContent = '⏳';
            await fetch(`/api/v1/documents/${encodeURIComponent(orgId)}/${encodeURIComponent(did)}`, { method: 'DELETE' });
            openOrgDocumentsViewer();
          }
        });
      });

      // 4. Delete all points button
      const btnPurge = document.getElementById('btn-purge-collection');
      if (btnPurge) {
        btnPurge.addEventListener('click', async () => {
          if (confirm(`Are you sure you want to delete all stored documents in ${data.collection_name}?`)) {
            btnPurge.textContent = '⏳ Deleting...';
            await fetch(`/api/v1/collections/${encodeURIComponent(orgId)}`, { method: 'DELETE' });
            openOrgDocumentsViewer();
          }
        });
      }
    } catch (err) {
      qdrantModalContent.innerHTML = `
        <div style="padding: 1rem; color: #f87171; background: rgba(239, 68, 68, 0.1); border-radius: 8px;">
          <strong>Error loading organization documents:</strong> ${escapeHtml(err.message)}
        </div>
      `;
    }
  }

  if (btnOpenOrgDocs) {
    btnOpenOrgDocs.addEventListener('click', openOrgDocumentsViewer);
  }

  if (btnViewQdrant) {
    btnViewQdrant.addEventListener('click', openOrgDocumentsViewer);
  }

  if (btnCloseQdrantModal && qdrantModal) {
    btnCloseQdrantModal.addEventListener('click', () => {
      qdrantModal.close();
    });
    qdrantModal.addEventListener('click', (e) => {
      if (e.target === qdrantModal) {
        qdrantModal.close();
      }
    });
  }
});
