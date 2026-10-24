import { defineConfig, type Plugin } from 'vite'

// The page generates private keys, so production forbids network connections
// (fetch/XHR/WebSocket: connect-src 'none') and any script, style or worker
// that isn't one of our own files. The <meta> tag covers the page wherever it's
// hosted; workers only get a policy from HTTP headers, so vercel.json sends the
// same policy as a header (tests/unit/csp.test.ts keeps the two in sync). CSP
// can't stop top-level navigation, so this narrows exfiltration paths rather
// than closing every one. Dev builds skip it (Vite's HMR needs inline scripts
// and a socket).
export const CONTENT_SECURITY_POLICY = [
  "default-src 'none'",
  "script-src 'self'",
  "worker-src 'self'",
  "style-src 'self'",
  "img-src 'self'",
  "manifest-src 'self'",
  "connect-src 'none'",
  "base-uri 'none'",
  "form-action 'none'",
  "object-src 'none'",
].join('; ')

// frame-ancestors only works as a header, not in <meta>.
export const CONTENT_SECURITY_POLICY_HEADER = CONTENT_SECURITY_POLICY + "; frame-ancestors 'none'"

function contentSecurityPolicy(): Plugin {
  return {
    name: 'content-security-policy',
    apply: 'build',
    transformIndexHtml() {
      return [{
        tag: 'meta',
        attrs: { 'http-equiv': 'Content-Security-Policy', content: CONTENT_SECURITY_POLICY },
        injectTo: 'head-prepend',
      }]
    },
  }
}

export default defineConfig({
  plugins: [contentSecurityPolicy()],
  build: {
    target: 'esnext',
  },
  // `vite preview` serves the production build with the same header as Vercel.
  preview: {
    headers: { 'Content-Security-Policy': CONTENT_SECURITY_POLICY_HEADER },
  },
  worker: {
    format: 'es',
  },
})
