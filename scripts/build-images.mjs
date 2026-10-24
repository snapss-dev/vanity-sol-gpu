#!/usr/bin/env node
// Regenerate every icon and social-card image from source, rendered with the
// installed Google Chrome so the PNGs match what browsers draw.
//   src/assets/icon-small.svg -> favicon.svg, favicon-16x16.png, favicon-32x32.png
//   src/assets/icon.svg       -> apple-touch-icon.png, android-chrome-192/512.png (and the header logo)
//   socialCard() below        -> og-image.png, twitter-card.png (1200 x 630)
//   npm run images
import { readFileSync, writeFileSync } from 'node:fs'
import { chromium } from 'playwright-core'

const root = new URL('../', import.meta.url)
const read = (p) => readFileSync(new URL(p, root), 'utf8')
const small = read('src/assets/icon-small.svg')
const icon = read('src/assets/icon.svg')
// iOS masks home-screen icons itself; transparent rounded corners would show as black.
const fullBleed = icon.replace('class="tile" width="64" height="64" rx="14"', 'class="tile" width="64" height="64" rx="0"')
if (fullBleed === icon) throw new Error('icon.svg: tile rect not found')

const BASE58 = /^[1-9A-HJ-NP-Za-km-z]+$/
const svgImg = (svg, size) =>
  `<img src="data:image/svg+xml;base64,${Buffer.from(svg).toString('base64')}" width="${size}" height="${size}" style="display:block">`

// Social cards are laid out with HTML so nothing depends on guessed text widths.
function socialCard({ tagline, label, pre, mid, suf }) {
  const address = pre + mid + suf
  if (address.length !== 44 || !BASE58.test(address)) throw new Error(`not a 44-char base58 address: ${address}`)
  return `<!doctype html><html><head><meta charset="utf-8"><style>
  * { box-sizing: border-box; margin: 0; }
  html, body { width: 1200px; height: 630px; }
  body {
    font-family: 'Segoe UI', system-ui, -apple-system, Roboto, sans-serif; color: #f0f1f5;
    background:
      radial-gradient(ellipse 70% 60% at 10% 0%, rgba(153, 69, 255, .34), transparent 60%),
      radial-gradient(ellipse 60% 55% at 100% 100%, rgba(20, 241, 149, .14), transparent 60%),
      #06070d;
    padding: 56px 72px; display: flex; flex-direction: column; justify-content: center; gap: 44px;
  }
  .brand { display: flex; align-items: center; gap: 30px; }
  .brand .icon { filter: drop-shadow(0 14px 34px rgba(153, 69, 255, .45)); }
  h1 { font-size: 76px; font-weight: 700; letter-spacing: -.025em; line-height: 1; }
  .tagline { margin-top: 14px; font-size: 30px; color: rgba(240, 241, 245, .7); }
  .result {
    padding: 26px 30px; border-radius: 20px;
    background: rgba(18, 20, 30, .85); border: 2px solid rgba(20, 241, 149, .5);
    box-shadow: 0 0 0 1px rgba(20, 241, 149, .15), 0 18px 50px rgba(20, 241, 149, .1);
  }
  .label { display: flex; justify-content: space-between; align-items: center; font-size: 19px;
    letter-spacing: .1em; text-transform: uppercase; color: rgba(240, 241, 245, .5); }
  .verified { letter-spacing: 0; text-transform: none; font-size: 19px; color: #14f195;
    padding: 5px 14px; border-radius: 999px; background: rgba(20, 241, 149, .1); border: 1px solid rgba(20, 241, 149, .35); }
  .addr { margin-top: 12px; font-family: 'Cascadia Mono', Consolas, ui-monospace, monospace; font-size: 36px;
    white-space: nowrap; color: rgba(240, 241, 245, .82); }
  .addr b { color: #14f195; font-weight: 700; }
  .chips { display: flex; gap: 16px; }
  .chip { font-size: 24px; font-weight: 600; padding: 14px 22px; border-radius: 14px; white-space: nowrap;
    background: rgba(153, 69, 255, .12); border: 1px solid rgba(186, 140, 255, .38); color: #ece2ff; }
</style></head><body>
  <div class="brand">
    <div class="icon">${svgImg(icon, 132)}</div>
    <div><h1>Vanity SOL GPU</h1><div class="tagline">${tagline}</div></div>
  </div>
  <div class="result">
    <div class="label"><span>${label}</span><span class="verified">✓ Verified locally</span></div>
    <div class="addr"><b>${pre}</b>${mid}<b>${suf}</b></div>
  </div>
  <div class="chips">
    <span class="chip">Runs on your GPU</span>
    <span class="chip">Keys never leave your browser</span>
    <span class="chip">Wallets &amp; program IDs</span>
  </div>
</body></html>`
}

const page = (html, size) => ({ html, width: size[0], height: size[1] })
const iconPage = (svg, s) => page(`<style>html,body{margin:0;background:transparent}</style>${svgImg(svg, s)}`, [s, s])
writeFileSync(new URL('public/favicon.svg', root), small)

const targets = [
  ['public/favicon-16x16.png', iconPage(small, 16)],
  ['public/favicon-32x32.png', iconPage(small, 32)],
  ['public/apple-touch-icon.png', iconPage(fullBleed, 180)],
  ['public/android-chrome-192x192.png', iconPage(icon, 192)],
  ['public/android-chrome-512x512.png', iconPage(icon, 512)],
  ['public/og-image.png', page(socialCard({
    tagline: 'Solana vanity addresses, generated on your GPU',
    label: 'Wallet address', pre: 'Moon', mid: '7xKq9VbN3hR2mTfW8cYdP4sJ6gZ5aEuX1nBv', suf: 'pump',
  }), [1200, 630])],
  ['public/twitter-card.png', page(socialCard({
    tagline: 'Vanity wallets and program IDs, in your browser',
    label: 'Program ID', pre: 'Dex', mid: '4Gm8TzQw2Ys7KcRb9NhVf3XpJ6dEuA5ka1WtL', suf: 'Swap',
  }), [1200, 630])],
]

const browser = await chromium.launch({ channel: 'chrome', headless: true })
try {
  for (const [file, { html, width, height }] of targets) {
    const tab = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 1 })
    await tab.setContent(html)
    await tab.evaluate(() => Promise.all([...document.images].map(img => img.decode())).then(() => document.fonts.ready))
    writeFileSync(new URL(file, root), await tab.screenshot({ omitBackground: true }))
    await tab.close()
    console.log(`${file} (${width}x${height})`)
  }
} finally {
  await browser.close()
}
