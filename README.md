# Vanity SOL GPU

A fast Solana vanity address generator that runs entirely in your browser.

## Features

- **GPU key generation** - SHA-512, ed25519 scalar multiplication and Base58 matching all run in a WebGPU compute shader (~33M keys/s on an RTX 40-series card, versus ~4K keys/s per CPU core)
- **CPU fallback** - Web Workers search with @noble/curves when WebGPU is unavailable or the GPU fails its self-test
- **Exact odds** - The difficulty and time estimate come from the exact share of keys a pattern matches (an address can only start with a letter after "J" if it is 43 characters long, so such prefixes are rarer than 58^n)
- **Privacy First** - All computations run locally in your browser, no server communication
- **Custom Prefix/Suffix** - Find addresses starting or ending with your desired Base58 characters
- **Wallet + Program ID Targets** - Scan either wallet addresses or program IDs
- **Keypair Export** - Download Solana-compatible keypair JSON files

## Requirements

- A modern browser. WebGPU (Chrome, Edge, recent Safari and Firefox) enables the GPU path; others use the CPU.

## Usage

1. Choose target: wallet address or program ID
2. Enter your desired prefix and/or suffix (Base58 characters)
3. Click Generate and wait for a match
4. Once found, reveal the secret key or download keypair JSON

## Security

- Disconnect from the internet before generating for maximum security
- The production build ships a Content Security Policy that forbids network connections (`connect-src 'none'`), as a `<meta>` tag and a Vercel header
- Every key the GPU reports is re-derived on the CPU with @noble/curves before it is shown
- Never share your secret key
- Always verify the address matches your requirements before use

## Development

```bash
# Install dependencies
npm install

# Start dev server
npm run dev

# Build for production
npm run build

# Run unit tests
npm run test:unit

# Run browser E2E tests
npm run test:e2e

# Self-test and benchmark the GPU kernel in the installed Google Chrome
npm run test:gpu                              # default adapter
npm run test:gpu -- --adapter-luid=0,1234     # pick a GPU on Windows (LUIDs: chrome://gpu)
npm run test:gpu -- --swiftshader             # CPU-emulated WebGPU (slow)

# Regenerate favicon.svg, the PNG icons and the social cards from src/assets/icon.svg,
# src/assets/icon-small.svg (favicon cut) and the card template in scripts/build-images.mjs
npm run images

# Run full quality gate (build + unit + e2e)
npm run check
```

## Testing

- Unit tests use `vitest` and cover core Base58/keys logic. The GPU kernel's field arithmetic and SHA-512 are generated for WGSL and as a JavaScript twin (`src/gpu/*Gen.ts`), so the tests check the exact arithmetic the shader runs against BigInt, node:crypto and @noble, and a step-by-step model of the kernel and its matcher.
- `npm run test:gpu` runs the kernel's known-answer self-test (the same one the app runs before using a GPU) on real hardware and reports keys/s.
- E2E tests use `@playwright/test` and exercise real UI flows in Chromium.
- CI runs `build`, `test:unit`, and `test:e2e` on every push/PR.
- For deterministic test runs, the dev server is started with `VITE_FORCE_BACKEND=cpu`.

## Tech Stack

- TypeScript
- Vite
- Web Workers
- WebGPU
- @noble/curves (ed25519)

## Notes

- Before using a GPU, the app runs a known-answer test on it: every key from a fixed dispatch is re-derived with @noble, and prefix/suffix patterns must report exactly the matching keys.
- If WebGPU is unavailable, the self-test fails or the device is lost mid-search, the app falls back to CPU search.
- You can force a backend for troubleshooting with `VITE_FORCE_BACKEND=cpu` or `VITE_FORCE_BACKEND=gpu`.

## License

MIT
