// Build the headless service bundle with esbuild's JS API so we can teach it
// Vite's `?raw` text-import convention (used by proxyServer.ts for the portal/
// admin HTML templates). electron-vite handles `?raw` natively; esbuild needs
// the shared rawPlugin to do the same.
import esbuild from 'esbuild'
import { rawPlugin } from './esbuild-plugins.mjs'

await esbuild.build({
  entryPoints: ['src/service/index.ts'],
  bundle: true,
  platform: 'node',
  format: 'cjs',
  packages: 'external',
  outfile: 'out/service/index.cjs',
  plugins: [rawPlugin]
})

console.log('[build:service] done → out/service/index.cjs')
